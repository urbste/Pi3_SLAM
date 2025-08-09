"""
Offline chunk creation pipeline for Pi3SLAM.

This module provides the `OfflineChunkCreator` class that:
- runs Pi3 over image sequences in chunks
- optionally scales metric depth using MoGe
- extracts keypoints (ALIKED or grid)
- interpolates 3D keypoint positions
- saves per-chunk results to disk (one file per chunk)

Saved chunk files contain the minimal data required to build reconstructions later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import json

import torch
import time

from torch.utils.data import DataLoader

from pi3.models.pi3 import Pi3
from pi3.utils.geometry import depth_edge

from utils.image_utils import calculate_target_size
from utils.keypoint_extraction import create_keypoint_extractor
from utils.camera_estimation import estimate_camera_parameters_single_chunk, print_camera_parameters_summary
from datasets.image_datasets import ChunkImageDataset


@dataclass
class OfflineCreatorConfig:
    model_path: str
    output_dir: str
    chunk_length: int = 100
    overlap: int = 10
    device: str = "cuda"
    do_metric_depth: bool = True
    keypoint_type: str = "aliked"  # 'aliked' | 'grid' | 'none'
    max_num_keypoints: int = 512
    keypoint_detection_threshold: float = 0.005
    estimate_camera_params: bool = True
    num_loader_workers: int = 2
    pin_memory: bool = True

torch._dynamo.config.capture_scalar_outputs = True

class OfflineChunkCreator:
    """Create per-chunk PI3 results (with MoGe scaling and keypoints) and save to disk."""

    def __init__(self, config: OfflineCreatorConfig):
        self.config = config

        os.makedirs(self.config.output_dir, exist_ok=True)
        self.chunks_dir = os.path.join(self.config.output_dir, "chunks")
        os.makedirs(self.chunks_dir, exist_ok=True)

        # Load Pi3
        self.model: Pi3 = Pi3.from_pretrained(self.config.model_path).to(self.config.device).eval()

        self.model = torch.compile(self.model)

        # Load MoGe if requested
        self.moge_model = None
        if self.config.do_metric_depth:
            try:
                from moge.model.v2 import MoGeModel
                self.moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vits-normal").to("cuda").eval()
                self.moge_model = torch.compile(self.moge_model)
                print("   MoGe loaded for metric scaling")
            except Exception as e:
                print(f"⚠️  Failed to initialize MoGe: {e}. Continuing without metric depth.")
                self.moge_model = None

        # Keypoint extractor
        self.keypoint_extractor = None
        if self.config.keypoint_type and self.config.keypoint_type.lower() != "none":
            try:
                self.keypoint_extractor = create_keypoint_extractor(
                    keypoint_type=self.config.keypoint_type,
                    max_num_keypoints=self.config.max_num_keypoints,
                    detection_threshold=self.config.keypoint_detection_threshold,
                    device=self.config.device,
                )
                print(f"   Keypoint extractor: {self.config.keypoint_type}")
            except Exception as e:
                print(f"⚠️  Failed to initialize keypoint extractor: {e}. Continuing without keypoints.")
                self.keypoint_extractor = None

        # Internal state
        self.target_size: Optional[Tuple[int, int]] = None  # (H, W)

    @staticmethod
    def _compute_masks(pi3_result: Dict[str, torch.Tensor]) -> torch.Tensor:
        masks = torch.sigmoid(pi3_result['conf'][..., 0]) > 0.1
        non_edge = ~depth_edge(pi3_result['local_points'][..., 2], rtol=0.03)
        masks = torch.logical_and(masks, non_edge)
        return masks

    @staticmethod
    def _get_scale_factor_for_pi3(moge_metric_depth: torch.Tensor, pi3_metric_depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        moge_metric_depth = moge_metric_depth[mask]
        pi3_metric_depth = pi3_metric_depth[mask]
        scale_factor = moge_metric_depth / pi3_metric_depth
        scale_factor = scale_factor.median()
        return scale_factor

    @staticmethod
    def _interpolate_world_points_for_keypoints(result_cpu_dense: Dict, keypoints: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Interpolate world/local points, confidences, and masks at keypoint locations.

        Args:
            result_cpu_dense: dict containing dense 'points', 'local_points', 'conf', 'masks', 'images'
            keypoints: Tensor (N, K, 2)
        Returns:
            Dict with 'points', 'local_points', 'conf', 'masks' keyed by frame index shape (N, K, C)
        """
        H, W = result_cpu_dense['images'].shape[-2:]
        # Normalize to [-1, 1]
        grid_x = (keypoints[:, :, 0] / (W - 1)) * 2 - 1
        grid_y = (keypoints[:, :, 1] / (H - 1)) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1).cpu()  # (1, 1, N, 2)

        points = torch.nn.functional.grid_sample(
            result_cpu_dense['points'].permute(0, 3, 1, 2), grid, mode="bilinear", align_corners=False, padding_mode="border")
        local_points = torch.nn.functional.grid_sample(
            result_cpu_dense['local_points'].permute(0, 3, 1, 2), grid, mode="bilinear", align_corners=False, padding_mode="border")
        confidences = torch.nn.functional.grid_sample(
            result_cpu_dense['conf'].permute(0, 3, 1, 2), grid, mode="nearest", align_corners=False, padding_mode="border")
        masks = torch.nn.functional.grid_sample(
            result_cpu_dense['masks'].unsqueeze(1).float(), grid, mode="nearest", align_corners=False, padding_mode="border")

        return {
            'points': points.squeeze().permute(0, 2, 1),
            'local_points': local_points.squeeze().permute(0, 2, 1),
            'conf': confidences.squeeze(1).permute(0, 2, 1),
            'masks': masks.squeeze(1).permute(0, 2, 1).bool(),
        }

    def _process_single_chunk(self, chunk_images: torch.Tensor, chunk_paths: List[str]) -> Dict:
        """Run Pi3, optional MoGe scaling, keypoint extraction and interpolation for one chunk."""
        assert chunk_images.ndim == 5, "Expected (B=1, N, C, H, W) tensor for chunk images"

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        num_frames = int(chunk_images.shape[1])
        with torch.no_grad():
            t0_inf = time.time()
            with torch.amp.autocast('cuda', dtype=dtype) if self.config.device.startswith('cuda') else torch.autocast(device_type='cpu', dtype=torch.float32):
                pi3_result = self.model(chunk_images.to(self.config.device))
            dt_inf = max(1e-6, time.time() - t0_inf)
            fps = num_frames / dt_inf if num_frames > 0 else 0.0
            print(f"   ⏱️ Inference: {dt_inf:.3f}s for {num_frames} frames  ->  {fps:.2f} FPS")
            # stash metrics for aggregation
            _metrics = {'infer_s': float(dt_inf), 'num_frames': int(num_frames), 'fps': float(fps)}

        # Compute masks
        masks = self._compute_masks(pi3_result)[0]

        # Optional MoGe metric scaling
        if self.moge_model is not None:
            with torch.amp.autocast('cuda', dtype=dtype):
                moge_metric_depth = self.moge_model.infer(chunk_images[0, 0].to("cuda"))["depth"]
            pi3_metric_depth = pi3_result['local_points'][0, 0][..., 2]
            mask0 = masks[0]
            scale_factor = self._get_scale_factor_for_pi3(moge_metric_depth, pi3_metric_depth, mask0)

            pi3_result["local_points"] = pi3_result["local_points"] * scale_factor
            pi3_result["points"] = pi3_result["points"] * scale_factor
            pi3_result["camera_poses"][:, :, :3, 3] = pi3_result["camera_poses"][:, :, :3, 3] * scale_factor
            pi3_result["camera_poses_cw"] = torch.linalg.inv(pi3_result["camera_poses"])

        # Optional intrinsics estimation
        camera_params = None
        if self.config.estimate_camera_params:
            try:
                camera_params = estimate_camera_parameters_single_chunk(pi3_result)
            except Exception as e:
                print(f"⚠️  Camera parameter estimation failed: {e}")
                camera_params = None

        # CPU result with dense maps needed for interpolation
        cpu_result = {
            'points': pi3_result['points'][0].cpu(),
            'local_points': pi3_result['local_points'][0].cpu(),
            'camera_poses': pi3_result['camera_poses'][0].cpu(),
            'conf': pi3_result['conf'][0].cpu(),
            'masks': masks.cpu(),
            'image_paths': chunk_paths,
            'images': chunk_images.squeeze(0).cpu(),
            '_metrics': _metrics,
        }

        if camera_params is not None:
            cpu_result['camera_params'] = {k: v.cpu() for k, v in camera_params.items()}

        # Provide size info for downstream consumers
        if self.target_size is not None:
            cpu_result['original_width'] = self.target_size[1]
            cpu_result['original_height'] = self.target_size[0]

        # Keypoint extraction (if enabled)
        if self.keypoint_extractor is not None:
            try:
                kp_res = self.keypoint_extractor.extract_with_colors(chunk_images)
                # Interpolate 3D/world at keypoints using dense maps
                interp = self._interpolate_world_points_for_keypoints(cpu_result, kp_res['keypoints'])

                # Replace dense maps with per-keypoint results for storage efficiency
                cpu_result['points'] = interp['points'].to(torch.float16)
                cpu_result['local_points'] = interp['local_points'].to(torch.float16)
                cpu_result['conf'] = interp['conf'].to(torch.float16)
                cpu_result['masks'] = interp['masks']
                # Save keypoints so reconstructor can add track observations and project to neighbors
                cpu_result['keypoints'] = kp_res['keypoints'].to(torch.float16).cpu()
                if kp_res.get('descriptors') is not None:
                    cpu_result['descriptors'] = kp_res['descriptors'].to(torch.float16)
                if kp_res.get('scores') is not None:
                    cpu_result['scores'] = kp_res['scores'].to(torch.float16)
                cpu_result['colors'] = kp_res['colors'].to(torch.float16)
            except Exception as e:
                print(f"⚠️  Keypoint extraction failed: {e}")

        # Add intrinsics if available
        if 'camera_params' in cpu_result and cpu_result['camera_params'] is not None:
            cpu_result['intrinsics'] = cpu_result['camera_params'].get('intrinsics', None)

        # Drop raw images before saving to reduce size (not needed for reconstruction)
        try:
            if 'images' in cpu_result:
                del cpu_result['images']
        except Exception:
            pass

        return cpu_result

    def process_and_save(self, image_paths: List[str]) -> List[str]:
        """Process the input images in chunks and save each chunk to disk.

        Returns list of saved file paths.
        """
        if not image_paths:
            raise ValueError("image_paths is empty")

        # Determine target size once from first image path
        self.target_size = calculate_target_size(image_paths[0], pixel_limit=255000 // 2)
        print(f"Target size: {self.target_size}")

        dataset = ChunkImageDataset(
            image_paths=image_paths,
            chunk_length=self.config.chunk_length,
            overlap=self.config.overlap,
            target_size=self.target_size,
            device='cpu',
            undistortion_maps=None,
        )

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_loader_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_loader_workers > 0,
            prefetch_factor=1,
        )

        saved_files: List[str] = []
        manifest: List[Dict] = []

        print(f"🔄 Processing {len(dataset)} chunks...")
        infer_times: List[float] = []
        infer_frames: List[int] = []
        per_chunk_fps: List[float] = []
        for chunk_idx, batch in enumerate(loader):
            start_idx = int(batch['start_idx'].item())
            end_idx = int(batch['end_idx'].item())
            chunk_images: torch.Tensor = batch['chunk']  # (1, N, C, H, W)
            chunk_paths: List[str] = batch['chunk_paths'][0]

            print(f"📦 Chunk {chunk_idx+1}/{len(dataset)}: frames {start_idx+1}-{end_idx}")
            chunk_result = self._process_single_chunk(chunk_images, chunk_paths)
            # collect metrics for totals
            m = chunk_result.get('_metrics', {})
            if m:
                infer_times.append(float(m.get('infer_s', 0.0)))
                infer_frames.append(int(m.get('num_frames', 0)))
                per_chunk_fps.append(float(m.get('fps', 0.0)))

            # Persist to disk (Torch .pt)
            out_name = f"chunk_{chunk_idx:06d}.pt"
            out_path = os.path.join(self.chunks_dir, out_name)
            # Add small metadata
            chunk_result['chunk_index'] = chunk_idx
            chunk_result['start_idx'] = start_idx
            chunk_result['end_idx'] = end_idx
            try:
                torch.save(chunk_result, out_path)
                saved_files.append(out_path)
                manifest.append({
                    'chunk_index': chunk_idx,
                    'file': out_name,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'num_frames': len(chunk_paths),
                    'image_paths': chunk_paths,
                })
                print(f"   💾 Saved: {out_path}")
            except Exception as e:
                print(f"❌ Failed to save chunk {chunk_idx}: {e}")

        # Summaries
        try:
            total_time = sum(infer_times) if infer_times else 0.0
            total_frames = sum(infer_frames) if infer_frames else 0
            overall_fps = (total_frames / total_time) if total_time > 0 else 0.0
            # steady-state: only full-size chunks
            steady_fps_vals = [f for f, n in zip(per_chunk_fps, infer_frames) if n == self.config.chunk_length]
            steady_fps = (sorted(steady_fps_vals)[len(steady_fps_vals)//2] if steady_fps_vals else 0.0)
            print(f"\n⏱️ Overall inference: {total_frames} frames in {total_time:.3f}s  ->  {overall_fps:.2f} FPS (weighted)")
            if steady_fps_vals:
                print(f"   Steady-state FPS (full {self.config.chunk_length}-frame chunks, median): {steady_fps:.2f} FPS")
        except Exception:
            pass

        # Write manifest JSON for convenience
        try:
            manifest_path = os.path.join(self.config.output_dir, "chunks_manifest.json")
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            print(f"📝 Manifest written: {manifest_path}")
        except Exception as e:
            print(f"⚠️  Failed to write manifest: {e}")

        # Write chunk metadata (for automatic reconstructor config)
        try:
            metadata = {
                'chunk_length': int(self.config.chunk_length),
                'overlap': int(self.config.overlap),
                'target_size': list(self.target_size) if self.target_size is not None else None,
            }
            meta_path = os.path.join(self.config.output_dir, "chunk_metadata.json")
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"📝 Chunk metadata written: {meta_path}")
        except Exception as e:
            print(f"⚠️  Failed to write chunk metadata: {e}")

        print(f"✅ Completed. Saved {len(saved_files)} chunks to {self.chunks_dir}")
        return saved_files



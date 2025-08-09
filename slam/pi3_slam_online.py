"""
Pi3SLAM Online model with modular components.
"""

import torch
import numpy as np
import time
import multiprocessing as mp
import os
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader

from pi3.models.pi3 import Pi3
from pi3.utils.basic import write_ply
from pi3.utils.geometry import depth_edge

# Import our modular components
from utils.image_utils import calculate_target_size
from utils.timestamp_utils import extract_timestamps_from_paths
from utils.camera_estimation import estimate_camera_parameters_single_chunk, print_camera_parameters_summary
from utils.keypoint_extraction import create_keypoint_extractor
from utils.chunk_reconstruction import ChunkPTRecon
from utils.reconstruction_alignment import (
    create_view_graph_matches, 
    align_and_refine_reconstructions
)
from utils.telemetry_converter import TelemetryImporter
import pytheia as pt
from datasets.image_datasets import ChunkImageDataset
from visualization.rerun_visualizer import visualization_process


class Pi3SLAMOnline:
    """
    Pi3SLAM with online processing and visualization.
    Processes long sequences in chunks with real-time visualization.
    """
    
    def __init__(self, model: Pi3, chunk_length: int = 100, overlap: int = 10, 
                 device: str = 'cuda', conf_threshold: float = 0.5, 
                 undistortion_maps=None, cam_scale: float = 1.0, visualization_port: int = 9090, 
                 estimate_camera_params: bool = False, 
                 keypoint_type: str = 'aliked', max_num_keypoints: int = 512, 
                 keypoint_detection_threshold: float = 0.005,
                 save_chunk_reconstructions: bool = False,
                 max_observations_per_track: int = 5, do_metric_depth: bool = False,
                 save_debug_projections: bool = False):
        """
        Initialize Pi3SLAM Online with visualization.
        
        Args:
            model: Pi3 model instance
            chunk_length: Number of frames per chunk
            overlap: Number of overlapping frames between chunks
            device: Device to run on ('cuda' or 'cpu')
            conf_threshold: Confidence threshold for point filtering
            undistortion_maps: Undistortion maps for camera calibration
            cam_scale: Camera scale factor
            visualization_port: Port for visualization
            estimate_camera_params: Whether to estimate camera parameters
            keypoint_type: Type of keypoint extraction ('aliked', 'grid', or 'none')
            max_num_keypoints: Maximum number of keypoints to extract
            keypoint_detection_threshold: Detection threshold for ALIKED keypoints
            save_chunk_reconstructions: Whether to save chunk reconstructions to disk
            max_observations_per_track: Maximum observations per track
        """
        self.model = model
        self.chunk_length = chunk_length
        self.overlap = overlap
        self.device = device
        self.conf_threshold = conf_threshold
        self.undistortion_maps = undistortion_maps
        self.cam_scale = cam_scale
        self.pixel_limit = 255000
        self.visualization_port = visualization_port
        self.estimate_camera_params = estimate_camera_params
        self.keypoint_type = keypoint_type.lower()
        self.save_chunk_reconstructions = save_chunk_reconstructions
        self.max_observations_per_track = max_observations_per_track
        self.do_metric_depth = do_metric_depth
        self.save_debug_projections = save_debug_projections
        # Initialize keypoint extractor if enabled
        self.keypoint_extractor = None
        if self.keypoint_type != 'none':
            try:
                if self.keypoint_type == 'aliked':
                    self.keypoint_extractor = create_keypoint_extractor(
                        keypoint_type='aliked',
                        max_num_keypoints=max_num_keypoints,
                        detection_threshold=keypoint_detection_threshold,
                        device=device
                    )
                    print(f"🔍 ALIKED keypoint extractor initialized with {max_num_keypoints} max keypoints")
                elif self.keypoint_type == 'grid':
                    self.keypoint_extractor = create_keypoint_extractor(
                        keypoint_type='grid',
                        max_num_keypoints=max_num_keypoints,
                        device=device
                    )
                    print(f"🔍 Grid-based keypoint extractor initialized with {max_num_keypoints} max keypoints (spacing calculated automatically)")
                else:
                    print(f"⚠️  Unknown keypoint type: {keypoint_type}, disabling keypoint extraction")
                    self.keypoint_type = 'none'
            except Exception as e:
                print(f"⚠️  Warning: Failed to initialize keypoint extractor ({keypoint_type}): {e}")
                print("   Continuing without keypoint extraction...")
                self.keypoint_type = 'none'
                self.keypoint_extractor = None
        else:
            print("🔍 Keypoint extraction disabled")
        
        # Initialize chunk reconstruction if enabled
        self.chunk_reconstructor = ChunkPTRecon()
        
        # Pytheia reconstructions storage (in-memory)
        self.chunk_reconstructions = []
        
        # Reconstruction alignment settings
        self.save_transformed_reconstructions = False  # Save PLY files of transformed reconstructions
        self.save_debug_reconstructions = False  # Save debug files for all reconstruction stages
        
        # Camera trajectory tracking
        self.full_camera_trajectory = []
        self.full_camera_orientations = []
        self.chunk_end_poses = []
        
        # Timestamp tracking
        self.timestamps = []
        
        # Configuration for correspondence search
        self.correspondence_subsample_factor = 1 if self.keypoint_type != 'none' else 10
        
        # Visualization
        self.vis_running = False
        self.vis_process = None
        self.vis_queue = None
        self.current_chunk_result = None
        self.max_points_visualization = 50000
        self.update_interval = 0.1
        self.visualization_subsample_ratio = 0.1
        
        # Timing statistics
        self.timing_stats: Dict[str, Dict[str, float]] = {}

        # Background loading
        self.loader_running = False
        self.loader_process = None
        self.loader_queue = None
        self.target_size = None
        
        # MoGe model
        self.moge_model = None
        if self.do_metric_depth:
            print("   Loading MoGe model...")
            from moge.model.v2 import MoGeModel
            self.moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda").eval()

        print(f"🚀 Pi3SLAM Online initialized with chunk_length={chunk_length}, overlap={overlap}")
        print(f"   Device: {device}, Confidence threshold: {conf_threshold}")
        print(f"   Camera scale: {cam_scale}")
        if undistortion_maps:
            print(f"   Undistortion: Enabled")
    
    def _extract_points_colors_from_reconstructions(self, latest_only: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Collect 3D points and colors from PyTheia reconstructions.
        
        Args:
            latest_only: If True, only extract from the latest reconstruction. If False, extract from all reconstructions.
        """
        if not self.chunk_reconstructions:
            return np.array([]), np.array([])

        all_points: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []
        # Determine which reconstructions to process
        reconstructions_to_process = [self.chunk_reconstructions[-1]] if latest_only else self.chunk_reconstructions
        
        for recon in reconstructions_to_process:
            if recon is None:
                continue
            
            for track_id in recon.TrackIds():
                track = recon.Track(track_id)
                p = track.Point()
                p3 = np.array(p[:3]/p[3], dtype=np.float32)
                all_points.append(p3)
                
                c = np.array(track.Color(), dtype=np.float32)
                all_colors.append(c)

        if not all_points:
            return np.array([]), np.array([])

        points_arr = np.asarray(all_points, dtype=np.float32)
        colors_arr = np.asarray(all_colors, dtype=np.float32)
        # Normalize colors if in 0-255 range
        if colors_arr.size > 0 and colors_arr.max() > 1.0:
            colors_arr = colors_arr / 255.0
        return points_arr, colors_arr

    def _extract_camera_positions_from_reconstructions(self, latest_only: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Collect camera positions and orientations from PyTheia reconstructions.
        
        Args:
            latest_only: If True, only extract from the latest reconstruction. If False, extract from all reconstructions.
        """
        if not self.chunk_reconstructions:
            return np.array([]), np.array([])

        positions: List[np.ndarray] = []
        orientations: List[np.ndarray] = []

        # Determine which reconstructions to process
        reconstructions_to_process = [self.chunk_reconstructions[-1]] if latest_only else self.chunk_reconstructions
        
        for recon in reconstructions_to_process:
            if recon is None:
                continue
        
            for vid in recon.ViewIds():
                view = recon.View(vid)
                if not view.IsEstimated() :
                    continue
                cam = view.Camera()
                positions.append(np.array(cam.GetPosition(), dtype=np.float32))
                R = cam.GetOrientationAsRotationMatrix().T
                orientations.append(np.array(R, dtype=np.float32))

        return (
            np.asarray(positions, dtype=np.float32) if positions else np.array([]),
            np.asarray(orientations, dtype=np.float32) if orientations else np.array([]),
        )

    def _extract_camera_poses_with_names(self, latest_only: bool = False) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Collect (view_name, position, orientation) tuples from reconstructions.

        Args:
            latest_only: If True, only extract from the latest reconstruction.
        Returns:
            List of tuples (name, position(3,), orientation(3x3)) preserving encounter order.
        """
        if not self.chunk_reconstructions:
            return []

        reconstructions_to_process = [self.chunk_reconstructions[-1]] if latest_only else self.chunk_reconstructions
        poses_with_names: List[Tuple[str, np.ndarray, np.ndarray]] = []

        for recon in reconstructions_to_process:
            if recon is None:
                continue
            for vid in sorted(recon.ViewIds()):
                view = recon.View(vid)
                if not view.IsEstimated():
                    continue
                try:
                    name = view.Name()
                except Exception:
                    # Fallback if name not available
                    name = f"view_{int(vid)}"
                cam = view.Camera()
                pos = np.array(cam.GetPosition(), dtype=np.float32)
                R = cam.GetOrientationAsRotationMatrix().T
                poses_with_names.append((name, pos, np.array(R, dtype=np.float32)))

        return poses_with_names

    def build_full_camera_trajectory_from_reconstructions(self) -> None:
        """Build full camera trajectory/orientations from all reconstructions.

        Deduplicates views by name to avoid duplicate entries across chunks.
        Preserves first-seen order across reconstructions.
        """
        entries = self._extract_camera_poses_with_names(latest_only=False)
        if not entries:
            self.full_camera_trajectory = []
            self.full_camera_orientations = []
            return

        seen_names: set = set()
        traj: List[np.ndarray] = []
        rots: List[np.ndarray] = []
        for name, pos, R in entries:
            if name in seen_names:
                continue
            seen_names.add(name)
            traj.append(pos)
            rots.append(R)

        self.full_camera_trajectory = traj
        self.full_camera_orientations = rots

    def _clear_chunk_data(self, chunk_result: Dict) -> None:
        """Free heavy per-chunk tensors once reconstruction is available/aligned."""
        keys_to_drop = [
            'points', 'local_points', 'conf', 'masks',
        ]
        for k in keys_to_drop:
            if k in chunk_result:
                try:
                    del chunk_result[k]
                except Exception:
                    pass
    
    def start_background_loader(self, all_image_paths: List):
        """Start background image loading for the entire sequence using chunk-based loading."""
        if self.loader_running:
            return
        
        try:
            # Store the full image paths list
            self.image_paths = all_image_paths
            
            # Calculate target size from first image if not already set
            if self.target_size is None:
                self.target_size = calculate_target_size(all_image_paths[0], self.pixel_limit)
            print(f"target_size: {self.target_size}")
            # Create chunk-based dataset
            dataset = ChunkImageDataset(all_image_paths, self.chunk_length, self.overlap, 
                                      self.target_size, device='cpu', 
                                      undistortion_maps=self.undistortion_maps)
            
            # Create DataLoader with multiple workers for parallel loading
            self.background_loader = DataLoader(
                dataset,
                batch_size=1,  # Each batch is one chunk
                shuffle=False,  # Keep original order
                num_workers=2,  # Use 2 worker processes for chunk loading
                pin_memory=True,  # Pin memory for faster GPU transfer
                persistent_workers=True,  # Keep workers alive between epochs
                prefetch_factor=1  # Prefetch 1 batch per worker
            )
            
            self.loader_running = True
            print(f"🔄 Background chunk loader started with {len(all_image_paths)} images in {len(dataset)} chunks")
            
        except Exception as e:
            print(f"❌ Failed to start background loader: {e}")
            print("🖥️  Continuing with synchronous loading...")
            self.loader_running = False
    
    def start_visualization(self):
        """Start the Rerun visualization in a separate process."""
        if self.vis_running:
            return
        
        try:
            # Create multiprocessing queue
            self.vis_queue = mp.Queue()
            
            # Start visualization process
            self.vis_process = mp.Process(
                target=visualization_process,
                args=(self.vis_queue, self.visualization_port, self.max_points_visualization, self.update_interval)
            )
            self.vis_process.start()
            self.vis_running = True
            
            print(f"🎥 Rerun visualization process started on port {self.visualization_port}")
            print(f"🌐 Open http://localhost:{self.visualization_port} in your browser to view the visualization")
            
        except Exception as e:
            print(f"❌ Failed to start visualization process: {e}")
            print("🖥️  Continuing without visualization...")
            self.vis_running = False
    
    def stop_visualization(self):
        """Stop the Rerun visualization process."""
        if self.vis_running and self.vis_process is not None:
            try:
                # Send shutdown signal
                if self.vis_queue is not None:
                    self.vis_queue.put(None)
                
                # Wait for process to finish
                self.vis_process.join(timeout=5.0)
                
                # Force terminate if still running
                if self.vis_process.is_alive():
                    self.vis_process.terminate()
                    self.vis_process.join(timeout=2.0)
                
                self.vis_running = False
                print("🎥 Rerun visualization process stopped")
                
            except Exception as e:
                print(f"⚠️  Error stopping visualization process: {e}")
    
    def process_chunks_with_background_loader(self, auto_close_visualization: bool = True) -> List[Dict]:
        """
        Process all chunks using the background loader for optimal performance.
        """
        if not self.loader_running or self.background_loader is None:
            raise ValueError("Background loader not started")
        
        results = []
        chunk_count = 0
        total_chunks = len(self.background_loader.dataset)
        processing_start_time = time.time()
        frames_before = len(self.timestamps)
        
        print(f"\n🔄 Starting chunk-based processing with {total_chunks} chunks...")
        print("=" * 60)
        
        try:
            for chunk_data in self.background_loader:
                chunk_count += 1
                start_idx = chunk_data['start_idx'].item()
                end_idx = chunk_data['end_idx'].item()
                chunk_images = chunk_data['chunk']
                chunk_paths = chunk_data['chunk_paths'][0]
                
                print(f"\n📦 Processing chunk {chunk_count}/{total_chunks}: frames {start_idx + 1}-{end_idx}")
                
                # Process chunk directly with loaded images
                _t0 = time.time()
                result = self._process_chunk_with_images(chunk_images, chunk_paths)
                self._record_timing("process_chunk", time.time() - _t0)
                results.append(result)
                
                # Small delay to allow visualization to update
                if self.vis_running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print(f"\n⚠️  Processing interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            raise
        
        # Optionally close visualization after processing completes
        if auto_close_visualization:
            self.stop_visualization()

        # Print timing after processing all chunks
        self.print_timing_statistics()

        # Compute and print average FPS over the whole processing
        elapsed_total = max(1e-6, time.time() - processing_start_time)
        frames_after = len(self.timestamps)
        frames_processed = max(0, frames_after - frames_before)
        avg_fps = frames_processed / elapsed_total
        print(f"\n⏱️ Overall performance: {frames_processed} frames in {elapsed_total:.2f}s  ->  average {avg_fps:.2f} FPS")
        return results

    def interpolate_world_points_for_ref(self, result, keypoints):
        # interpolate like the colors for keypoints
        H, W = result['images'].shape[-2:]
        grid_x = (keypoints[:, :, 0] / (W - 1)) * 2 - 1  # Convert to [-1, 1]
        grid_y = (keypoints[:, :, 1] / (H - 1)) * 2 - 1  # Convert to [-1, 1]
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1).cpu()  # Shape: (1, 1, N, 2)
        points = torch.nn.functional.grid_sample(
            result['points'].permute(0, 3, 1, 2), grid, mode="bilinear", align_corners=False, padding_mode="border")
        local_points = torch.nn.functional.grid_sample(
            result['local_points'].permute(0, 3, 1, 2), grid, mode="bilinear", align_corners=False, padding_mode="border")
        confidences = torch.nn.functional.grid_sample(
            result['conf'].permute(0, 3, 1, 2), grid, mode="nearest", align_corners=False, padding_mode="border")
        masks = torch.nn.functional.grid_sample(
            result['masks'].unsqueeze(1).float(), grid, mode="nearest", align_corners=False, padding_mode="border")
        return_result = {
            'points': points.squeeze().permute(0, 2, 1),
            'local_points': local_points.squeeze().permute(0, 2, 1),
            'conf': confidences.squeeze(1).permute(0, 2, 1),
            'masks': masks.squeeze(1).permute(0, 2, 1).bool()
        }
        return return_result

    def _record_timing(self, name: str, duration_s: float) -> None:
        """Accumulate timing statistics per step name."""
        entry = self.timing_stats.setdefault(name, {"total": 0.0, "count": 0})
        entry["total"] += float(duration_s)
        entry["count"] += 1

    def get_timing_statistics(self) -> Dict[str, Dict[str, float]]:
        """Return timing totals and averages per step."""
        summary: Dict[str, Dict[str, float]] = {}
        for k, v in self.timing_stats.items():
            total = v.get("total", 0.0)
            count = max(1, int(v.get("count", 0)))
            summary[k] = {"total_s": total, "count": count, "avg_ms": 1000.0 * total / count}
        return summary

    def print_timing_statistics(self) -> None:
        """Pretty-print timing statistics sorted by total time descending."""
        stats = self.get_timing_statistics()
        if not stats:
            print("⏱️ No timing data recorded")
            return
        print("\n⏱️ Timing summary (totals and averages):")
        for name, data in sorted(stats.items(), key=lambda x: x[1]["total_s"], reverse=True):
            print(f"   {name:32s} total={data['total_s']:.3f}s  count={data['count']:4d}  avg={data['avg_ms']:.1f}ms")
    
    def _get_scale_factor_for_pi3(self, moge_metric_depth, pi3_metric_depth, mask):
        moge_metric_depth = moge_metric_depth[mask]
        pi3_metric_depth = pi3_metric_depth[mask]
        scale_factor = moge_metric_depth / pi3_metric_depth
        scale_factor = scale_factor.median()
        return scale_factor
    
    def _process_chunk_with_images(self, chunk_images: torch.Tensor, chunk_paths: List) -> Dict:
        """Process a chunk using pre-loaded images."""
        # Extract timestamps for this chunk
        actual_paths = []
        for path_item in chunk_paths:
            if isinstance(path_item, list):
                actual_paths.extend(path_item)
            else:
                actual_paths.append(path_item)
        
        chunk_timestamps = extract_timestamps_from_paths(actual_paths)
        self.timestamps.extend(chunk_timestamps)
        
        # Model inference
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            _t0_inf = time.time()
            with torch.amp.autocast('cuda', dtype=dtype):
                result = self.model(chunk_images.to(self.device))
            self._record_timing("model_inference", time.time() - _t0_inf)
                
        # Process masks
        masks = torch.sigmoid(result['conf'][..., 0]) > 0.1
        non_edge = ~depth_edge(result['local_points'][..., 2], rtol=0.03)
        masks = torch.logical_and(masks, non_edge)[0]


        if self.moge_model is not None:
            _t0_moge = time.time()
            # get scale factor from first image
            with torch.amp.autocast('cuda', dtype=dtype):
                moge_metric_depth = self.moge_model.infer(chunk_images[0,0].to("cuda"))["depth"]
            pi3_metric_depth = result['local_points'][0,0][..., 2]
            mask = masks[0]
            scale_factor = self._get_scale_factor_for_pi3(moge_metric_depth, pi3_metric_depth, mask)
            print(f"  MOGE Scale factor: {scale_factor}")
            # Scale the reconstruction
            result["local_points"] = result["local_points"] * scale_factor
            result["points"] = result["points"] * scale_factor
            result["camera_poses"][:,:,:3,3] = result["camera_poses"][:,:,:3,3] * scale_factor
            result["camera_poses_cw"] = torch.linalg.inv(result["camera_poses"])
            result["original_width"] = self.target_size[1]
            result["original_height"] = self.target_size[0]
            self._record_timing("moge_inference_and_scale", time.time() - _t0_moge)

        # Estimate camera parameters for this chunk (if enabled)
        camera_params = None
        if self.estimate_camera_params:
            _t0_cam = time.time()
            camera_params = estimate_camera_parameters_single_chunk(result)
            self._record_timing("estimate_camera_params", time.time() - _t0_cam)
            print_camera_parameters_summary(camera_params)


        # Move all results to CPU to save GPU memory
        cpu_result = {
            'points': result['points'][0].cpu(),
            'local_points': result['local_points'][0].cpu(),
            'camera_poses': result['camera_poses'][0].cpu(),
            'conf': result['conf'][0].cpu(),
            'masks': masks.cpu(),
            'image_paths': chunk_paths,
            'images': chunk_images.squeeze(0).cpu()
        }

        # Remove all tensors from result that are still on gpu
        del result
        
        _t0_kp = time.time()
        keypoint_results = self.keypoint_extractor.extract_with_colors(chunk_images)
        self._record_timing("keypoint_extraction", time.time() - _t0_kp)
        print(f"🔍 Extracted {keypoint_results['keypoints'].shape[-2]} keypoints per frame")
    
        # Add camera parameters if estimated
        cpu_result['camera_params'] = {k: v.cpu() for k, v in camera_params.items()}
        
        # Add keypoint results if extracted
        # interpolate 3D points for each keypoint
        _t0_interp = time.time()
        res = self.interpolate_world_points_for_ref(cpu_result, keypoint_results['keypoints'])
        self._record_timing("interpolate_points", time.time() - _t0_interp)

        cpu_result['points'] = res['points']
        cpu_result['local_points'] = res['local_points']
        cpu_result['conf'] = res['conf']
        cpu_result['masks'] = res['masks']
        cpu_result['keypoints'] = keypoint_results['keypoints'].cpu()
        cpu_result['descriptors'] = keypoint_results['descriptors'].cpu()
        cpu_result['scores'] = keypoint_results['scores'].cpu()
        cpu_result['colors'] = keypoint_results['colors'].cpu()
    
        # Create chunk reconstruction if enabled
        chunk_reconstruction = None
        
        # Set target size for reconstruction
        self.chunk_reconstructor.set_target_size(self.target_size[1], self.target_size[0])
        
        # Add camera intrinsics if available
        if camera_params is not None:
            cpu_result['intrinsics'] = camera_params['intrinsics']
        
        # Create reconstruction
        _t0_recon = time.time()
        chunk_reconstruction = self.chunk_reconstructor.create_recon_from_chunk(
            cpu_result, max_observations_per_track = self.max_observations_per_track)
        self._record_timing("create_reconstruction", time.time() - _t0_recon)

        if self.save_debug_projections:
            if hasattr(self, 'output_dir') and self.output_dir:
                debug_gif_dir = os.path.join(self.output_dir, 'debug_projections')
                os.makedirs(debug_gif_dir, exist_ok=True)
                gif_filename = f"chunk_{len(self.chunk_reconstructions):03d}_projections.gif"
                gif_path = os.path.join(debug_gif_dir, gif_filename)
            else:
                # Fallback to current directory
                gif_path = f"chunk_{len(self.chunk_reconstructions):03d}_projections.gif"

            self.chunk_reconstructor.debug_projections(cpu_result, 0, [1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                                      save_path=gif_path, fps=1)
            
            self.chunk_reconstructor.print_reconstruction_summary()
        
        # Store reconstruction in memory instead of saving to disk in online mode
        self.chunk_reconstructions.append(chunk_reconstruction)
        
        # Only save to disk if explicitly enabled (for offline processing)
        if self.save_chunk_reconstructions:
            self._save_chunk_reconstruction(chunk_reconstruction, len(self.chunk_reconstructions))
    
        print_camera_parameters_summary(camera_params)
        
        # Generate camera IDs for this chunk
        chunk_idx = len(self.chunk_reconstructions)
        start_frame = chunk_idx * (self.chunk_length - self.overlap)
        chunk_camera_ids = list(range(start_frame + 1, start_frame + len(chunk_paths) + 1))
        
        # Track current chunk for frame display
        self.current_chunk_result = cpu_result
        
        # Align with previous chunks
        _t0_align = time.time()
        aligned_chunk = self._align_chunk_online(cpu_result, chunk_camera_ids)
        self._record_timing("align_chunk", time.time() - _t0_align)
        
        # Update visualization
        if self.vis_running:
            _t0_viz = time.time()
            self._update_visualization(aligned_chunk)
            self._record_timing("update_visualization", time.time() - _t0_viz)
        
        # Print progress
        print(f"📊 Chunk {len(self.chunk_reconstructions)}: {len(chunk_paths)} frames processed")
        # Get total points from reconstructions
        all_points, _ = self._extract_points_colors_from_reconstructions(latest_only=False)
        total_points = len(all_points) if all_points.size > 0 else 0
        print(f"   Total points: {total_points}")
        
        return {
            'chunk_result': cpu_result,
            'aligned_chunk': aligned_chunk,
            'camera_ids': chunk_camera_ids,
            'timestamps': chunk_timestamps
        }

    
    def _align_chunk_online(self, chunk_result: Dict, chunk_camera_ids: List[int]) -> Dict:
        """Align a new chunk using ONLY reconstruction-based alignment and then clear chunk data."""

        if len(self.chunk_reconstructions) == 1:
            # First chunk: record end pose from reconstruction (if any) and clear tensors
            cam_positions, cam_orientations = self._extract_camera_positions_from_reconstructions(latest_only=True)
            if cam_positions.size > 0:
                self.chunk_end_poses.append(cam_positions[-1])
            
            # Extract aligned data from the first reconstruction
            aligned_points, aligned_colors = self._extract_points_colors_from_reconstructions(latest_only=True)
            aligned_camera_positions, aligned_camera_orientations = self._extract_camera_positions_from_reconstructions(latest_only=True)
            
            self._clear_chunk_data(chunk_result)
            return {
                'points': aligned_points,
                'camera_poses': aligned_camera_positions,
                'camera_orientations': aligned_camera_orientations,
                'colors': aligned_colors,
                'transformation': np.eye(4)
            }

        # Align reconstructions and update chunk end pose
        transformation_matrix = self.align_with_reconstruction_tracks(chunk_result, chunk_camera_ids)

        cam_positions, cam_orientations = self._extract_camera_positions_from_reconstructions(latest_only=True)
        if cam_positions.size > 0:
            self.chunk_end_poses.append(cam_positions[-1])

        # Extract aligned data from the latest reconstruction (which now includes the aligned data)
        aligned_points, aligned_colors = self._extract_points_colors_from_reconstructions(latest_only=True)
        aligned_camera_positions, aligned_camera_orientations = self._extract_camera_positions_from_reconstructions(latest_only=True)

        # Clear heavy tensors; rely on reconstructions
        self._clear_chunk_data(chunk_result)

        return {
            'points': aligned_points,
            'camera_poses': aligned_camera_positions,
            'camera_orientations': aligned_camera_orientations,
            'colors': aligned_colors,
            'transformation': transformation_matrix if transformation_matrix is not None else np.eye(4)
        }


    def _save_chunk_reconstruction(self, reconstruction, chunk_idx: int):
        """
        Save chunk reconstruction to disk.
        
        Args:
            reconstruction: PyTheia reconstruction object
            chunk_idx: Chunk index for filename
        """
        if not hasattr(self, 'output_dir') or self.output_dir is None:
            print("⚠️  Warning: No output directory set, skipping chunk reconstruction save")
            return
        
        try:
            # Create reconstructions subdirectory
            recon_dir = os.path.join(self.output_dir, 'reconstructions')
            os.makedirs(recon_dir, exist_ok=True)
            
            # Save reconstruction file
            recon_filename = f"chunk_{chunk_idx:06d}.sfm"
            recon_path = os.path.join(recon_dir, recon_filename)
            
            # Use the reconstructor's save method
            self.chunk_reconstructor.reconstruction = reconstruction
            self.chunk_reconstructor.save_reconstruction(recon_path)
            
            print(f"💾 Saved chunk reconstruction {chunk_idx} to: {recon_path}")
            
        except Exception as e:
            print(f"❌ Error saving chunk reconstruction {chunk_idx}: {e}")
    
    def set_output_directory(self, output_dir: str):
        """
        Set the output directory for saving chunk reconstructions.
        
        Args:
            output_dir: Directory path for output files
        """
        self.output_dir = output_dir
        print(f"📁 Output directory set to: {output_dir}")
    
    def get_chunk_reconstructions(self):
        """
        Get list of all stored Pytheia reconstructions.
        
        Returns:
            List of PyTheia reconstruction objects
        """
        return self.chunk_reconstructions
    
    def get_latest_reconstruction(self):
        """
        Get the most recent Pytheia reconstruction.
        
        Returns:
            Latest PyTheia reconstruction object or None if no reconstructions exist
        """
        return self.chunk_reconstructions[-1] if self.chunk_reconstructions else None
    
    def get_reconstruction_count(self):
        """
        Get the number of stored reconstructions.
        
        Returns:
            Number of reconstructions in memory
        """
        return len(self.chunk_reconstructions)
    
    def save_reconstructions_to_disk(self, output_dir: str = None):
        """
        Save all stored reconstructions to disk.
        
        Args:
            output_dir: Output directory (uses self.output_dir if None)
        """
        if output_dir is None:
            output_dir = getattr(self, 'output_dir', None)
        
        if output_dir is None:
            print("⚠️  Warning: No output directory specified for saving reconstructions")
            return
        
        
        # Create reconstructions subdirectory
        recon_dir = os.path.join(output_dir, 'reconstructions')
        os.makedirs(recon_dir, exist_ok=True)
        
        print(f"💾 Saving {len(self.chunk_reconstructions)} reconstructions to {recon_dir}")
        
        for idx, reconstruction in enumerate(self.chunk_reconstructions):
            if reconstruction is not None:
                try:
                    # Save SFM reconstruction
                    recon_filename = f"chunk_{idx:06d}.sfm"
                    recon_path = os.path.join(recon_dir, recon_filename)
                    pt.io.WriteReconstruction(reconstruction, recon_path)
                    
                    # Save PLY point cloud
                    ply_filename = f"chunk_{idx:06d}.ply"
                    ply_path = os.path.join(recon_dir, ply_filename)
                    color = [255, 255, 255]  # White color for points
                    pt.io.WritePlyFile(ply_path, reconstruction, color, 1)
                    
                    print(f"   ✅ Saved chunk {idx}: {recon_filename} & {ply_filename}")
                    
                except Exception as e:
                    print(f"   ❌ Failed to save chunk {idx}: {e}")
        
        print(f"✅ Reconstruction saving completed")
    
    def align_with_reconstruction_tracks(self, chunk_result: Dict, chunk_camera_ids: List[int]) -> Optional[np.ndarray]:
        """
        Align the current chunk with the previous chunk using common tracks in PyTheia reconstructions.
        
        Args:
            chunk_result: Current chunk result
            chunk_camera_ids: Camera IDs for current chunk
            
        Returns:
            Transformation matrix (4x4) if alignment successful, None otherwise
        """
        if len(self.chunk_reconstructions) < 2:
            print("🔍 Skipping reconstruction-based alignment: need at least 2 reconstructions")
            return None
        
        try:
            # Get the two most recent reconstructions
            recon_ref = self.chunk_reconstructions[-2]  # Previous chunk
            recon_qry = self.chunk_reconstructions[-1]  # Current chunk
                        
            # Create view graph matches for overlapping frames
            view_graph_matches = create_view_graph_matches(self.chunk_length, self.overlap)
           
            success, alignment_info = align_and_refine_reconstructions(
                recon_ref, recon_qry, view_graph_matches)
            
            if success:                
                return alignment_info
            else:
                print(f"❌ Complete reconstruction alignment failed: {alignment_info.get('error', 'unknown')}")
                return None
            
            
        except Exception as e:
            print(f"❌ Reconstruction-based alignment failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _update_visualization(self, aligned_chunk_data: Dict = None):
        """Update visualization with current data."""
        if not self.vis_running or self.vis_queue is None:
            return
        
        # Throttle visualization updates
        current_time = time.time()
        if hasattr(self, '_last_viz_update') and current_time - self._last_viz_update < self.update_interval:
            return
        self._last_viz_update = current_time
        
        try:
            # Use aligned data if provided, otherwise fall back to reconstruction data
            if aligned_chunk_data is not None and aligned_chunk_data.get('points') is not None:
                # Use the aligned data directly from the alignment process
                all_points = aligned_chunk_data['points']
                all_colors = aligned_chunk_data['colors']
                camera_positions = aligned_chunk_data['camera_poses']
                camera_orientations = aligned_chunk_data.get('camera_orientations', np.array([]))
            else:
                # Fall back to reconstruction-only visualization - only get latest reconstruction data
                # since all previous data was already sent to rerun before
                all_points, all_colors = self._get_visualization_points()
                camera_positions, camera_orientations = self._extract_camera_positions_from_reconstructions(latest_only=True)
            
            # Get current frame and chunk frames
            current_frame = None
            chunk_start_frame = None
            chunk_end_frame = None
            
            # Get keypoint data for visualization
            current_keypoints = None
            chunk_start_keypoints = None
            chunk_end_keypoints = None
            
            if self.current_chunk_result is not None and 'images' in self.current_chunk_result:
                chunk_images = self.current_chunk_result['images']
                if len(chunk_images) > 0:
                    # Current frame is the last frame in the chunk
                    current_frame = chunk_images[-1].numpy()
                    
                    # Chunk start frame is the first frame in the chunk
                    chunk_start_frame = chunk_images[0].numpy()
                    
                    # Chunk end frame is the last frame in the chunk (same as current frame)
                    chunk_end_frame = chunk_images[-1].numpy()
                    
                    # Convert from CHW to HWC format for visualization
                    if len(current_frame.shape) == 3 and current_frame.shape[0] == 3:
                        current_frame = np.transpose(current_frame, (1, 2, 0))
                        chunk_start_frame = np.transpose(chunk_start_frame, (1, 2, 0))
                        chunk_end_frame = np.transpose(chunk_end_frame, (1, 2, 0))
                    
                    # Normalize to 0-255 range if needed
                    if current_frame.max() <= 1.0:
                        current_frame = (current_frame * 255).astype(np.uint8)
                        chunk_start_frame = (chunk_start_frame * 255).astype(np.uint8)
                        chunk_end_frame = (chunk_end_frame * 255).astype(np.uint8)
                    
                    # Extract keypoint data for the current frame (last frame in chunk)
                    if 'keypoints' in self.current_chunk_result and len(self.current_chunk_result['keypoints']) > 0:
                        current_keypoints = self.current_chunk_result['keypoints'][-1].numpy()  # Last frame keypoints

                    # Extract keypoint data for the chunk start frame (first frame in chunk)
                    if 'keypoints' in self.current_chunk_result and len(self.current_chunk_result['keypoints']) > 0:
                        chunk_start_keypoints = self.current_chunk_result['keypoints'][0].numpy()  # First frame keypoints

                    # Extract keypoint data for the chunk end frame (last frame in chunk)
                    if 'keypoints' in self.current_chunk_result and len(self.current_chunk_result['keypoints']) > 0:
                        chunk_end_keypoints = self.current_chunk_result['keypoints'][-1].numpy()  # Last frame keypoints

            # Create frame info and statistics
            frame_info = f"Frame: {len(self.timestamps)}, Chunk: {len(self.chunk_reconstructions)}"
            stats_text = f"Chunks: {len(self.chunk_reconstructions)}, Frames: {len(self.timestamps)}, Points: {len(all_points)}"
            
            # Send data to visualization process
            viz_data = {
                'points': all_points,
                'colors': all_colors,
                'camera_positions': camera_positions,
                'camera_orientations': camera_orientations,
                'chunk_end_poses': np.array(self.chunk_end_poses) if self.chunk_end_poses else np.array([]),
                'current_frame': current_frame,
                'chunk_start_frame': chunk_start_frame,
                'chunk_end_frame': chunk_end_frame,
                'frame_info': frame_info,
                'stats_text': stats_text,
                'current_keypoints': current_keypoints,
                'chunk_start_keypoints': chunk_start_keypoints,
                'chunk_end_keypoints': chunk_end_keypoints,
            }
            
            # Non-blocking put to avoid blocking the main process
            try:
                self.vis_queue.put_nowait(viz_data)
            except mp.queues.Full:
                pass
            
        except Exception as e:
            print(f"⚠️  Error sending visualization data: {e}")
    
    def _get_visualization_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get subsampled points for visualization from PyTheia reconstructions only."""
        # Only get points from the latest reconstruction since all previous data was already sent to rerun
        combined_points, combined_colors = self._extract_points_colors_from_reconstructions(latest_only=True)
        if combined_points.size == 0:
            return np.array([]), np.array([])
        
        # Subsample for visualization
        if len(combined_points) > 0:
            subsample_size = max(1000, int(len(combined_points) * self.visualization_subsample_ratio))
            if len(combined_points) > subsample_size:
                indices = np.random.choice(len(combined_points), subsample_size, replace=False)
                combined_points = combined_points[indices]
                combined_colors = combined_colors[indices]
        
        return combined_points, combined_colors
    
    def save_final_result(self, save_path: str, max_points: int = 1000000):
        """Save the final aligned trajectory result."""
        # Use reconstruction data instead of old aligned_points
        all_points, all_colors = self._extract_points_colors_from_reconstructions(latest_only=False)
        
        if all_points.size == 0:
            print("No points to save")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Subsample if needed
        if len(all_points) > max_points:
            indices = np.random.choice(len(all_points), max_points, replace=False)
            all_points = all_points[indices]
            all_colors = all_colors[indices]
        
        # Save point cloud
        print(f"💾 Saving final trajectory to: {save_path}")
        write_ply(torch.from_numpy(all_points), torch.from_numpy(all_colors), save_path, max_points=max_points)
        print(f"✅ Saved trajectory with {len(all_points)} points from {len(self.chunk_reconstructions)} reconstructions")
        
        # Save camera poses
        camera_positions, _ = self._extract_camera_positions_from_reconstructions(latest_only=False)
        if len(camera_positions) > 0:
            camera_colors = np.full((len(camera_positions), 3), [1.0, 0.0, 0.0])
            
            camera_filename = save_path.replace('.ply', '_camera_poses.ply')
            write_ply(torch.from_numpy(camera_positions), torch.from_numpy(camera_colors), camera_filename)
            print(f"📷 Saved {len(camera_positions)} camera poses to: {camera_filename}")
    
    def save_trajectory_tum(self, save_path: str, timestamps: List[float] = None, integer_timestamp: bool = False):
        """
        Save the final aligned trajectory in TUM format.
        
        Args:
            save_path: Path to save the TUM format trajectory file
            timestamps: Optional list of timestamps for each pose (if None, will use stored timestamps or frame indices)
            integer_timestamp: If True, saves timestamps as integers (for 7-scenes). Otherwise, saves as float (for EuRoC).
        """
        # Build full trajectory from reconstructions if empty or lengths mismatch timestamps
        if not self.full_camera_trajectory:
            self.build_full_camera_trajectory_from_reconstructions()
        if not self.full_camera_trajectory:
            print("No camera trajectory available to save from reconstructions")
            return
        
        try:
            from scipy.spatial.transform import Rotation
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Override timestamps with a simple range if integer_timestamp is requested
            if integer_timestamp:
                timestamps_to_use = list(range(len(self.full_camera_trajectory)))
                print("   Using integer frame indices as timestamps.")
            elif timestamps is not None:
                timestamps_to_use = timestamps
            elif self.timestamps and len(self.timestamps) >= len(self.full_camera_trajectory):
                timestamps_to_use = self.timestamps[:len(self.full_camera_trajectory)]
            else:
                timestamps_to_use = list(range(len(self.full_camera_trajectory)))

            
            print(f"💾 Saving trajectory in TUM format to: {save_path}")
            print(f"   Using {len(timestamps_to_use)} timestamps for {len(self.full_camera_trajectory)} poses")
            
            with open(save_path, 'w') as f:
                # TUM format header
                f.write("# timestamp tx ty tz qx qy qz qw\n")
                
                for i, (camera_pos, camera_rot) in enumerate(zip(self.full_camera_trajectory, self.full_camera_orientations)):
                    # Get timestamp
                    t = timestamps_to_use[i] if i < len(timestamps_to_use) else i
                    
                    # Extract translation (x, y, z)
                    x, y, z = camera_pos
                    
                    # Convert rotation matrix to quaternion
                    quat = Rotation.from_matrix(camera_rot).as_quat()
                    qx, qy, qz, qw = quat
                    
                    # Write TUM format line: timestamp tx ty tz qx qy qz qw
                    if integer_timestamp:
                        f.write(f"{int(t)} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
                    else:
                        # Use full precision for timestamp to match ground truth format
                        f.write(f"{t:.9f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
            
            print(f"✅ Saved trajectory with {len(self.full_camera_trajectory)} poses to: {save_path}")
            
        except ImportError:
            print("❌ Error: scipy.spatial.transform.Rotation not available")
        except Exception as e:
            print(f"❌ Error saving trajectory: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.stop_visualization()
        except:
            pass
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        return {
            'total_chunks': len(self.chunk_reconstructions),
            'total_frames': len(self.timestamps)
        }
    
    def _debug_print_overlap_indices(self, overlap_cameras_current: List[int], overlap_cameras_next: List[int], 
                                   common_camera_ids: set, chunk_result: Dict) -> None:
        """
        Debug function to print overlap indices and statistics during chunk alignment.
        
        Args:
            overlap_cameras_current: Camera IDs from current chunk that overlap
            overlap_cameras_next: Camera IDs from next chunk that overlap
            common_camera_ids: Set of common camera IDs between chunks
            chunk_result: Current chunk result data
        """
        print(f"\n🔍 CHUNK OVERLAP DEBUG:")
        print(f"   Current chunk overlap cameras: {overlap_cameras_current}")
        print(f"   Next chunk overlap cameras: {overlap_cameras_next}")
        print(f"   Common camera IDs: {sorted(common_camera_ids)}")
        print(f"   Number of common cameras: {len(common_camera_ids)}")
        
        # Print overlap indices for keypoints if available
        if 'points' in chunk_result:
            num_keypoints = chunk_result['points'].shape[1] if len(chunk_result['points'].shape) > 1 else chunk_result['points'].shape[0]
            print(f"   Keypoints in current chunk: {num_keypoints}")
            
            # Calculate overlap indices based on common cameras
            overlap_indices = []
            for cam_id in common_camera_ids:
                if cam_id in overlap_cameras_current:
                    # This camera is in the overlap region of current chunk
                    # The indices would be the keypoints from this camera
                    # For now, just note which cameras are overlapping
                    overlap_indices.append(f"cam_{cam_id}")
            
            print(f"   Overlap camera indices: {overlap_indices}")
        
        # Print point cloud statistics
        if 'points' in chunk_result:
            points_shape = chunk_result['points'].shape
            print(f"   Point cloud shape: {points_shape}")
            
            if len(points_shape) == 4:  # (N, H, W, 3)
                total_points = points_shape[0] * points_shape[1] * points_shape[2]
                print(f"   Total points in chunk: {total_points}")
            elif len(points_shape) == 3:  # (N, M, 3)
                total_points = points_shape[0] * points_shape[1]
                print(f"   Total points in chunk: {total_points}")
        
        # Print confidence statistics if available
        if 'conf' in chunk_result:
            conf_shape = chunk_result['conf'].shape
            print(f"   Confidence shape: {conf_shape}")
            if len(conf_shape) > 0:
                mean_conf = chunk_result['conf'].mean().item()
                print(f"   Mean confidence: {mean_conf:.3f}")
        
        print(f"   Overlap size: {self.overlap}")
        print(f"   Chunk length: {self.chunk_length}")
        print("-" * 50) 
"""
Offline reconstructor that progressively loads saved chunk files and builds a global reconstruction.

It mirrors the online reconstructor's reconstruction and alignment steps, without any visualization.
Outputs per-chunk .sfm/.ply files and a final merged reconstruction.
"""

from __future__ import annotations

import os
import glob
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import pytheia as pt
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from pi3.utils.basic import write_ply
from dataclasses import dataclass

from utils.chunk_reconstruction import ChunkPTRecon
from utils.reconstruction_alignment import create_view_graph_matches, align_and_refine_reconstructions


def _reconstruct_chunk_worker(args: Tuple[int, str, str, int, bool, bool, bool, bool, bool]) -> Dict:
    """Top-level worker to reconstruct a single chunk and persist to disk.

    Args:
        args: Tuple of (chunk_index, chunk_path, recon_dir, max_obs_per_track,
                        use_inverse_depth, write_ply_flag, use_lk_refinement, debug_lk_refinement)

    Returns:
        Dict with result metadata: {'idx', 'sfm_path', 'ply_path', 'num_frames', 'duration_s', 'fps', 'error'}
    """
    idx, chunk_path, recon_dir, max_obs_per_track, use_inverse_depth, write_ply_flag, use_lk_refinement, debug_lk_refinement, run_bundle_adjustment = args
    t0 = time.time()
    sfm_path = os.path.join(recon_dir, f"chunk_{idx:06d}.sfm")
    ply_path = os.path.join(recon_dir, f"chunk_{idx:06d}.ply")

    try:
        # Load data
        data: Dict = torch.load(chunk_path, map_location='cpu')

        # Determine target size
        H = int(data.get('original_height', 0) or (int(data['gray_images'].shape[-2]) if 'gray_images' in data and data['gray_images'] is not None else 1080))
        W = int(data.get('original_width', 0) or (int(data['gray_images'].shape[-1]) if 'gray_images' in data and data['gray_images'] is not None else 1920))

        # Ensure intrinsics if camera_params present
        if 'camera_params' in data and data['camera_params'] is not None:
            data['intrinsics'] = data['camera_params'].get('intrinsics', None)

        # Build reconstruction
        reconstructor = ChunkPTRecon()
        reconstructor.set_target_size(W, H)
        recon = reconstructor.create_recon_from_chunk(
            data,
            max_observations_per_track=max_obs_per_track,
            use_inverse_depth=use_inverse_depth,
            use_lk_refinement=use_lk_refinement,
            collect_debug=debug_lk_refinement,
            run_bundle_adjustment=run_bundle_adjustment,
        )

        # Optional: save a debug visualization comparing projected vs final LK points
        if debug_lk_refinement:
            try:
                # Pick the first available source frame from collected pairs
                pairs_keys = list(reconstructor._debug_pairs.keys())
                if pairs_keys:
                    src = pairs_keys[0][0]
                    tgts = [t for (s, t) in pairs_keys if s == src]
                    if tgts:
                        dbg_path = os.path.join(recon_dir, f"debug_chunk_{idx:06d}_src_{src}.png")
                        reconstructor.debug_projections(data, src, tgts, save_path=dbg_path)
            except Exception as _:
                pass

        # Save SFM
        pt.io.WriteReconstruction(recon, sfm_path)
        # Optionally save PLY
        if write_ply_flag:
            color = [255, 255, 255]
            pt.io.WritePlyFile(ply_path, recon, color, 1)

        # Metrics
        try:
            if 'keypoints' in data and data['keypoints'] is not None:
                num_frames = int(data['keypoints'].shape[0])
            else:
                num_frames = int(data['camera_poses'].shape[0])
        except Exception:
            num_frames = 0

        dt = max(1e-6, time.time() - t0)
        fps = (num_frames / dt) if num_frames > 0 else 0.0

        return {
            'idx': idx,
            'sfm_path': sfm_path,
            'ply_path': (ply_path if write_ply_flag else None),
            'num_frames': num_frames,
            'duration_s': dt,
            'fps': fps,
            'error': None,
        }
    except Exception as e:
        return {
            'idx': idx,
            'sfm_path': None,
            'ply_path': None,
            'num_frames': 0,
            'duration_s': time.time() - t0,
            'fps': 0.0,
            'error': f"{type(e).__name__}: {e}",
        }


@dataclass
class OfflineReconstructorConfig:
    chunk_dir: str
    output_dir: str
    chunk_length: Optional[int] = None
    overlap: Optional[int] = None
    max_observations_per_track: int = 5
    save_per_chunk: bool = False
    use_inverse_depth: bool = False
    num_workers: Optional[int] = None
    use_lk_refinement: bool = False
    debug_lk_refinement: bool = False
    shared_intrinsics: bool = True

class OfflineReconstructor:
    def __init__(self, config: OfflineReconstructorConfig):
        self.chunk_dir = config.chunk_dir
        self.output_dir = config.output_dir
        self.use_lk_refinement = config.use_lk_refinement
        self.debug_lk_refinement = config.debug_lk_refinement
        self.shared_intrinsics = config.shared_intrinsics

        # Auto-load metadata if not provided
        meta_path = os.path.join(self.chunk_dir, 'chunk_metadata.json')
        loaded_chunk_length = None
        loaded_overlap = None

        try:
            import json
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                loaded_chunk_length = int(meta.get('chunk_length')) if meta.get('chunk_length') is not None else None
                loaded_overlap = int(meta.get('overlap')) if meta.get('overlap') is not None else None
        except Exception:
            pass

        self.chunk_length = int(config.chunk_length) if config.chunk_length is not None else (loaded_chunk_length or 100)
        self.overlap = int(config.overlap) if config.overlap is not None else (loaded_overlap or 10)
        self.max_observations_per_track = config.max_observations_per_track
        self.save_per_chunk = config.save_per_chunk
        self.use_inverse_depth = config.use_inverse_depth
        self.num_workers = int(config.num_workers) if config.num_workers is not None else max(1, (os.cpu_count() or 1))

        os.makedirs(self.output_dir, exist_ok=True)
        self.recon_dir = os.path.join(self.output_dir, 'reconstructions')
        os.makedirs(self.recon_dir, exist_ok=True)

        self.reconstructor = ChunkPTRecon()
        self.reconstructions: List[pt.sfm.Reconstruction] = []

    def _load_chunks(self) -> List[str]:
        files = sorted(glob.glob(os.path.join(self.chunk_dir, "chunks", 'chunk_*.pt')))
        if not files:
            raise FileNotFoundError(f"No chunk_*.pt files found in {self.chunk_dir}")
        return files

    def _create_reconstruction_from_chunk(self, chunk: Dict) -> pt.sfm.Reconstruction:
        # Determine target size
        H = int(chunk.get('original_height', 1080))
        W = int(chunk.get('original_width', 1920))
        self.reconstructor.set_target_size(W, H)

        # Ensure intrinsics if camera_params present
        if 'camera_params' in chunk and chunk['camera_params'] is not None:
            chunk['intrinsics'] = chunk['camera_params'].get('intrinsics', None)

        # Build reconstruction
        recon = self.reconstructor.create_recon_from_chunk(chunk, 
            max_observations_per_track=self.max_observations_per_track,
            use_inverse_depth=self.use_inverse_depth,
            use_lk_refinement=self.use_lk_refinement,
            collect_debug=self.debug_lk_refinement,
            shared_intrinsics=self.shared_intrinsics)
        
        
        return recon

    def _save_chunk_reconstruction(self, recon: pt.sfm.Reconstruction, idx: int) -> None:
        try:
            # Save SFM
            sfm_path = os.path.join(self.recon_dir, f"chunk_{idx:06d}.sfm")
            pt.io.WriteReconstruction(recon, sfm_path)
            # Save PLY
            ply_path = os.path.join(self.recon_dir, f"chunk_{idx:06d}.ply")
            color = [255, 255, 255]
            pt.io.WritePlyFile(ply_path, recon, color, 1)
            print(f"   💾 Wrote: {sfm_path} and {ply_path}")
        except Exception as e:
            print(f"   ❌ Failed to save recon {idx}: {e}")

    def _align_last_two(self) -> Optional[np.ndarray]:
        if len(self.reconstructions) < 2:
            return None
        recon_ref = self.reconstructions[-2]
        recon_qry = self.reconstructions[-1]
        matches = create_view_graph_matches(self.chunk_length, self.overlap)
        success, info = align_and_refine_reconstructions(recon_ref, recon_qry, matches)
        if not success:
            print(f"   ❌ Alignment failed for chunk {len(self.reconstructions)-1}")
            return None
        return info

    def run(self) -> None:
        chunk_files = self._load_chunks()
        num_chunks = len(chunk_files)
        print(f"🔄 Reconstructing {num_chunks} chunks from {self.chunk_dir}")

        # Stage 1: parallel per-chunk reconstruction (without BA) -> write SFM (and optional PLY) to disk
        print(f"🚀 Launching {self.num_workers} threads for per-chunk reconstruction (no BA in parallel)...")
        tasks = [
            (idx, path, self.recon_dir, self.max_observations_per_track, 
              self.use_inverse_depth, self.save_per_chunk, 
              self.use_lk_refinement, self.debug_lk_refinement, False)  # run_bundle_adjustment=False
            for idx, path in enumerate(chunk_files)
        ]
        results_by_idx: Dict[int, Dict] = {}

        if self.num_workers and self.num_workers > 0:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_idx = {executor.submit(_reconstruct_chunk_worker, t): t[0] for t in tasks}
                for future in as_completed(future_to_idx):
                    res = future.result()
                    idx = res['idx']
                    results_by_idx[idx] = res
                    if res['error'] is None:
                        print(f"   ✅ Chunk {idx+1}/{num_chunks}: ⏱️ {res['duration_s']:.3f}s for {res['num_frames']} frames -> {res['fps']:.2f} FPS")
                    else:
                        print(f"   ❌ Chunk {idx+1}/{num_chunks} failed: {res['error']}")
        else:
            print("🧵 Running sequential per-chunk reconstruction (num_workers=0, no BA)...")
            for task in tasks:
                res = _reconstruct_chunk_worker(task)
                idx = res['idx']
                results_by_idx[idx] = res
                if res['error'] is None:
                    print(f"   ✅ Chunk {idx+1}/{num_chunks}: ⏱️ {res['duration_s']:.3f}s for {res['num_frames']} frames -> {res['fps']:.2f} FPS")
                else:
                    print(f"   ❌ Chunk {idx+1}/{num_chunks} failed: {res['error']}")

        # Stage 2a: sequential bundle adjustment per chunk
        print("\n🧮 Running bundle adjustment sequentially for each chunk...")
        for idx in range(num_chunks):
            res = results_by_idx.get(idx)
            if res is None or res.get('sfm_path') is None:
                print(f"   ⚠️ Skipping BA for chunk {idx}: no reconstruction available")
                continue
            try:
                success, recon = pt.io.ReadReconstruction(res['sfm_path'])
                if not success:
                    print(f"   ❌ Failed to load reconstruction for BA (chunk {idx}): {recon}")
                    continue
            except Exception as e:
                print(f"   ❌ Failed to load reconstruction for BA (chunk {idx}): {e}")
                continue

            # Run BA in-place using helper
            try:
                tmp = ChunkPTRecon()
                tmp.reconstruction = recon
                tmp.use_inverse_depth = self.use_inverse_depth
                tmp.run_bundle_adjustment()
                # Write back after BA
                pt.io.WriteReconstruction(tmp.reconstruction, res['sfm_path'])
                if self.save_per_chunk:
                    ply_path = os.path.join(self.recon_dir, f"chunk_{idx:06d}.ply")
                    color = [255, 255, 255]
                    pt.io.WritePlyFile(ply_path, tmp.reconstruction, color, 1)
            except Exception as e:
                print(f"   ❌ BA failed for chunk {idx}: {e}")

        # Stage 2b: sequential load + alignment
        print("\n🔗 Aligning reconstructions sequentially...")
        for idx in range(num_chunks):
            res = results_by_idx.get(idx)
            if res is None or res.get('sfm_path') is None:
                print(f"   ⚠️ Skipping chunk {idx}: no reconstruction available")
                continue

            # Load reconstruction from disk
            try:
                success, recon = pt.io.ReadReconstruction(res['sfm_path'])
                if not success:
                    print(f"   ❌ Failed to load reconstruction for chunk {idx}: {recon}")
                    continue
            except Exception as e:
                print(f"   ❌ Failed to load reconstruction for chunk {idx}: {e}")
                continue

            self.reconstructions.append(recon)

            # Align with previous
            if idx > 0:
                print(f"   🔗 Aligning chunk {idx} with previous reconstruction...")
                self._align_last_two()

        # Export final results: combined PLY and TUM trajectory
        if self.reconstructions:
            # Combined PLY
            try:
                points, colors = self._extract_points_colors_from_reconstructions(latest_only=False)
                if points.size > 0:
                    ply_path = os.path.join(self.output_dir, 'final_points.ply')
                    write_ply(torch.from_numpy(points), torch.from_numpy(colors if colors.size > 0 else np.ones_like(points)), ply_path)
                    print(f"\n✅ Final point cloud saved: {ply_path}")
                else:
                    print("\n⚠️ No points extracted from reconstructions; skipping final PLY")
            except Exception as e:
                print(f"❌ Failed to save final PLY: {e}")

            # Camera trajectory as PLY
            try:
                cam_positions, _, _ = self._extract_camera_positions_from_reconstructions()
                if cam_positions:
                    cam_pts = np.asarray(cam_positions, dtype=np.float32)
                    cam_cols = np.full((len(cam_pts), 3), [1.0, 0.0, 0.0], dtype=np.float32)
                    cam_ply_path = os.path.join(self.output_dir, 'final_camera_poses.ply')
                    write_ply(torch.from_numpy(cam_pts), torch.from_numpy(cam_cols), cam_ply_path)
                    print(f"✅ Final camera trajectory PLY saved: {cam_ply_path}")
                else:
                    print("⚠️ No camera poses extracted; skipping trajectory PLY")
            except Exception as e:
                print(f"❌ Failed to save camera trajectory PLY: {e}")

            # TUM trajectory (integer timestamps)
            try:
                tum_path = os.path.join(self.output_dir, 'trajectory_tum.txt')
                self._save_trajectory_tum(tum_path, integer_timestamp=True)
            except Exception as e:
                print(f"❌ Failed to save TUM trajectory: {e}")

            # Optional: interpolate missing frames based on frame_selection.json
            try:
                selection_path = os.path.join(os.path.dirname(self.chunk_dir), 'frame_selection.json')
                if not os.path.exists(selection_path):
                    selection_path = os.path.join(self.chunk_dir, 'frame_selection.json')
                if os.path.exists(selection_path):
                    import json
                    with open(selection_path, 'r') as f:
                        sel = json.load(f)
                    total_frames = int(sel.get('original_total_frames', 0))
                    selected_indices = sel.get('selected_indices', [])
                    if total_frames > 0 and selected_indices:
                        print(f"\n🔧 Interpolating missing frames (total={total_frames}, selected={len(selected_indices)})...")
                        self._interpolate_and_save_full_trajectory(total_frames, selected_indices)
                else:
                    print("ℹ️  No frame_selection.json found; skipping pose interpolation")
            except Exception as e:
                print(f"⚠️  Pose interpolation failed: {e}")

    def _extract_points_colors_from_reconstructions(self, latest_only: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if not self.reconstructions:
            return np.array([]), np.array([])
        all_points: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []
        reconstructions_to_process = [self.reconstructions[-1]] if latest_only else self.reconstructions
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
        colors_arr = np.asarray(all_colors, dtype=np.float32) if all_colors else np.array([])
        if colors_arr.size > 0 and colors_arr.max() > 1.0:
            colors_arr = colors_arr / 255.0
        return points_arr, colors_arr

    def _extract_camera_positions_from_reconstructions(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        if not self.reconstructions:
            return [], [], []
        positions: List[np.ndarray] = []
        orientations: List[np.ndarray] = []
        names: List[str] = []
        for recon in self.reconstructions:
            if recon is None:
                continue
            for vid in sorted(recon.ViewIds()):
                view = recon.View(vid)
                if not view.IsEstimated():
                    continue
                try:
                    name = view.Name()
                except Exception:
                    name = f"view_{int(vid)}"
                cam = view.Camera()
                positions.append(np.array(cam.GetPosition(), dtype=np.float32))
                R = cam.GetOrientationAsRotationMatrix().T
                orientations.append(np.array(R, dtype=np.float32))
                names.append(name)
        return positions, orientations, names

    def _build_full_camera_trajectory(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        positions, orientations, names = self._extract_camera_positions_from_reconstructions()
        seen = set()
        traj: List[np.ndarray] = []
        rots: List[np.ndarray] = []
        for name, pos, R in zip(names, positions, orientations):
            if name in seen:
                continue
            seen.add(name)
            traj.append(pos)
            rots.append(R)
        return traj, rots

    def _save_trajectory_tum(self, save_path: str, integer_timestamp: bool = True) -> None:
        traj, rots = self._build_full_camera_trajectory()
        if not traj:
            print("No camera trajectory available to save from reconstructions")
            return
        try:
            from scipy.spatial.transform import Rotation
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Use integer indices as timestamps by default
            timestamps_to_use = list(range(len(traj)))
            with open(save_path, 'w') as f:
                f.write("# timestamp tx ty tz qx qy qz qw\n")
                for i, (pos, R) in enumerate(zip(traj, rots)):
                    x, y, z = pos
                    quat = Rotation.from_matrix(R).as_quat()
                    qx, qy, qz, qw = quat
                    if integer_timestamp:
                        f.write(f"{i} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
                    else:
                        f.write(f"{float(i):.9f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
            print(f"✅ Saved trajectory with {len(traj)} poses to: {save_path}")
        except ImportError:
            print("❌ Error: scipy.spatial.transform.Rotation not available")
        except Exception as e:
            print(f"❌ Error saving trajectory: {e}")

    def _interpolate_and_save_full_trajectory(self, total_frames: int, selected_indices: List[int]) -> None:
        try:
            from scipy.spatial.transform import Rotation, Slerp
        except ImportError:
            print("❌ scipy not available; cannot perform SLERP interpolation")
            return

        # Extract known camera poses in order of appearance
        positions, orientations, _ = self._extract_camera_positions_from_reconstructions()
        if not positions or not orientations:
            print("⚠️  No poses available for interpolation")
            return

        # Map selected_indices -> pose index order
        if len(selected_indices) != len(positions):
            print(f"⚠️  Mismatch: selected {len(selected_indices)} vs reconstructed {len(positions)}; proceeding with min length")
        k = min(len(selected_indices), len(positions))
        selected_indices = selected_indices[:k]
        positions = positions[:k]
        orientations = orientations[:k]

        key_times = np.array(selected_indices, dtype=float)
        key_rots = Rotation.from_matrix(np.stack(orientations, axis=0))
        slerp = Slerp(key_times, key_rots)

        # Interpolate per-frame
        all_times = np.arange(total_frames, dtype=float)
        interp_rots = slerp(all_times)

        # Linear translation interpolation per axis
        positions_np = np.stack(positions, axis=0)
        interp_pos = np.empty((total_frames, 3), dtype=np.float32)
        for dim in range(3):
            interp_pos[:, dim] = np.interp(all_times, key_times, positions_np[:, dim])

        # Save TUM full
        tum_full = os.path.join(self.output_dir, 'trajectory_tum_full.txt')
        try:
            with open(tum_full, 'w') as f:
                f.write("# timestamp tx ty tz qx qy qz qw\n")
                quats = interp_rots.as_quat()  # x,y,z,w
                for i in range(total_frames):
                    x, y, z = interp_pos[i]
                    qx, qy, qz, qw = quats[i]
                    f.write(f"{i} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
            print(f"✅ Saved interpolated full TUM trajectory: {tum_full}")
        except Exception as e:
            print(f"❌ Failed to save interpolated TUM: {e}")

        # Save camera positions PLY full
        try:
            cam_cols = np.full((total_frames, 3), [0.0, 1.0, 0.0], dtype=np.float32)
            cam_ply_full = os.path.join(self.output_dir, 'final_camera_poses_full.ply')
            write_ply(torch.from_numpy(interp_pos), torch.from_numpy(cam_cols), cam_ply_full)
            print(f"✅ Saved interpolated camera PLY: {cam_ply_full}")
        except Exception as e:
            print(f"❌ Failed to save interpolated camera PLY: {e}")

        # Optionally add dummy unestimated views to reconstruction to reflect full timeline
        try:
            current_views = set(self.reconstruction.ViewIds())
            # Build map from existing timestamps (names are filenames or frame indices as timestamp)
            existing_names = {}
            for vid in current_views:
                view = self.reconstruction.View(vid)
                try:
                    name = view.Name()
                    existing_names[name] = vid
                except Exception:
                    pass
            # Append missing views with intrinsics prior only
            intrinsics_example = None
            for vid in current_views:
                cam = self.reconstruction.View(vid).Camera()
                try:
                    intrinsics_example = cam.GetCalibrationMatrix()
                    break
                except Exception:
                    continue
            for i in range(total_frames):
                name = f"frame_{i}"
                if name in existing_names:
                    continue
                view_id = self.reconstruction.AddView(name, 0, i)
                view = self.reconstruction.MutableView(view_id)
                # Set pose estimate
                R = interp_rots[i].as_matrix().T  # our orientation storage expects transpose
                t = interp_pos[i]
                cam = view.MutableCamera()
                cam.SetPosition(t)
                cam.SetOrientationFromRotationMatrix(R)
                # mark as unestimated to avoid polluting BA
                view.SetIsEstimated(True)
        except Exception as e:
            print(f"⚠️  Failed to append interpolated views to reconstruction: {e}")



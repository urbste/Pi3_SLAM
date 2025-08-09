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
from pi3.utils.basic import write_ply

from utils.chunk_reconstruction import ChunkPTRecon
from utils.reconstruction_alignment import create_view_graph_matches, align_and_refine_reconstructions


class OfflineReconstructor:
    def __init__(self, chunk_dir: str, output_dir: str, chunk_length: Optional[int] = None, overlap: Optional[int] = None, max_observations_per_track: int = 5, save_per_chunk: bool = False):
        self.chunk_dir = chunk_dir
        self.output_dir = output_dir

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

        self.chunk_length = int(chunk_length) if chunk_length is not None else (loaded_chunk_length or 100)
        self.overlap = int(overlap) if overlap is not None else (loaded_overlap or 10)
        self.max_observations_per_track = max_observations_per_track
        self.save_per_chunk = save_per_chunk

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
        recon = self.reconstructor.create_recon_from_chunk(chunk, max_observations_per_track=self.max_observations_per_track)
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
        print(f"🔄 Reconstructing {len(chunk_files)} chunks from {self.chunk_dir}")

        for idx, path in enumerate(chunk_files):
            print(f"\n📦 Loading {os.path.basename(path)} ({idx+1}/{len(chunk_files)})")
            data: Dict = torch.load(path, map_location='cpu')

            # Build per-chunk reconstruction
            t0 = time.time()
            recon = self._create_reconstruction_from_chunk(data)
            dt = max(1e-6, time.time() - t0)
            try:
                if 'keypoints' in data and data['keypoints'] is not None:
                    num_frames = int(data['keypoints'].shape[0])
                else:
                    num_frames = int(data['camera_poses'].shape[0])
            except Exception:
                num_frames = 0
            fps = (num_frames / dt) if num_frames > 0 else 0.0
            print(f"   ⏱️ Reconstruction: {dt:.3f}s for {num_frames} frames  ->  {fps:.2f} FPS")
            self.reconstructions.append(recon)
            if self.save_per_chunk:
                self._save_chunk_reconstruction(recon, idx)

            # Align with previous
            if idx > 0:
                print("   🔗 Aligning with previous reconstruction...")
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

            # TUM trajectory (integer timestamps)
            try:
                tum_path = os.path.join(self.output_dir, 'trajectory_tum.txt')
                self._save_trajectory_tum(tum_path, integer_timestamp=True)
            except Exception as e:
                print(f"❌ Failed to save TUM trajectory: {e}")

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



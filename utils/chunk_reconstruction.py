"""
Chunk-based PyTheia reconstruction utilities for Pi3SLAM.
"""

from sympy import true
import torch
import numpy as np
from typing import Dict, List, Optional
import pytheia as pt
from pi3.utils.camera import Camera
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import os
import time

class ChunkPTRecon:
    """
    PyTheia reconstruction manager for chunks with keypoint projection capabilities.
    """
    
    def __init__(self):
        """
        Initialize ChunkPTRecon.
        
        """

        self.reconstruction = pt.sfm.Reconstruction()
        self.view_ids = []
        self.track_ids = []
        self.use_inverse_depth = False
        # cache for uint8 grayscale images used by LK
        self._gray_image_cache = {}
        self._gray_cache_dir = os.path.join("logs", "gray_cache")
        # store projected vs final (LK-refined) points for debugging visualization
        self._debug_pairs = {}
        self._collect_debug = False

    def set_target_size(self, original_width: int, original_height: int):
        """
        Set the target size for the reconstruction.
        """
        self.original_width = original_width
        self.original_height = original_height

    def create_recon_from_chunk(self, chunk_data: Dict, 
        max_observations_per_track: int = 5, 
        use_inverse_depth: bool = False, 
        use_lk_refinement: bool = False, 
        lk_params: Optional[Dict] = None, 
        collect_debug: bool = False,
        run_bundle_adjustment: bool = True,
        shared_intrinsics: bool = True) -> pt.sfm.Reconstruction:
        """
        Create PyTheia reconstruction from chunk data.
        
        Args:
            chunk_data: Dictionary containing:
                - 'keypoints': Keypoint coordinates (N, num_keypoints, 2) - optional
                - 'colors': Keypoint colors (N, num_keypoints, 3) - optional
                - 'points_kp': 3D world points for keypoints (N, num_keypoints, 3) - optional
                - 'camera_poses': Camera poses (N, 4, 4)
                - 'intrinsics': Camera intrinsics (N, 3, 3) - optional
                - 'conf_kp': Keypoint confidences (N, num_keypoints) - optional
                - 'masks_kp': Keypoint masks (N, num_keypoints) - optional
            max_observations_per_track: Maximum number of observations to create per track (default: 5)
            use_inverse_depth: Toggle inverse-depth parametrization during BA
            use_lk_refinement: If True, refine projected 2D points using Lucas–Kanade optical flow
            lk_params: Optional dict overriding LK params: {
                'win': (w, h), 'levels': int, 'criteria': (type, count, eps), 'max_deviation_px': float
            }
            collect_debug: If True, record projected vs final 2D points for debugging plots
            run_bundle_adjustment: If True, run bundle adjustment
            shared_intrinsics: If True, use same shared intrinsics for all chunks. we will set the first frame's intrinsics as the shared intrinsics
        
        Returns:
            PyTheia reconstruction object
        """
        # Clear previous reconstruction
        self.reconstruction = pt.sfm.Reconstruction()
        self.view_ids = []
        self.track_ids = []
        self.use_inverse_depth = use_inverse_depth
        self._collect_debug = bool(collect_debug)
        self._debug_pairs = {} if self._collect_debug else {}
        
        # Check if keypoints are available
        has_keypoints = ('keypoints' in chunk_data and 
                        'colors' in chunk_data and 
                        'points' in chunk_data and
                        chunk_data['keypoints'] is not None)
        
        if has_keypoints:
            num_frames = chunk_data['keypoints'].shape[0]
            num_keypoints = chunk_data['keypoints'].shape[1]
            print(f"🔧 Creating PyTheia reconstruction from chunk: {num_frames} frames, {num_keypoints} keypoints")
        else:
            num_frames = chunk_data['camera_poses'].shape[0]
            print(f"🔧 Creating PyTheia reconstruction from chunk: {num_frames} frames, no keypoints available")

        # Add cameras to reconstruction
        t_add_cams0 = time.time()
        for frame_idx in range(num_frames):
            # Create view with actual image filename as name
            if 'image_paths' in chunk_data and chunk_data['image_paths']:
                # Extract filename from path
                image_path = chunk_data['image_paths'][frame_idx]
                if isinstance(image_path, list):
                    # Handle case where image_paths might be nested
                    image_path = image_path[0] if image_path else f"frame_{frame_idx}"
                import os
                view_name = os.path.basename(image_path)
            else:
                view_name = f"frame_{frame_idx}"
            
            timestamp_ns = frame_idx
            intrinsic_group_id = 0 if shared_intrinsics else frame_idx
            view_id = self.reconstruction.AddView(view_name, intrinsic_group_id, timestamp_ns)
            view = self.reconstruction.MutableView(view_id)
            
            # Create camera
            camera = Camera()
            
            # Use provided intrinsics or create default ones
            if shared_intrinsics:
                intrinsics = chunk_data['intrinsics'][0] #[frame_idx]
            else:
                intrinsics = chunk_data['intrinsics'][frame_idx]
            
            camera.create_from_intrinsics(intrinsics.cpu().numpy(), self.original_width, self.original_height, 1.0)
            
            # Set camera parameters
            camera_obj = view.MutableCamera()
            view.SetCameraIntrinsicsPrior(camera.prior)
            
            # Set camera pose
            pose = chunk_data['camera_poses'][frame_idx].cpu().numpy()
            camera_obj.SetPosition(pose[:3, 3])
            camera_obj.SetOrientationFromRotationMatrix(pose[:3, :3].T)
            view.SetIsEstimated(True)
            
            self.view_ids.append(view_id)
        t_add_cams1 = time.time()
        print(f"   ⏱️ Added {num_frames} cameras in {(t_add_cams1 - t_add_cams0):.3f}s")
        
        # Set camera intrinsics from priors
        t_intr0 = time.time()
        pt.sfm.SetCameraIntrinsicsFromPriors(self.reconstruction)
        t_intr1 = time.time()
        print(f"   ⏱️ Set intrinsics from priors in {(t_intr1 - t_intr0):.3f}s")
        
        # Add tracks and observations only if keypoints are available
        if has_keypoints:
            for frame_idx in range(num_frames):
                t_frame0 = time.time()
                # Get 3D points and colors for this frame
                points_3d = chunk_data['points'][frame_idx].cpu().numpy()  # (num_keypoints, 3)
                colors = chunk_data['colors'][frame_idx].cpu().numpy()  # (num_keypoints, 3)
                keypoints_2d = chunk_data['keypoints'][frame_idx].cpu().numpy()  # (num_keypoints, 2)
                
                # Create tracks for this frame
                t_tracks0 = time.time()
                frame_track_ids = []
                for kp_idx in range(keypoints_2d.shape[0]):
                    track_id = self.reconstruction.AddTrack()
                    track = self.reconstruction.MutableTrack(track_id)
                    track.SetPoint(np.hstack([points_3d[kp_idx], 1]))
                    track.SetColor(colors[kp_idx])
                    track.SetIsEstimated(True)
                    frame_track_ids.append(track_id)
                t_tracks1 = time.time()
                
                # Add observation for this frame
                t_obs_self0 = time.time()
                for kp_idx in range(keypoints_2d.shape[0]):
                    self.reconstruction.AddObservation(
                        self.view_ids[frame_idx], 
                        frame_track_ids[kp_idx], 
                        pt.sfm.Feature(keypoints_2d[kp_idx])
                    )
                t_obs_self1 = time.time()
                
                # Project keypoints to subset of other frames
                all_frames = [i for i in range(num_frames)]
                frame_idx_index = all_frames.index(frame_idx)
                all_frames_before = all_frames[:frame_idx_index]
                all_frames_after = all_frames[frame_idx_index + 1 : frame_idx_index + max_observations_per_track // 2 + 1]
                all_frames = all_frames_before + all_frames_after

                # Project points to all other frames
                t_proj0 = time.time()
                projected_points = self._project_points_to_other_cams(
                    chunk_data, frame_idx, all_frames
                )
                t_proj1 = time.time()
                
                added_obs = 0
                t_lk_total = 0.0
                t_obs_total = 0.0
                if use_lk_refinement:
                    # Prepare Lucas–Kanade refinement inputs (grayscale uint8 images)
                    try:
                        source_gray = self._get_gray_image(chunk_data, frame_idx)
                        target_grays = {t_idx: self._get_gray_image(chunk_data, t_idx) for t_idx in all_frames}
                    except Exception as _:
                        source_gray = None
                        target_grays = {}
                    
                    # Add observations for projected points (with LK refinement and bounds check)
                    _lk = lk_params or {}
                    lk_win = tuple(_lk.get('win', (9, 9)))
                    lk_levels = int(_lk.get('levels', 2))
                    lk_criteria = _lk.get('criteria', (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
                    deviation_thresh_px = float(_lk.get('max_deviation_px', 3.0))

                    for other_frame_idx, projected_kps in zip(all_frames, projected_points):
                        # default: use projections; optionally refine with LK if images available
                        final_points = projected_kps.copy()

                        if source_gray is not None and other_frame_idx in target_grays and target_grays[other_frame_idx] is not None:
                            t_lk0 = time.time()
                            prev_pts = chunk_data['keypoints'][frame_idx].cpu().numpy().astype(np.float32).reshape(-1, 1, 2)
                            next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
                                source_gray, target_grays[other_frame_idx], prev_pts, None,
                                winSize=lk_win, maxLevel=lk_levels, criteria=lk_criteria
                            )
                            t_lk1 = time.time()
                            t_lk_total += (t_lk1 - t_lk0)
                            if next_pts is not None and status is not None:
                                next_pts = next_pts.reshape(-1, 2)
                                status = status.reshape(-1)
                                # decide per point whether to accept LK or keep projection
                                for kp_idx in range(final_points.shape[0]):
                                    if status[kp_idx] == 1:
                                        lk_pt = next_pts[kp_idx]
                                        proj_pt = projected_kps[kp_idx]
                                        if np.linalg.norm(lk_pt - proj_pt) <= deviation_thresh_px:
                                            final_points[kp_idx] = lk_pt
                        
                        # Store for debug: projections vs final
                        if self._collect_debug:
                            try:
                                self._debug_pairs[(frame_idx, other_frame_idx)] = {
                                    'projected': projected_kps.copy(),
                                    'final': final_points.copy(),
                                }
                            except Exception:
                                pass

                        # Add observations for final points
                        t_obs0 = time.time()
                        for kp_idx, (track_id, final_pt) in enumerate(zip(frame_track_ids, final_points)):
                            if (0 <= final_pt[0] < self.original_width and 
                                0 <= final_pt[1] < self.original_height):
                                self.reconstruction.AddObservation(
                                    self.view_ids[other_frame_idx],
                                    track_id,
                                    pt.sfm.Feature(final_pt)
                                )
                                added_obs += 1
                        t_obs1 = time.time()
                        t_obs_total += (t_obs1 - t_obs0)
                else:
                    # LK disabled: add observations from projected points only
                    for other_frame_idx, projected_kps in zip(all_frames, projected_points):
                        # Store for debug: projections only
                        if self._collect_debug:
                            try:
                                self._debug_pairs[(frame_idx, other_frame_idx)] = {
                                    'projected': projected_kps.copy(),
                                    'final': projected_kps.copy(),
                                }
                            except Exception:
                                pass
                        t_obs0 = time.time()
                        for kp_idx, (track_id, projected_pt) in enumerate(zip(frame_track_ids, projected_kps)):
                            if (0 <= projected_pt[0] < self.original_width and 
                                0 <= projected_pt[1] < self.original_height):
                                self.reconstruction.AddObservation(
                                    self.view_ids[other_frame_idx],
                                    track_id,
                                    pt.sfm.Feature(projected_pt)
                                )
                                added_obs += 1
                        t_obs1 = time.time()
                        t_obs_total += (t_obs1 - t_obs0)

                t_frame1 = time.time()
                print(f"   Frame {frame_idx}: tracks {keypoints_2d.shape[0]} in {(t_tracks1 - t_tracks0):.3f}s, proj {(t_proj1 - t_proj0):.3f}s, LK {t_lk_total:.3f}s, obs {added_obs} in {t_obs_total:.3f}s, total {(t_frame1 - t_frame0):.3f}s")

        if self.use_inverse_depth:
            self.reconstruction.InitializeInverseDepth()

        print(f"✅ Created reconstruction with {len(self.reconstruction.ViewIds())} views and {len(self.reconstruction.TrackIds())} tracks")
        
        if run_bundle_adjustment:
            self.run_bundle_adjustment()
        
        return self.reconstruction

    def run_bundle_adjustment(self) -> None:
        """
        Run bundle adjustment on the current reconstruction in-place.
        """
        # Bundle adjust the reconstruction
        ba_options = pt.sfm.BundleAdjustmentOptions()
        ba_options.max_num_iterations = 10
        ba_options.use_inner_iterations = False
        ba_options.use_mixed_precision_solves = False
        ba_options.max_num_refinement_iterations = 1
        ba_options.verbose = False
        ba_options.num_threads = 20
        ba_options.linear_solver_type = pt.sfm.LinearSolverType.DENSE_SCHUR
        ba_options.preconditioner_type = pt.sfm.PreconditionerType.JACOBI
        ba_options.visibility_clustering_type = pt.sfm.VisibilityClusteringType.CANONICAL_VIEWS
        dense_type = pt.sfm.DenseLinearAlgebraLibraryType.CUDA if torch.cuda.is_available() and self.reconstruction.NumViews() > 100 else pt.sfm.DenseLinearAlgebraLibraryType.EIGEN
        ba_options.dense_linear_algebra_library_type = dense_type
        ba_options.sparse_linear_algebra_library_type = pt.sfm.SparseLinearAlgebraLibraryType.SUITE_SPARSE 

        if self.use_inverse_depth:
            ba_options.use_homogeneous_point_parametrization = False
            ba_options.use_inverse_depth_parametrization = True
        else:
            ba_options.use_homogeneous_point_parametrization = True
            ba_options.use_inverse_depth_parametrization = False

        ba_options.robust_loss_width = 2.0
        ba_options.loss_function_type = pt.sfm.LossFunctionType.HUBER
        pt.sfm.BundleAdjustReconstruction(ba_options, self.reconstruction)

        removed_tracks = pt.sfm.SetOutlierTracksToUnestimated(
            set(self.reconstruction.TrackIds()), 2, 0.25, self.reconstruction)
        print(f"   Removed {removed_tracks} tracks after bundle adjustment")

    def _get_gray_image(self, chunk_data: Dict, frame_idx: int) -> np.ndarray:
        """
        Return uint8 grayscale image (H, W) for the given frame index, caching and saving to disk.
        Images are resized to (original_height, original_width).
        """
        if frame_idx in self._gray_image_cache:
            return self._gray_image_cache[frame_idx]

        # Do not write images to disk here; only use in-memory cache or precomputed gray_images

        # Prefer precomputed gray_images in chunk_data to avoid recomputation
        try:
            if 'gray_images' in chunk_data and frame_idx < len(chunk_data['gray_images']):
                g = chunk_data['gray_images'][frame_idx]
                if isinstance(g, torch.Tensor):
                    gray_u8 = g.detach().cpu().numpy()
                else:
                    gray_u8 = np.array(g)
                if gray_u8.dtype != np.uint8:
                    gray_u8 = gray_u8.astype(np.uint8)
                # Ensure HxW
                if gray_u8.ndim == 3:
                    gray_u8 = gray_u8.squeeze()
                # Ensure size
                if gray_u8.shape[0] != self.original_height or gray_u8.shape[1] != self.original_width:
                    gray_u8 = cv2.resize(gray_u8, (self.original_width, self.original_height), interpolation=cv2.INTER_AREA)
                self._gray_image_cache[frame_idx] = gray_u8
                return gray_u8
        except Exception:
            pass
        # If not available, return None (caller can handle fallback behavior)
        return None

    def debug_projections(self, chunk_data: Dict, source_frame: int, target_frames: List[int], 
                         save_path: Optional[str] = None) -> None:
        """
        Visualize projections vs. final observations used during reconstruction, to validate LK refinement.
        
        This plots, for each target frame, the projected points (green) and the actually used points
        (magenta) as stored in `self._debug_pairs[(source_frame, target_frame)]`.
        """
        import time
        _t0 = time.time()

        if not target_frames:
            print("⚠️  debug_projections: no target_frames provided")
            return

        # Build image list (prefer stored grayscale to avoid disk I/O)
        images = []
        frames_to_fetch = [source_frame] + target_frames
        if 'gray_images' in chunk_data and chunk_data['gray_images'] is not None:
            for frame_idx in frames_to_fetch:
                if frame_idx < len(chunk_data['gray_images']):
                    g = chunk_data['gray_images'][frame_idx]
                    if isinstance(g, torch.Tensor):
                        g = g.cpu().numpy()
                    if g.ndim == 3:
                        g = np.squeeze(g)
                    if g.dtype != np.uint8:
                        g = g.astype(np.uint8)
                    images.append(g)
                else:
                    images.append(np.zeros((self.original_height, self.original_width), dtype=np.uint8))
        elif 'images' in chunk_data and chunk_data['images'] is not None:
            for frame_idx in frames_to_fetch:
                if frame_idx < len(chunk_data['images']):
                    img = chunk_data['images'][frame_idx]
                    if isinstance(img, torch.Tensor):
                        img = img.cpu().numpy()
                    if img.ndim == 3 and img.shape[0] == 3:
                        img = np.transpose(img, (1, 2, 0))
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    images.append(img)
                else:
                    images.append(np.zeros((self.original_height, self.original_width, 3), dtype=np.uint8))
        else:
            for _ in range(len(target_frames) + 1):
                images.append(np.zeros((self.original_height, self.original_width), dtype=np.uint8))

        # Create side-by-side plot: source + per-target overlay of projected vs final
        fig, axes = plt.subplots(1, len(target_frames) + 1, figsize=(3.2 * (len(target_frames) + 1), 3.0))
        if len(target_frames) == 0:
            axes = [axes]

        # Plot source frame with original keypoints (red)
        ax = axes[0]
        if images[0] is not None:
            if images[0].ndim == 2:
                ax.imshow(images[0], cmap='gray', vmin=0, vmax=255)
            else:
                ax.imshow(images[0])
        src_kp = chunk_data['keypoints'][source_frame].cpu().numpy()
        ax.scatter(src_kp[:, 0], src_kp[:, 1], c='red', s=8, alpha=0.7)
        ax.set_title(f'Source {source_frame}\nK={len(src_kp)}', fontsize=10)
        ax.axis('off')

        # Plot per target: projected (green) vs final (magenta)
        for i, target_frame in enumerate(target_frames):
            ax = axes[i + 1]
            if images[i + 1] is not None:
                if images[i + 1].ndim == 2:
                    ax.imshow(images[i + 1], cmap='gray', vmin=0, vmax=255)
                else:
                    ax.imshow(images[i + 1])

            pair = self._debug_pairs.get((source_frame, target_frame), None)
            if pair is not None:
                projected = pair['projected']
                final = pair['final']
                # Bounds mask for clarity
                valid_mask = ((0 <= projected[:, 0]) & (projected[:, 0] < self.original_width) &
                              (0 <= projected[:, 1]) & (projected[:, 1] < self.original_height))
                proj_valid = projected[valid_mask]
                fin_valid = final[valid_mask]
                if len(proj_valid) > 0:
                    ax.scatter(proj_valid[:, 0], proj_valid[:, 1], c='lime', s=10, alpha=0.8, label='Projected')
                if len(fin_valid) > 0:
                    ax.scatter(fin_valid[:, 0], fin_valid[:, 1], c='magenta', s=10, alpha=0.8, label='Final')
                if len(proj_valid) > 0 or len(fin_valid) > 0:
                    ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center', va='center')

            ax.set_title(f'Target {target_frame}', fontsize=10)
            ax.axis('off')

        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=110, bbox_inches='tight')
            print(f"💾 Saved debug overlay to: {save_path}")
            plt.close(fig)
        else:
            print(f"   ⏱️ debug overlay time: {(time.time()-_t0)*1000:.1f}ms")
            plt.show()
    
    def _project_points_to_other_cams(self, chunk_data: Dict, source_frame: int, target_frames: List[int]) -> List[np.ndarray]:
        """
        Project keypoints from source frame to target frames.
        
        Args:
            chunk_data: Chunk data containing camera poses and intrinsics
            source_frame: Source frame index
            target_frames: List of target frame indices
        
        Returns:
            List of projected keypoint coordinates for each target frame
        """
        # Get source camera pose and intrinsics
        source_pose = chunk_data['camera_poses'][source_frame].cpu().numpy()
        source_intrinsics = chunk_data['intrinsics'][source_frame].cpu().numpy() if 'intrinsics' in chunk_data else None
        
        # Get source keypoints
        source_keypoints = chunk_data['keypoints'][source_frame]  # (num_keypoints, 2)
        source_points_3d = chunk_data['points'][source_frame]  # (num_keypoints, 3)
        
        projected_points_list = []
        
        for target_frame in target_frames:
            # Get target camera pose and intrinsics
            target_pose = chunk_data['camera_poses'][target_frame].cpu().numpy()
            target_intrinsics = chunk_data['intrinsics'][target_frame].cpu().numpy() if 'intrinsics' in chunk_data else None
            
            # Project 3D points to target camera
            projected_points = self._project_3d_points_to_camera(
                source_points_3d, source_pose, target_pose, target_intrinsics
            )
            
            projected_points_list.append(projected_points[:, :2])
        
        return projected_points_list
    
    def _project_3d_points_to_camera(self, points_3d: torch.Tensor, source_pose: np.ndarray, 
                                   target_pose: np.ndarray, target_intrinsics: np.ndarray) -> np.ndarray:
        """
        Project 3D points from source camera to target camera.
        
        Args:
            points_3d: 3D points in world coordinates (num_keypoints, 3)
            source_pose: Source camera pose (4, 4)
            target_pose: Target camera pose (4, 4)
            target_intrinsics: Target camera intrinsics (3, 3)
        
        Returns:
            Projected 2D points in target camera (num_keypoints, 2)
        """
        # Transform points from world to target camera coordinates
        world_to_target = np.linalg.inv(target_pose)
        points_in_target = (world_to_target @ np.hstack([points_3d.cpu().numpy(), np.ones((points_3d.shape[0], 1))]).T).T
        
        # Project to 2D
        points_2d = points_in_target[:, :3] / points_in_target[:, 2:3]
        
        # Apply camera intrinsics
        if target_intrinsics is not None:
            projected_points = (target_intrinsics @ points_2d.T).T
        else:
            # Use normalized coordinates if no intrinsics provided
            projected_points = points_2d[:, :2]
        
        return projected_points
    
    def get_reconstruction_stats(self) -> Dict:
        """
        Get statistics about the reconstruction.
        
        Returns:
            Dictionary containing reconstruction statistics
        """
        num_views = self.reconstruction.NumViews()
        num_tracks = self.reconstruction.NumTracks()
        
        # Count observations
        total_observations = 0
        for track_id in self.reconstruction.TrackIds():
            track = self.reconstruction.Track(track_id)
            total_observations += track.NumViews()
        
        return {
            'num_views': num_views,
            'num_tracks': num_tracks,
            'total_observations': total_observations,
            'avg_observations_per_track': total_observations / num_tracks if num_tracks > 0 else 0
        }
    
    def print_reconstruction_summary(self) -> None:
        """Print a summary of the reconstruction."""
        stats = self.get_reconstruction_stats()
        
        print("📊 PyTheia Reconstruction Summary:")
        print(f"   Views: {stats['num_views']}")
        print(f"   Tracks: {stats['num_tracks']}")
        print(f"   Total observations: {stats['total_observations']}")
        print(f"   Average observations per track: {stats['avg_observations_per_track']:.1f}")
    
    def save_reconstruction(self, filepath: str) -> None:
        """
        Save reconstruction to file.
        
        Args:
            filepath: Path to save the reconstruction
        """
        try:
            pt.io.WritePlyFile(filepath.replace('.sfm', '.ply'), self.reconstruction, 
                               np.random.randint(0,255, (3)).tolist(),1)
            pt.io.WriteReconstruction(self.reconstruction, filepath)
            print(f"💾 Saved reconstruction to: {filepath}")
        except Exception as e:
            print(f"❌ Error saving reconstruction: {e}")
    
    def load_reconstruction(self, filepath: str) -> None:
        """
        Load reconstruction from file.
        
        Args:
            filepath: Path to load the reconstruction from
        """
        try:
            self.reconstruction = pt.io.ReadReconstruction(filepath)
            print(f"📂 Loaded reconstruction from: {filepath}")
        except Exception as e:
            print(f"❌ Error loading reconstruction: {e}") 
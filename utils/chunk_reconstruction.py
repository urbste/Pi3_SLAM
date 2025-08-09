"""
Chunk-based PyTheia reconstruction utilities for Pi3SLAM.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import pytheia as pt
from pi3.utils.camera import Camera
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

    def set_target_size(self, original_width: int, original_height: int):
        """
        Set the target size for the reconstruction.
        """
        self.original_width = original_width
        self.original_height = original_height

    def create_recon_from_chunk(self, chunk_data: Dict, max_observations_per_track: int = 5) -> pt.sfm.Reconstruction:
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
        
        Returns:
            PyTheia reconstruction object
        """
        # Clear previous reconstruction
        self.reconstruction = pt.sfm.Reconstruction()
        self.view_ids = []
        self.track_ids = []
        
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
            view_id = self.reconstruction.AddView(view_name, frame_idx, timestamp_ns)
            view = self.reconstruction.MutableView(view_id)
            
            # Create camera
            camera = Camera()
            
            # Use provided intrinsics or create default ones
            if 'intrinsics' in chunk_data:
                intrinsics = chunk_data['intrinsics'][frame_idx]
            else:
                # Create default intrinsics (principal point at center)
                fx = fy = max(self.original_width, self.original_height)
                cx = self.original_width / 2
                cy = self.original_height / 2
                intrinsics = torch.tensor([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
            
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
        
        # Set camera intrinsics from priors
        pt.sfm.SetCameraIntrinsicsFromPriors(self.reconstruction)
        
        # Add tracks and observations only if keypoints are available
        if has_keypoints:
            for frame_idx in range(num_frames):
                # Get 3D points and colors for this frame
                points_3d = chunk_data['points'][frame_idx].cpu().numpy()  # (num_keypoints, 3)
                colors = chunk_data['colors'][frame_idx].cpu().numpy()  # (num_keypoints, 3)
                keypoints_2d = chunk_data['keypoints'][frame_idx].cpu().numpy()  # (num_keypoints, 2)
                masks = chunk_data['masks'][frame_idx].cpu().numpy()  # (num_keypoints, 1)
                #descriptors = chunk_data['descriptors'][frame_idx].cpu().numpy()  # (num_keypoints, 128)

                # Create tracks for this frame
                frame_track_ids = []
                
                for kp_idx in range(keypoints_2d.shape[0]):
                    
                    # Create track
                    track_id = self.reconstruction.AddTrack()
                    track = self.reconstruction.MutableTrack(track_id)
                    
                    # Set 3D point (homogeneous)
                    track.SetPoint(np.hstack([points_3d[kp_idx], 1]))
                    
                    # Set color
                    track.SetColor(colors[kp_idx])
                    track.SetIsEstimated(True)
                    #track.SetReferenceDescriptor(descriptors[kp_idx])
                    
                    frame_track_ids.append(track_id)
                    
                    # Add observation for this frame
                    self.reconstruction.AddObservation(
                        self.view_ids[frame_idx], 
                        track_id, 
                        pt.sfm.Feature(keypoints_2d[kp_idx])
                    )
                
                # Project keypoints to subset of other frames
                all_frames = [i for i in range(num_frames)]
                frame_idx_index = all_frames.index(frame_idx)
                all_frames_before = all_frames[:frame_idx_index]
                all_frames_after = all_frames[frame_idx_index + 1 : frame_idx_index + max_observations_per_track // 2 + 1]
                all_frames = all_frames_before + all_frames_after

                # Project points to all other frames
                projected_points = self._project_points_to_other_cams(
                    chunk_data, frame_idx, all_frames
                )
                
                # Add observations for projected points (with limit per track)
                for other_frame_idx, projected_kps in zip(all_frames, projected_points):
                    for kp_idx, (track_id, projected_pt) in enumerate(zip(frame_track_ids, projected_kps)):
                        # Check if projected point is within image bounds
                        if (0 <= projected_pt[0] < self.original_width and 
                            0 <= projected_pt[1] < self.original_height):
                            # Only add observation if we haven't reached the limit
                            self.reconstruction.AddObservation(
                                self.view_ids[other_frame_idx],
                                track_id,
                                pt.sfm.Feature(projected_pt)
                            )

    
        print(f"✅ Created reconstruction with {len(self.reconstruction.ViewIds())} views and {len(self.reconstruction.TrackIds())} tracks")
        # Bundle adjust the reconstruction
        ba_options = pt.sfm.BundleAdjustmentOptions()
        ba_options.max_num_iterations = 5
        ba_options.verbose = False
        ba_options.robust_loss_width = 2.0
        ba_options.loss_function_type = pt.sfm.LossFunctionType.HUBER
        ba_summary = pt.sfm.BundleAdjustReconstruction(ba_options, self.reconstruction)

        removed_tracks = pt.sfm.SetOutlierTracksToUnestimated(
            set(self.reconstruction.TrackIds()), 2, 0.25, self.reconstruction)
        print(f"   Removed {removed_tracks} tracks after initial bundle adjustment")

        return self.reconstruction
    
    def debug_projections(self, chunk_data: Dict, source_frame: int, target_frames: List[int], 
                         save_path: Optional[str] = None, fps: int = 1) -> None:
        """
        Debug keypoint projections from source frame to target frames.
        
        This method visualizes how keypoints from a source frame are projected onto target frames,
        which is useful for debugging the reconstruction process.
        
        Args:
            chunk_data: Dictionary containing chunk data with keypoints, camera poses, and intrinsics
            source_frame: Index of the source frame
            target_frames: List of target frame indices to project to
            save_path: Optional path to save the visualization as GIF (e.g., "debug_projection.gif")
            fps: Frames per second for the GIF animation
        """
        print(f"🔍 Debugging keypoint projections from frame {source_frame} to frames {target_frames}")
        import time
        _t0 = time.time()
        
        # Check if keypoints are available
        if not all(key in chunk_data for key in ['keypoints', 'points', 'camera_poses']):
            print("❌ Missing required data for debug_projections")
            return
        
        # Determine number of frames available
        def _num_frames(x):
            try:
                import torch
                if isinstance(x, torch.Tensor):
                    return int(x.shape[0])
            except Exception:
                pass
            try:
                return len(x)
            except Exception:
                return 0

        num_frames = _num_frames(chunk_data.get('camera_poses', []))

        # Validate source and target frames against available frames
        if not (0 <= source_frame < num_frames):
            print(f"❌ debug_projections: source_frame {source_frame} out of range [0, {num_frames-1}] — skipping")
            return

        # Filter target_frames to valid range and exclude source_frame
        filtered_targets = [f for f in target_frames if 0 <= f < num_frames and f != source_frame]
        dropped = [f for f in target_frames if f not in filtered_targets]
        if dropped:
            print(f"⚠️  debug_projections: dropped out-of-range targets {dropped}; valid range is [0, {num_frames-1}]")
        target_frames = filtered_targets
        if len(target_frames) == 0:
            print("⚠️  debug_projections: no valid target frames after filtering — skipping")
            return

        # Get source frame data
        source_keypoints = chunk_data['keypoints'][source_frame].cpu().numpy()  # (num_keypoints, 2)
        source_points_3d = chunk_data['points'][source_frame].cpu().numpy()  # (num_keypoints, 3)
        
        print(f"   Source frame {source_frame}: {len(source_keypoints)} keypoints")
        
        # Project points to target frames
        _t0_proj = time.time()
        projected_points_list = self._project_points_to_other_cams(chunk_data, source_frame, target_frames)
        print(f"   ⏱️ projection time: {(time.time()-_t0_proj)*1000:.1f}ms for {len(target_frames)} targets")
        
        # Load images if available
        images = []
        if 'images' in chunk_data:
            for frame_idx in [source_frame] + target_frames:
                if frame_idx < len(chunk_data['images']):
                    img = chunk_data['images'][frame_idx]
                    if isinstance(img, torch.Tensor):
                        img = img.cpu().numpy()
                    
                    # Convert from CHW to HWC if needed
                    if len(img.shape) == 3 and img.shape[0] == 3:
                        img = np.transpose(img, (1, 2, 0))
                    
                    # Normalize to 0-255 range if needed
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    
                    images.append(img)
                else:
                    images.append(None)
        else:
            # Create blank images if no images available
            for _ in range(len(target_frames) + 1):
                blank_img = np.zeros((self.original_height, self.original_width, 3), dtype=np.uint8)
                images.append(blank_img)
        
        # Create single plot for video-style GIF
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # For static visualization, show side-by-side comparison
        if not save_path or not save_path.lower().endswith('.gif'):
            # Create side-by-side visualization for static images
            fig, axes = plt.subplots(1, len(target_frames) + 1, figsize=(2.5 * (len(target_frames) + 1), 2.5))
            if len(target_frames) == 0:
                axes = [axes]
            
            # Plot source frame
            ax = axes[0]
            if images[0] is not None:
                ax.imshow(images[0])
            ax.scatter(source_keypoints[:, 0], source_keypoints[:, 1], c='red', s=8, alpha=0.7)
            ax.set_title(f'Source Frame {source_frame}\n{len(source_keypoints)} keypoints', fontsize=10)
            ax.axis('off')
            
            # Plot target frames with projected keypoints
            for i, (target_frame, projected_points) in enumerate(zip(target_frames, projected_points_list)):
                ax = axes[i + 1]
                if images[i + 1] is not None:
                    ax.imshow(images[i + 1])
                
                # Filter projected points that are within image bounds
                valid_mask = ((0 <= projected_points[:, 0]) & (projected_points[:, 0] < self.original_width) &
                             (0 <= projected_points[:, 1]) & (projected_points[:, 1] < self.original_height))
                
                valid_projected = projected_points[valid_mask]
                invalid_projected = projected_points[~valid_mask]
                
                # Plot valid projected keypoints in green (smaller size)
                if len(valid_projected) > 0:
                    ax.scatter(valid_projected[:, 0], valid_projected[:, 1], c='green', s=8, alpha=0.7, label='Valid')
                
                # Plot invalid projected keypoints in red (outside bounds, smaller size)
                if len(invalid_projected) > 0:
                    ax.scatter(invalid_projected[:, 0], invalid_projected[:, 1], c='red', s=8, alpha=0.7, label='Invalid')
                
                ax.set_title(f'Target Frame {target_frame}\n{len(valid_projected)}/{len(projected_points)} valid', fontsize=10)
                ax.axis('off')
                
                if len(valid_projected) > 0 or len(invalid_projected) > 0:
                    ax.legend(fontsize=8)
            
            plt.tight_layout()
        
        # Save visualization if path provided
        if save_path:
            if save_path.lower().endswith('.gif'):
                # Create video-style animated GIF that shows sequential frames with keypoints
                def animate(frame_idx):
                    ax.clear()
                    ax.axis('off')
                    
                    if frame_idx == 0:
                        # Show source frame with source keypoints
                        if images[0] is not None:
                            ax.imshow(images[0])
                        ax.scatter(source_keypoints[:, 0], source_keypoints[:, 1], c='red', s=12, alpha=0.8)
                        ax.set_title(f'Source Frame {source_frame} - Original Keypoints\n{len(source_keypoints)} keypoints', fontsize=12, pad=10)
                    
                    elif frame_idx <= len(target_frames):
                        # Show target frames with projected keypoints
                        target_idx = frame_idx - 1
                        target_frame = target_frames[target_idx]
                        projected_points = projected_points_list[target_idx]
                        
                        if images[target_idx + 1] is not None:
                            ax.imshow(images[target_idx + 1])
                        
                        # Filter projected points that are within image bounds
                        valid_mask = ((0 <= projected_points[:, 0]) & (projected_points[:, 0] < self.original_width) &
                                     (0 <= projected_points[:, 1]) & (projected_points[:, 1] < self.original_height))
                        
                        valid_projected = projected_points[valid_mask]
                        invalid_projected = projected_points[~valid_mask]
                        
                        # Plot valid projected keypoints in bright green
                        if len(valid_projected) > 0:
                            ax.scatter(valid_projected[:, 0], valid_projected[:, 1], c='lime', s=12, alpha=0.8, 
                                     edgecolors='darkgreen', linewidths=0.5)
                        
                        # Plot invalid projected keypoints in red (outside bounds)
                        if len(invalid_projected) > 0:
                            ax.scatter(invalid_projected[:, 0], invalid_projected[:, 1], c='red', s=12, alpha=0.8,
                                     edgecolors='darkred', linewidths=0.5)
                        
                        ax.set_title(f'Frame {target_frame} - Projected Keypoints\n{len(valid_projected)} valid, {len(invalid_projected)} invalid', 
                                   fontsize=12, pad=10)
                    
                    # Set consistent axis limits
                    ax.set_xlim(0, self.original_width)
                    ax.set_ylim(self.original_height, 0)  # Invert y-axis for image coordinates
                    
                    return [ax]
                
                # Create animation that shows source frame + all target frames
                total_frames = 1 + len(target_frames)  # Source frame + target frames
                anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                             interval=max(500, 1000//fps), repeat=True)
                
                # Ensure directory exists
                import os
                save_dir = os.path.dirname(save_path)
                if save_dir:  # Only create directory if path contains one
                    os.makedirs(save_dir, exist_ok=True)
                
                anim.save(save_path, writer='pillow', fps=fps)
                print(f"💾 Saved video-style projection GIF to: {save_path}")
                plt.close(fig)  # Close figure to save memory
            else:
                # Save as static image
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                print(f"💾 Saved debug projection image to: {save_path}")
                plt.close(fig)  # Close figure to save memory
        else:
            print(f"   ⏱️ total debug projection time: {(time.time()-_t0)*1000:.1f}ms")
            plt.show()
        
        # Print statistics
        print(f"\n📊 Projection Statistics:")
        print(f"   Source frame {source_frame}: {len(source_keypoints)} keypoints")
        for target_frame, projected_points in zip(target_frames, projected_points_list):
            valid_mask = ((0 <= projected_points[:, 0]) & (projected_points[:, 0] < self.original_width) &
                         (0 <= projected_points[:, 1]) & (projected_points[:, 1] < self.original_height))
            valid_count = np.sum(valid_mask)
            total_count = len(projected_points)
            print(f"   Target frame {target_frame}: {valid_count}/{total_count} valid projections ({100*valid_count/total_count:.1f}%)")
    
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
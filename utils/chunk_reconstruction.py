"""
Chunk-based PyTheia reconstruction utilities for Pi3SLAM.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import pytheia as pt
from pi3.utils.camera import Camera


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

    def create_recon_from_chunk(self, chunk_data: Dict) -> pt.sfm.Reconstruction:
        """
        Create PyTheia reconstruction from chunk data.
        
        Args:
            chunk_data: Dictionary containing:
                - 'keypoints': Keypoint coordinates (N, num_keypoints, 2)
                - 'colors': Keypoint colors (N, num_keypoints, 3)
                - 'points_kp': 3D world points for keypoints (N, num_keypoints, 3)
                - 'camera_poses': Camera poses (N, 4, 4)
                - 'intrinsics': Camera intrinsics (N, 3, 3) - optional
                - 'conf_kp': Keypoint confidences (N, num_keypoints) - optional
                - 'masks_kp': Keypoint masks (N, num_keypoints) - optional
        
        Returns:
            PyTheia reconstruction object
        """
        # Clear previous reconstruction
        self.reconstruction = pt.sfm.Reconstruction()
        self.view_ids = []
        self.track_ids = []
        
        num_frames = chunk_data['keypoints'].shape[0]
        num_keypoints = chunk_data['keypoints'].shape[1]
        
        print(f"🔧 Creating PyTheia reconstruction from chunk: {num_frames} frames, {num_keypoints} keypoints")
        
        # Add cameras to reconstruction
        for frame_idx in range(num_frames):
            # Create view
            timestamp_ns = frame_idx
            view_id = self.reconstruction.AddView(str(timestamp_ns), frame_idx, timestamp_ns)
            view = self.reconstruction.MutableView(view_id)
            
            # Create camera
            camera = Camera()
            
            # Use provided intrinsics or create default ones
            if 'intrinsics' in chunk_data:
                intrinsics = chunk_data['intrinsics'][frame_idx].cpu().numpy()
            else:
                # Create default intrinsics (principal point at center)
                fx = fy = max(self.original_width, self.original_height)
                cx = self.original_width / 2
                cy = self.original_height / 2
                intrinsics = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
            
            camera.create_from_intrinsics(intrinsics, self.original_width, self.original_height, 1.0)
            
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
        
        # Add tracks for each frame
        for frame_idx in range(num_frames):
            # Get 3D points and colors for this frame
            points_3d = chunk_data['points_kp'][frame_idx].cpu().numpy()  # (num_keypoints, 3)
            colors = chunk_data['colors'][frame_idx].cpu().numpy()  # (num_keypoints, 3)
            keypoints_2d = chunk_data['keypoints'][frame_idx].cpu().numpy()  # (num_keypoints, 2)
            descriptors = chunk_data['descriptors'][frame_idx].cpu().numpy()  # (num_keypoints, 128)

            # Create tracks for this frame
            frame_track_ids = []
            
            for kp_idx in range(num_keypoints):
                # Create track
                track_id = self.reconstruction.AddTrack()
                track = self.reconstruction.MutableTrack(track_id)
                
                # Set 3D point
                # make point homogeneous
                track.SetPoint(np.hstack([points_3d[kp_idx], 1]))
                
                # Set color
                track.SetColor(colors[kp_idx])
                track.SetIsEstimated(True)
                track.SetReferenceDescriptor(descriptors[kp_idx])
                
                frame_track_ids.append(track_id)
                
                # Add observation for this frame
                self.reconstruction.AddObservation(
                    self.view_ids[frame_idx], 
                    track_id, 
                    pt.sfm.Feature(keypoints_2d[kp_idx])
                )
            
            # Project keypoints to all subsequent frames
            next_frames = list(range(frame_idx + 1, num_frames))
            
            if next_frames:
                # Project points to neighboring frames
                projected_points = self._project_points_to_other_cams(
                    chunk_data, frame_idx, next_frames
                )
                
                # Add observations for projected points
                for next_frame_idx, projected_kps in zip(next_frames, projected_points):
                    for kp_idx, (track_id, projected_pt) in enumerate(zip(frame_track_ids, projected_kps)):
                        # Check if projected point is within image bounds
                        if (0 <= projected_pt[0] < self.original_width and 
                            0 <= projected_pt[1] < self.original_height):
                            self.reconstruction.AddObservation(
                                self.view_ids[next_frame_idx],
                                track_id,
                                pt.sfm.Feature(projected_pt)
                            )
            
            self.track_ids.extend(frame_track_ids)
        
        print(f"✅ Created reconstruction with {len(self.view_ids)} views and {len(self.track_ids)} tracks")
        return self.reconstruction
    
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
        source_points_3d = chunk_data['points_kp'][source_frame]  # (num_keypoints, 3)
        
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
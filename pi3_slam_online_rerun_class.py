import torch
import numpy as np
from typing import Dict, List, Tuple
from pi3.models.pi3 import Pi3
from pi3.utils.basic import  write_ply, TORCHCODEC_AVAILABLE
from pi3.utils.geometry import depth_edge, homogenize_points
import time
import multiprocessing as mp
import rerun as rr
import os
from torch.utils.data import Dataset, DataLoader



def rodrigues_to_rotation_matrix(rodrigues: np.ndarray) -> np.ndarray:
    """
    Convert Rodrigues parameters to rotation matrix.
    
    Args:
        rodrigues: Rodrigues parameters (3,) - rotation vector
    
    Returns:
        Rotation matrix (3, 3)
    """
    theta = np.linalg.norm(rodrigues)
    if theta < 1e-8:
        return np.eye(3)
    
    k = rodrigues / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def rotation_matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to Rodrigues parameters.
    
    Args:
        R: Rotation matrix (3, 3)
    
    Returns:
        Rodrigues parameters (3,) - rotation vector
    """
    # Ensure R is a proper rotation matrix
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    
    # Extract rotation angle and axis
    trace = np.trace(R)
    theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    
    if theta < 1e-8:
        return np.zeros(3)
    
    # Extract rotation axis
    K = (R - R.T) / (2 * np.sin(theta))
    rodrigues = np.array([K[2, 1], K[0, 2], K[1, 0]])
    rodrigues = rodrigues * theta
    
    return rodrigues


class ChunkImageDataset(Dataset):
    """
    Custom dataset for loading entire chunks of images at once.
    This is optimized for batch processing where we need chunk_length images at a time.
    """
    
    def __init__(self, image_paths: List, chunk_length: int, overlap: int, 
                 target_size: Tuple[int, int], device: str = 'cpu', 
                 undistortion_maps=None):
        """
        Initialize the chunk dataset.
        
        Args:
            image_paths: List of image paths or (video_path, frame_idx) tuples
            chunk_length: Number of frames per chunk
            overlap: Number of overlapping frames between chunks
            target_size: Target size (height, width) for resizing
            device: Device to load images on
            undistortion_maps: Optional UndistortionMaps object for applying undistortion
        """
        self.image_paths = image_paths
        self.chunk_length = chunk_length
        self.overlap = overlap
        self.target_size = target_size
        self.device = device
        self.undistortion_maps = undistortion_maps
        
        # Calculate chunk indices
        self.chunk_indices = []
        chunk_idx = 0
        while chunk_idx < len(image_paths):
            end_idx = min(chunk_idx + chunk_length, len(image_paths))
            if end_idx - chunk_idx >= 2:  # At least 2 frames required
                self.chunk_indices.append((chunk_idx, end_idx))
            chunk_idx += chunk_length - overlap
        
        # Check if we're dealing with video files for optimization
        self.is_video = isinstance(image_paths[0], tuple) if image_paths else False
        if self.is_video:
            self.video_path = image_paths[0][0]  # All frames should be from the same video
            print(f"🎬 Video detected: {self.video_path}")
            print(f"   Optimizing for video chunk loading with {len(self.chunk_indices)} chunks")
            
            # Initialize video undistortion loader if undistortion maps are provided
            if undistortion_maps is not None:
                try:
                    from pi3.utils.undistortion import VideoUndistortionLoader
                    self.video_loader = VideoUndistortionLoader(undistortion_maps, device=device)
                    print(f"   Using VideoUndistortionLoader for efficient video processing")
                except ImportError:
                    print(f"   VideoUndistortionLoader not available, using standard loading")
                    self.video_loader = None
            else:
                self.video_loader = None
        
    def __len__(self):
        return len(self.chunk_indices)
    
    def __getitem__(self, idx):
        """Load and preprocess an entire chunk of images."""
        start_idx, end_idx = self.chunk_indices[idx]
        chunk_paths = self.image_paths[start_idx:end_idx]
        
        if self.is_video:
            # Optimized video chunk loading
            images = self._load_video_chunk(chunk_paths)
        else:
            # Regular image loading
            images = self._load_image_chunk(chunk_paths)
        
        # Stack images into a chunk tensor
        chunk_tensor = torch.stack(images)
        return {
            'chunk': chunk_tensor.to(self.device),
            'start_idx': torch.tensor([start_idx]),
            'end_idx': torch.tensor([end_idx]),
            'chunk_paths': [chunk_paths]  # Wrap in list for batching
        }
    
    def _load_video_chunk(self, chunk_paths: List) -> List[torch.Tensor]:
        """Load a chunk of video frames efficiently using bulk loading when possible."""
        if not chunk_paths:
            return []
        
        # Extract video path and frame indices
        video_path = chunk_paths[0][0]  # All frames should be from the same video
        frame_indices = [path_info[1] for path_info in chunk_paths]
        
        # Use VideoUndistortionLoader if available (most efficient for undistorted video)
        if hasattr(self, 'video_loader') and self.video_loader is not None:
            try:
                # Load frames with undistortion in bulk
                frames_tensor = self.video_loader.load_and_undistort_frames(
                    video_path, frame_indices, self.target_size
                )
                # Return individual frames as list
                return [frames_tensor[i] for i in range(frames_tensor.shape[0])]
            except Exception as e:
                print(f"Warning: VideoUndistortionLoader failed, falling back to standard loading: {e}")
        
        # Try to use bulk loading with torchcodec if available
        try:
            if TORCHCODEC_AVAILABLE:
                from torchcodec.decoders import VideoDecoder
                
                # Load frames in bulk using torchcodec
                decoder = VideoDecoder(video_path, device="cpu")
                frames_tensor = decoder.get_frames_at(indices=frame_indices)  # Shape: [N, C, H, W], uint8
                del decoder  # Clean up decoder
                
                # Convert to float [0, 1] range
                frames_tensor = frames_tensor.data.float() / 255.0
                
                # Resize if target size is specified
                if self.target_size is not None:
                    frames_tensor = torch.nn.functional.interpolate(
                        frames_tensor, size=self.target_size, mode='bilinear', align_corners=False
                    )
                
                # Apply undistortion if maps are provided
                if self.undistortion_maps is not None:
                    undistorted_frames = []
                    for i in range(frames_tensor.shape[0]):
                        # Convert tensor to numpy for undistortion
                        frame_np = frames_tensor[i].permute(1, 2, 0).numpy()  # CHW to HWC
                        undistorted_np = self.undistortion_maps.undistort_image(frame_np, self.target_size)
                        # Convert back to tensor
                        undistorted_frame = torch.from_numpy(undistorted_np).float() / 255.0
                        undistorted_frame = undistorted_frame.permute(2, 0, 1)  # HWC to CHW
                        undistorted_frames.append(undistorted_frame)
                    return undistorted_frames
                else:
                    # Return individual frames as list
                    return [frames_tensor[i] for i in range(frames_tensor.shape[0])]
        
        except Exception as e:
            print(f"Warning: Bulk video loading failed, falling back to individual loading: {e}")
        
        # Fallback to individual frame loading
        images = []
        for path_info in chunk_paths:
            video_path, frame_idx = path_info
            from pi3.utils.basic import load_video_frame
            
            # Load frame using torchcodec (with fallback to OpenCV)
            image = load_video_frame(video_path, frame_idx, self.target_size, use_torchcodec=True)
            
            # Apply undistortion if maps are provided
            if self.undistortion_maps is not None:
                # Convert tensor to numpy for undistortion
                image_np = image.permute(1, 2, 0).numpy()  # CHW to HWC
                undistorted_np = self.undistortion_maps.undistort_image(image_np, self.target_size)
                # Convert back to tensor
                image = torch.from_numpy(undistorted_np).float() / 255.0
                image = image.permute(2, 0, 1)  # HWC to CHW
            
            images.append(image)
        
        return images
    
    def _load_image_chunk(self, chunk_paths: List) -> List[torch.Tensor]:
        """Load a chunk of image files."""
        images = []
        
        for path_info in chunk_paths:
            # Image file - handle None case
            if path_info is None:
                raise ValueError(f"Invalid path_info: {path_info}")
            
            # Check if file exists
            import os
            if not os.path.exists(path_info):
                raise ValueError(f"Image file not found: {path_info}")
            
            # Use PIL for image loading
            from PIL import Image
            import torchvision.transforms as transforms
            
            pil_image = Image.open(path_info).convert('RGB')
            
            # Apply undistortion if maps are provided
            if self.undistortion_maps is not None:
                # Convert PIL to numpy for undistortion
                image_np = np.array(pil_image)
                undistorted_np = self.undistortion_maps.undistort_image(image_np, self.target_size)
                # Convert back to tensor
                image = torch.from_numpy(undistorted_np).float() / 255.0
                image = image.permute(2, 0, 1)  # HWC to CHW
            else:
                # Use PIL transforms for regular loading
                transform = transforms.Compose([
                    transforms.Resize(self.target_size),  # (height, width)
                    transforms.ToTensor()
                ])
                image = transform(pil_image)
            
            images.append(image)
        
        return images
    
    def __del__(self):
        """Cleanup video loader if it exists."""
        if hasattr(self, 'video_loader') and self.video_loader is not None:
            try:
                # Close any open video decoders
                if hasattr(self.video_loader, 'decoders'):
                    for video_path in list(self.video_loader.decoders.keys()):
                        self.video_loader.close_decoder(video_path)
            except:
                pass  # Ignore errors during cleanup


class AsyncImageDataset(Dataset):
    """
    Custom dataset for asynchronous image loading with optional undistortion support.
    """
    
    def __init__(self, image_paths: List, target_size: Tuple[int, int], device: str = 'cpu', 
                 undistortion_maps=None):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of image paths or (video_path, frame_idx) tuples
            target_size: Target size (height, width) for resizing
            device: Device to load images on
            undistortion_maps: Optional UndistortionMaps object for applying undistortion
        """
        self.image_paths = image_paths
        self.target_size = target_size
        self.device = device
        self.undistortion_maps = undistortion_maps
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Load and preprocess a single image."""
        path_info = self.image_paths[idx]
        
        if isinstance(path_info, tuple):
            # Video frame
            video_path, frame_idx = path_info
            from pi3.utils.basic import load_video_frame
            
            # Load frame using torchcodec (with fallback to OpenCV)
            image = load_video_frame(video_path, frame_idx, self.target_size, use_torchcodec=True)
            
            # Apply undistortion if maps are provided
            if self.undistortion_maps is not None:
                # Convert tensor to numpy for undistortion
                image_np = image.permute(1, 2, 0).numpy()  # CHW to HWC
                undistorted_np = self.undistortion_maps.undistort_image(image_np, self.target_size)
                # Convert back to tensor
                image = torch.from_numpy(undistorted_np).float() / 255.0
                image = image.permute(2, 0, 1)  # HWC to CHW
        else:
            # Image file - handle None case
            if path_info is None:
                raise ValueError(f"Invalid path_info at index {idx}: {path_info}")
            
            # Check if file exists
            import os
            if not os.path.exists(path_info):
                raise ValueError(f"Image file not found: {path_info}")
            
            # Use PIL for image loading
            from PIL import Image
            import torchvision.transforms as transforms
            
            pil_image = Image.open(path_info).convert('RGB')
            
            # Apply undistortion if maps are provided
            if self.undistortion_maps is not None:
                # Convert PIL to numpy for undistortion
                image_np = np.array(pil_image)
                undistorted_np = self.undistortion_maps.undistort_image(image_np, self.target_size)
                # Convert back to tensor
                image = torch.from_numpy(undistorted_np).float() / 255.0
                image = image.permute(2, 0, 1)  # HWC to CHW
            else:
                # Use PIL transforms for regular loading
                transform = transforms.Compose([
                    transforms.Resize(self.target_size),  # (height, width)
                    transforms.ToTensor()
                ])
                image = transform(pil_image)
        
        return image.to(self.device)


def visualization_process(queue: mp.Queue, rerun_port: int, max_points: int = 50000, update_interval: float = 0.1):
    """
    Separate process for handling Rerun visualization.
    
    Args:
        queue: Multiprocessing queue for receiving visualization data
        rerun_port: Port for Rerun server
        max_points: Maximum number of points to visualize
        update_interval: Minimum time between updates
    """
    try:
        # Initialize Rerun in this process
        rr.init("Pi3SLAM Online - Visualization Process")
        rr.spawn(port=rerun_port, connect=True)
        
        # Set up the coordinate system
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        
        # Log initial camera trajectory
        rr.log("world/camera_trajectory", rr.Points3D(
            positions=np.zeros((1, 3)), 
            colors=np.array([[1, 0, 0]])
        ), static=True)
        
        print(f"🎥 Visualization process started on port {rerun_port}")
        
        last_update_time = 0
        
        while True:
            try:
                # Get data from queue with timeout
                data = queue.get(timeout=1.0)
                
                if data is None:  # Shutdown signal
                    break
                
                current_time = time.time()
                if current_time - last_update_time < update_interval:
                    continue
                
                last_update_time = current_time
                
                # Extract visualization data
                all_points = data.get('points', np.array([]))
                all_colors = data.get('colors', np.array([]))
                camera_positions = data.get('camera_positions', np.array([]))
                camera_orientations = data.get('camera_orientations', np.array([]))
                chunk_end_poses = data.get('chunk_end_poses', np.array([]))
                current_frame = data.get('current_frame', None)
                chunk_start_frame = data.get('chunk_start_frame', None)
                chunk_end_frame = data.get('chunk_end_frame', None)
                frame_info = data.get('frame_info', '')
                stats_text = data.get('stats_text', '')
                
                # Subsample points for visualization
                if len(all_points) > max_points:
                    indices = np.random.choice(len(all_points), max_points, replace=False)
                    all_points = all_points[indices]
                    all_colors = all_colors[indices]
                
                # Log point cloud
                if len(all_points) > 0:
                    rr.log("world/point_cloud", rr.Points3D(
                        positions=all_points,
                        colors=all_colors
                    ), static=True)
                
                # Log camera trajectory and poses
                if len(camera_positions) > 0:
                    # Camera positions as points
                    rr.log("world/camera_poses", rr.Points3D(
                        positions=camera_positions,
                        colors=np.full((len(camera_positions), 3), [1.0, 0.0, 0.0])
                    ), static=True)
                    
                    # Camera trajectory as line
                    if len(camera_positions) > 1:
                        rr.log("world/camera_trajectory", rr.LineStrips3D(
                            strips=[camera_positions],
                            colors=np.array([1.0, 0.0, 0.0])
                        ), static=True)
                
                # Log chunk end poses as linestrips
                if len(chunk_end_poses) > 1:
                    rr.log("world/chunk_end_trajectory", rr.LineStrips3D(
                        strips=[chunk_end_poses],
                        colors=np.array([0.0, 1.0, 0.0])  # Green color for chunk end trajectory
                    ), static=True)
                    
                    # Also log chunk end poses as points
                    rr.log("world/chunk_end_poses", rr.Points3D(
                        positions=chunk_end_poses,
                        colors=np.full((len(chunk_end_poses), 3), [0.0, 1.0, 0.0])
                    ), static=True)
                    
                                    # Camera coordinate frames (only recent ones)
                num_cameras = len(camera_positions)
                recent_cameras = min(5, num_cameras)
                
                for i in range(num_cameras):
                    if i < num_cameras - recent_cameras:
                        # Clear old coordinate frames
                        rr.log(f"world/camera_{i}/x_axis", rr.LineStrips3D(strips=[]), static=True)
                        rr.log(f"world/camera_{i}/y_axis", rr.LineStrips3D(strips=[]), static=True)
                        rr.log(f"world/camera_{i}/z_axis", rr.LineStrips3D(strips=[]), static=True)
                        rr.log(f"world/camera_{i}/frustum", rr.LineStrips3D(strips=[]), static=True)
                    else:
                        # Show recent coordinate frames
                        camera_pos = camera_positions[i]
                        camera_rot = camera_orientations[i]
                        
                        # Create coordinate frame vectors
                        scale = 0.15
                        origin = camera_pos
                        x_axis = origin + scale * camera_rot[:, 0]
                        y_axis = origin + scale * camera_rot[:, 1]
                        z_axis = origin + scale * camera_rot[:, 2]
                        
                        # Log coordinate frame axes
                        rr.log(f"world/camera_{i}/x_axis", rr.LineStrips3D(
                            strips=[[origin, x_axis]],
                            colors=np.array([1.0, 0.0, 0.0])
                        ), static=True)
                        
                        rr.log(f"world/camera_{i}/y_axis", rr.LineStrips3D(
                            strips=[[origin, y_axis]],
                            colors=np.array([0.0, 1.0, 0.0])
                        ), static=True)
                        
                        rr.log(f"world/camera_{i}/z_axis", rr.LineStrips3D(
                            strips=[[origin, z_axis]],
                            colors=np.array([0.0, 0.0, 1.0])
                        ), static=True)
                        
                        # Add camera frustum for the most recent camera
                        if i == num_cameras - 1:
                            frustum_scale = 0.2
                            near_plane = origin + frustum_scale * camera_rot[:, 2]
                            
                            corner1 = near_plane + frustum_scale * 0.5 * (camera_rot[:, 0] + camera_rot[:, 1])
                            corner2 = near_plane + frustum_scale * 0.5 * (camera_rot[:, 0] - camera_rot[:, 1])
                            corner3 = near_plane + frustum_scale * 0.5 * (-camera_rot[:, 0] - camera_rot[:, 1])
                            corner4 = near_plane + frustum_scale * 0.5 * (-camera_rot[:, 0] + camera_rot[:, 1])
                            
                            rr.log(f"world/camera_{i}/frustum", rr.LineStrips3D(
                                strips=[[origin, corner1], [origin, corner2], [origin, corner3], [origin, corner4],
                                       [corner1, corner2], [corner2, corner3], [corner3, corner4], [corner4, corner1]],
                                colors=np.array([1.0, 1.0, 0.0])
                            ), static=True)
                
                # Always show frustum for the most recent camera (even if it's the only one)
                if num_cameras > 0:
                    latest_camera_pos = camera_positions[-1]
                    latest_camera_rot = camera_orientations[-1]
                    
                    frustum_scale = 0.2
                    origin = latest_camera_pos
                    near_plane = origin + frustum_scale * latest_camera_rot[:, 2]
                    
                    corner1 = near_plane + frustum_scale * 0.5 * (latest_camera_rot[:, 0] + latest_camera_rot[:, 1])
                    corner2 = near_plane + frustum_scale * 0.5 * (latest_camera_rot[:, 0] - latest_camera_rot[:, 1])
                    corner3 = near_plane + frustum_scale * 0.5 * (-latest_camera_rot[:, 0] - latest_camera_rot[:, 1])
                    corner4 = near_plane + frustum_scale * 0.5 * (-latest_camera_rot[:, 0] + latest_camera_rot[:, 1])
                    
                    rr.log("world/latest_camera_frustum", rr.LineStrips3D(
                        strips=[[origin, corner1], [origin, corner2], [origin, corner3], [origin, corner4],
                               [corner1, corner2], [corner2, corner3], [corner3, corner4], [corner4, corner1]],
                        colors=np.array([1.0, 1.0, 0.0])
                    ), static=False)
                
                # Log current video frame
                if current_frame is not None:
                    try:
                        print(f"📷 Logging current frame: shape {current_frame.shape}, dtype {current_frame.dtype}")
                        
                        # Ensure the frame is in the correct format for Rerun
                        if len(current_frame.shape) == 3 and current_frame.shape[2] == 3:  # HWC format
                            rr.log("world/current_frame", rr.Image(current_frame), static=False)
                        else:
                            print(f"⚠️  Skipping current frame: invalid shape {current_frame.shape}")
                    except Exception as e:
                        print(f"⚠️  Error logging current frame: {e}")
                
                # Log current chunk start and end frames
                if chunk_start_frame is not None:
                    try:
                        print(f"📷 Logging chunk start frame: shape {chunk_start_frame.shape}, dtype {chunk_start_frame.dtype}")
                        
                        # Ensure the frame is in the correct format for Rerun
                        if len(chunk_start_frame.shape) == 3 and chunk_start_frame.shape[2] == 3:  # HWC format
                            rr.log("world/chunk_start_frame", rr.Image(chunk_start_frame), static=False)
                        else:
                            print(f"⚠️  Skipping chunk start frame: invalid shape {chunk_start_frame.shape}")
                    except Exception as e:
                        print(f"⚠️  Error logging chunk start frame: {e}")
                
                if chunk_end_frame is not None:
                    try:
                        print(f"📷 Logging chunk end frame: shape {chunk_end_frame.shape}, dtype {chunk_end_frame.dtype}")
                        
                        # Ensure the frame is in the correct format for Rerun
                        if len(chunk_end_frame.shape) == 3 and chunk_end_frame.shape[2] == 3:  # HWC format
                            rr.log("world/chunk_end_frame", rr.Image(chunk_end_frame), static=False)
                        else:
                            print(f"⚠️  Skipping chunk end frame: invalid shape {chunk_end_frame.shape}")
                    except Exception as e:
                        print(f"⚠️  Error logging chunk end frame: {e}")
                
                # Log frame info and statistics
                if frame_info:
                    rr.log("world/frame_info", rr.TextDocument(frame_info), static=False)
                
                if stats_text:
                    rr.log("world/stats", rr.TextDocument(stats_text), static=True)
                
            except mp.queues.Empty:
                continue  # No data available, continue waiting
            except Exception as e:
                print(f"⚠️  Visualization process error: {e}")
                continue
        
        # Cleanup
        rr.disconnect()
        print("🎥 Visualization process stopped")
        
    except Exception as e:
        print(f"❌ Failed to start visualization process: {e}")


class SIM3Transformation:
    """SIM3 transformation class for aligning trajectories."""
    
    def __init__(self, s: float = 1.0, R: np.ndarray = None, t: np.ndarray = None):
        """
        Initialize SIM3 transformation.
        
        Args:
            s: Scale factor
            R: 3x3 rotation matrix
            t: 3x1 translation vector
        """
        self.s = s
        self.R = R if R is not None else np.eye(3)
        self.t = t if t is not None else np.zeros(3)
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform points using SIM3 transformation."""
        return self.s * (self.R @ points.T).T + self.t
    
    def compose(self, other: 'SIM3Transformation') -> 'SIM3Transformation':
        """Compose this transformation with another."""
        new_s = self.s * other.s
        new_R = self.R @ other.R
        new_t = self.s * (self.R @ other.t) + self.t
        return SIM3Transformation(new_s, new_R, new_t)
    
    def inverse(self) -> 'SIM3Transformation':
        """Compute inverse transformation."""
        inv_s = 1.0 / self.s
        inv_R = self.R.T
        inv_t = -inv_s * (inv_R @ self.t)
        return SIM3Transformation(inv_s, inv_R, inv_t)
    
    def get_matrix(self) -> np.ndarray:
        """Get 4x4 homogeneous transformation matrix."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.s * self.R  # Scale * Rotation
        matrix[:3, 3] = self.t            # Translation
        return matrix


def estimate_sim3_transformation_robust(points1: np.ndarray, points2: np.ndarray, 
                                      max_iterations: int = 100, 
                                      inlier_threshold: float = 0.1,
                                      min_inlier_ratio: float = 0.3) -> SIM3Transformation:
    """
    Estimate SIM3 transformation between two sets of corresponding points using RANSAC.
    
    Args:
        points1: First set of points (N, 3)
        points2: Second set of points (N, 3)
        max_iterations: Maximum RANSAC iterations
        inlier_threshold: Distance threshold for inlier classification
        min_inlier_ratio: Minimum ratio of inliers to consider transformation valid
    
    Returns:
        SIM3Transformation object
    """
    if len(points1) < 3 or len(points2) < 3:
        raise ValueError(f"Need at least 3 corresponding points, got {len(points1)} and {len(points2)}")
    
    # Ensure both point sets have the same number of points
    if len(points1) != len(points2):
        min_len = min(len(points1), len(points2))
        points1 = points1[:min_len]
        points2 = points2[:min_len]
    
    best_transform = None
    best_inliers = []
    best_inlier_count = 0
    
    # RANSAC iterations
    for iteration in range(max_iterations):
        # Randomly sample 3 points (minimum for SIM3 estimation)
        if len(points1) == 3:
            sample_indices = np.arange(3)
        else:
            sample_indices = np.random.choice(len(points1), 3, replace=False)
        
        sample_points1 = points1[sample_indices]
        sample_points2 = points2[sample_indices]
        
        try:
            # Estimate transformation from sample
            transform = estimate_sim3_transformation(sample_points1, sample_points2)
            
            # Apply transformation to all points
            transformed_points2 = transform.transform_points(points2)
            
            # Calculate distances
            distances = np.linalg.norm(points1 - transformed_points2, axis=1)
            
            # Find inliers
            inliers = distances < inlier_threshold
            inlier_count = np.sum(inliers)
            
            # Update best if we have more inliers
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
                best_transform = transform
                
        except Exception as e:
            # Skip this iteration if estimation fails
            continue
    
    # If we found a good transformation, refine it using all inliers
    if best_transform is not None and best_inlier_count >= min_inlier_ratio * len(points1):
        inlier_points1 = points1[best_inliers]
        inlier_points2 = points2[best_inliers]
        
        if len(inlier_points1) >= 3:
            try:
                # Refine transformation using all inliers
                refined_transform = estimate_sim3_transformation(inlier_points1, inlier_points2)
                return refined_transform
            except:
                pass
    
    # Fallback to original transformation if RANSAC fails
    if best_transform is not None:
        return best_transform
    
    # Final fallback to all points
    return estimate_sim3_transformation(points1, points2)


def detect_outliers_using_mahalanobis(points1: np.ndarray, points2: np.ndarray, 
                                    transform: SIM3Transformation, 
                                    threshold: float = 3.0) -> np.ndarray:
    """
    Detect outliers using Mahalanobis distance.
    
    Args:
        points1: First set of points (N, 3)
        points2: Second set of points (N, 3)
        transform: SIM3 transformation
        threshold: Mahalanobis distance threshold for outlier detection
    
    Returns:
        Boolean array indicating inliers (True) and outliers (False)
    """
    # Apply transformation
    transformed_points2 = transform.transform_points(points2)
    
    # Calculate residuals
    residuals = points1 - transformed_points2
    
    # Calculate covariance matrix of residuals
    cov_matrix = np.cov(residuals.T)
    
    # Add small regularization to avoid singular matrix
    cov_matrix += np.eye(3) * 1e-6
    
    try:
        # Calculate Mahalanobis distances
        inv_cov = np.linalg.inv(cov_matrix)
        mahal_distances = np.sqrt(np.sum(residuals @ inv_cov * residuals, axis=1))
        
        # Return inlier mask
        return mahal_distances < threshold
    except:
        # Fallback to simple distance-based outlier detection
        distances = np.linalg.norm(residuals, axis=1)
        median_distance = np.median(distances)
        mad = np.median(np.abs(distances - median_distance))
        threshold_distance = median_distance + 2.5 * mad
        return distances < threshold_distance


def estimate_sim3_transformation_robust_irls(points1: np.ndarray, points2: np.ndarray,
                                           max_iterations: int = 20,
                                           convergence_threshold: float = 1e-6) -> SIM3Transformation:
    """
    Estimate SIM3 transformation using Iteratively Reweighted Least Squares (IRLS).
    
    Args:
        points1: First set of points (N, 3)
        points2: Second set of points (N, 3)
        max_iterations: Maximum IRLS iterations
        convergence_threshold: Convergence threshold
    
    Returns:
        SIM3Transformation object
    """
    if len(points1) < 3 or len(points2) < 3:
        raise ValueError(f"Need at least 3 corresponding points, got {len(points1)} and {len(points2)}")
    
    # Ensure both point sets have the same number of points
    if len(points1) != len(points2):
        min_len = min(len(points1), len(points2))
        points1 = points1[:min_len]
        points2 = points2[:min_len]
    
    # Initial weights (all equal)
    weights = np.ones(len(points1))
    
    # Initial transformation
    transform = estimate_sim3_transformation(points1, points2)
    
    for iteration in range(max_iterations):
        # Apply current transformation
        transformed_points2 = transform.transform_points(points2)
        
        # Calculate residuals
        residuals = points1 - transformed_points2
        distances = np.linalg.norm(residuals, axis=1)
        
        # Update weights using Huber loss
        sigma = np.median(distances) * 1.4826  # Robust scale estimate
        delta = 1.345 * sigma  # Huber threshold
        
        # Huber weights
        new_weights = np.where(distances <= delta, 
                              np.ones_like(distances), 
                              delta / distances)
        
        # Check convergence
        weight_change = np.mean(np.abs(new_weights - weights))
        if weight_change < convergence_threshold:
            break
        
        weights = new_weights
        
        # Re-estimate transformation with weighted points
        # We'll use a simple approach: repeat points based on weights
        # This is a simplified version - in practice, you'd use weighted least squares
        
        # Create weighted point sets
        total_weight = np.sum(weights)
        if total_weight > 0:
            # Normalize weights
            normalized_weights = weights / total_weight
            
            # Use weighted centroids
            weighted_centroid1 = np.average(points1, axis=0, weights=normalized_weights)
            weighted_centroid2 = np.average(points2, axis=0, weights=normalized_weights)
            
            # Center points
            centered_points1 = points1 - weighted_centroid1
            centered_points2 = points2 - weighted_centroid2
            
            # Weighted scale estimation
            scale1 = np.sqrt(np.sum(centered_points1 ** 2, axis=1))
            scale2 = np.sqrt(np.sum(centered_points2 ** 2, axis=1))
            
            # Weighted median scale ratio
            scale_ratios = scale2 / (scale1 + 1e-8)
            s = np.median(scale_ratios)
            
            # Weighted rotation estimation
            H = centered_points1.T @ (centered_points2 * weights[:, np.newaxis])
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Ensure proper rotation matrix
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Compute translation
            t = weighted_centroid2 - s * (R @ weighted_centroid1)
            
            transform = SIM3Transformation(s, R, t)
    
    return transform


def estimate_sim3_transformation(points1: np.ndarray, points2: np.ndarray) -> SIM3Transformation:
    """
    Estimate SIM3 transformation between two sets of corresponding points.
    
    Args:
        points1: First set of points (N, 3)
        points2: Second set of points (N, 3)
    
    Returns:
        SIM3Transformation object
    """
    if len(points1) < 3 or len(points2) < 3:
        raise ValueError(f"Need at least 3 corresponding points, got {len(points1)} and {len(points2)}")
    
    # Ensure both point sets have the same number of points
    if len(points1) != len(points2):
        min_len = min(len(points1), len(points2))
        points1 = points1[:min_len]
        points2 = points2[:min_len]
    
    # Center the points
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)
    
    centered_points1 = points1 - centroid1
    centered_points2 = points2 - centroid2
    
    # Compute scale
    scale1 = np.sqrt(np.sum(centered_points1 ** 2, axis=1))
    scale2 = np.sqrt(np.sum(centered_points2 ** 2, axis=1))
    
    # Use median scale ratio for robustness
    scale_ratios = scale2 / (scale1 + 1e-8)
    s = np.median(scale_ratios)
    
    # Estimate rotation using SVD
    H = centered_points1.T @ centered_points2
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation matrix (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = centroid2 - s * (R @ centroid1)
    
    return SIM3Transformation(s, R, t)


def find_corresponding_points(points1: torch.Tensor, points2: torch.Tensor, 
                            camera_ids1: List[int], camera_ids2: List[int],
                            conf1: torch.Tensor = None, conf2: torch.Tensor = None,
                            subsample_factor: int = 4,  # Increased default subsampling for speed
                            conf_threshold: float = 0.5, 
                            threshold: float = 0.1,
                            use_robust_filtering: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find corresponding points between two chunks based on camera IDs, using confidence filtering.
    Optimized for speed with aggressive subsampling and vectorized operations.
    
    Args:
        points1: Points from first chunk (N, 3)
        points2: Points from second chunk (M, 3)
        camera_ids1: Camera IDs for first chunk
        camera_ids2: Camera IDs for second chunk
        conf1: Confidence values for first chunk (N, 1) or (N, H, W, 1)
        conf2: Confidence values for second chunk (M, 1) or (M, H, W, 1)
        subsample_factor: Factor to subsample points (take every nth point, default: 4)
        conf_threshold: Confidence threshold for filtering points (default: 0.5)
        threshold: Distance threshold for correspondence
        use_robust_filtering: Whether to use robust outlier filtering
    
    Returns:
        Tuple of corresponding points (N_corr, 3), (N_corr, 3)
    """
    # Find common camera IDs
    common_ids = set(camera_ids1) & set(camera_ids2)
    
    if len(common_ids) == 0:
        return np.array([]), np.array([])
    
    # Pre-allocate lists for better performance
    all_corr_points1 = []
    all_corr_points2 = []
    
    for cam_id in common_ids:
        idx1 = camera_ids1.index(cam_id)
        idx2 = camera_ids2.index(cam_id)
        
        # Get points for this camera with aggressive subsampling
        pts1 = points1[idx1].view(-1, 3)[::subsample_factor].cpu().numpy()
        pts2 = points2[idx2].view(-1, 3)[::subsample_factor].cpu().numpy()
        
        # Vectorized valid point detection (much faster than loop)
        valid1 = np.linalg.norm(pts1, axis=1) > 1e-6
        valid2 = np.linalg.norm(pts2, axis=1) > 1e-6
        
        # Apply confidence filtering if provided
        if conf1 is not None:
            cam_conf1 = conf1[idx1].view(-1, 1)[::subsample_factor].cpu().numpy()
            # Vectorized sigmoid and threshold
            cam_conf1 = 1 / (1 + np.exp(-cam_conf1.flatten()))
            high_conf1 = cam_conf1 > conf_threshold
            valid1 = np.logical_and(valid1, high_conf1)
        
        if conf2 is not None:
            cam_conf2 = conf2[idx2].view(-1, 1)[::subsample_factor].cpu().numpy()
            # Vectorized sigmoid and threshold
            cam_conf2 = 1 / (1 + np.exp(-cam_conf2.flatten()))
            high_conf2 = cam_conf2 > conf_threshold
            valid2 = np.logical_and(valid2, high_conf2)
        
        # Combine valid points efficiently
        valid_pts = np.logical_and(valid1, valid2)
        
        if np.sum(valid_pts) > 0:
            # Use boolean indexing for efficient selection
            points1_selected = pts1[valid_pts]
            points2_selected = pts2[valid_pts]
            
            # Extend lists efficiently
            all_corr_points1.extend(points1_selected)
            all_corr_points2.extend(points2_selected)
    
    if not all_corr_points1:
        return np.array([]), np.array([])
    
    # Convert to numpy arrays efficiently
    all_points1 = np.array(all_corr_points1)
    all_points2 = np.array(all_corr_points2)
    
    # Ensure equal number of points
    min_total = min(len(all_points1), len(all_points2))
    if min_total > 0:
        all_points1 = all_points1[:min_total]
        all_points2 = all_points2[:min_total]
        
        # Apply robust outlier filtering if enabled and we have enough points
        if use_robust_filtering and len(all_points1) >= 10:
            try:
                # Initial transformation for outlier detection
                initial_transform = estimate_sim3_transformation(all_points1, all_points2)
                
                # Detect outliers using Mahalanobis distance
                inlier_mask = detect_outliers_using_mahalanobis(all_points1, all_points2, initial_transform)
                
                # Keep only inliers
                if np.sum(inlier_mask) >= 5:  # Need at least 5 points for robust estimation
                    all_points1 = all_points1[inlier_mask]
                    all_points2 = all_points2[inlier_mask]
                    print(f"  🔍 Robust filtering: {len(inlier_mask)} -> {len(all_points1)} points ({np.sum(inlier_mask)/len(inlier_mask)*100:.1f}% inliers)")
                
            except Exception as e:
                print(f"  ⚠️  Robust filtering failed: {e}, using all points")
        
        return all_points1, all_points2
    else:
        return np.array([]), np.array([])


class Pi3SLAMOnlineRerun:
    """
    Pi3SLAM with online processing and Rerun visualization.
    Processes long sequences in chunks with real-time visualization.
    """
    
    def __init__(self, model: Pi3, chunk_length: int = 100, overlap: int = 10, 
                 device: str = 'cuda', conf_threshold: float = 0.5, 
                 undistortion_maps=None, cam_scale: float = 1.0,
                 max_chunks_in_memory: int = 5, enable_disk_cache: bool = False,
                 cache_dir: str = None, rerun_port: int = 9090):
        """
        Initialize Pi3SLAM Online with Rerun visualization.
        
        Args:
            model: Pre-trained Pi3 model instance
            chunk_length: Number of frames per chunk
            overlap: Number of overlapping frames between chunks
            device: Device to run inference on
            conf_threshold: Confidence threshold for filtering points
            undistortion_maps: Optional UndistortionMaps object for applying undistortion
            cam_scale: Scale factor for camera poses
            max_chunks_in_memory: Maximum number of chunks to keep in memory
            enable_disk_cache: Whether to enable disk caching for processed chunks
            cache_dir: Directory for disk cache (if None, uses temporary directory)
        """
        self.model = model
        self.chunk_length = chunk_length
        self.overlap = overlap
        self.device = device
        self.conf_threshold = conf_threshold
        self.undistortion_maps = undistortion_maps
        self.cam_scale = cam_scale
        self.max_chunks_in_memory = max_chunks_in_memory
        self.enable_disk_cache = enable_disk_cache
        self.cache_dir = cache_dir
        self.pixel_limit = 255000/2
        self.memory_monitoring = True
        self.rerun_port = rerun_port
        
        # Initialize data structures
        self.chunk_results = []
        self.camera_ids = []
        self.aligned_points = []
        self.aligned_camera_poses = []
        self.aligned_camera_ids = []
        self.aligned_colors = []
        
        # Camera trajectory tracking
        self.full_camera_trajectory = []
        self.full_camera_orientations = []
        self.chunk_end_poses = []
        
        # Timestamp tracking
        self.timestamps = []
        
        # Statistics
        self.stats = {
            'total_chunks': 0,
            'total_frames': 0,
            'processing_times': [],
            'alignment_times': [],
            'load_times': [],
            'inference_times': [],
            'postprocess_times': [],
            'cpu_transfer_times': []
        }
        
        # Configuration for correspondence search
        self.correspondence_subsample_factor = 4
        self.use_robust_alignment = False
        self.robust_alignment_method = 'auto'
        
        # Visualization
        self.vis_running = False
        self.vis_process = None
        self.vis_queue = None
        self.current_chunk_result = None
        
        # Background loading
        self.loader_running = False
        self.loader_process = None
        self.loader_queue = None
        self.image_cache = {}
        self.target_size = None
        
        # Disk cache
        self.disk_cache = {}
        if self.enable_disk_cache:
            self._setup_disk_cache()
        
        print(f"🚀 Pi3SLAM Online initialized with chunk_length={chunk_length}, overlap={overlap}")
        print(f"   Device: {device}, Confidence threshold: {conf_threshold}")
        print(f"   Camera scale: {cam_scale}")
        if undistortion_maps:
            print(f"   Undistortion: Enabled")
        if enable_disk_cache:
            print(f"   Disk cache: Enabled ({cache_dir or 'Temporary'})")
        print(f"   Max chunks in memory: {max_chunks_in_memory}")
    
    def _setup_disk_cache(self):
        """Set up disk cache if enabled."""
        if self.cache_dir is None:
            import tempfile
            self.cache_dir = tempfile.mkdtemp(prefix="pi3slam_cache_")
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"💾 Disk caching enabled: {self.cache_dir}")
    
    def _save_chunk_to_disk(self, chunk_idx: int, points: np.ndarray, colors: np.ndarray, poses: np.ndarray):
        """Save chunk data to disk to free memory."""
        if self.cache_dir is None:
            return
        
        try:
            chunk_file = os.path.join(self.cache_dir, f"chunk_{chunk_idx:06d}.npz")
            np.savez_compressed(
                chunk_file,
                points=points,
                colors=colors,
                poses=poses
            )
        except Exception as e:
            print(f"⚠️  Failed to save chunk {chunk_idx} to disk: {e}")
    
    def _load_chunk_from_disk(self, chunk_idx: int) -> Dict:
        """Load chunk data from disk."""
        if self.cache_dir is None:
            return None
        
        try:
            chunk_file = os.path.join(self.cache_dir, f"chunk_{chunk_idx:06d}.npz")
            if os.path.exists(chunk_file):
                data = np.load(chunk_file)
                return {
                    'points': data['points'],
                    'colors': data['colors'],
                    'poses': data['poses']
                }
        except Exception as e:
            print(f"⚠️  Failed to load chunk {chunk_idx} from disk: {e}")
        
        return None
    
    def _manage_memory(self):
        """Manage memory by moving old chunks to disk and keeping only recent ones in memory."""
        if len(self.aligned_points) <= self.max_chunks_in_memory:
            return
        
        # Move oldest chunks to disk
        chunks_to_move = len(self.aligned_points) - self.max_chunks_in_memory
        
        for i in range(chunks_to_move):
            chunk_idx = i
            if chunk_idx < len(self.aligned_points):
                # Save to disk
                self._save_chunk_to_disk(
                    chunk_idx,
                    self.aligned_points[chunk_idx],
                    self.aligned_colors[chunk_idx],
                    self.aligned_camera_poses[chunk_idx]
                )
                
                # Clear from memory (but keep camera trajectory data)
                self.aligned_points[chunk_idx] = None
                self.aligned_colors[chunk_idx] = None
                self.aligned_camera_poses[chunk_idx] = None
        
        # Compact lists by removing None entries
        self.aligned_points = [p for p in self.aligned_points if p is not None]
        self.aligned_colors = [c for c in self.aligned_colors if c is not None]
        self.aligned_camera_poses = [p for p in self.aligned_camera_poses if p is not None]
        
        print(f"💾 Moved {chunks_to_move} chunks to disk, keeping {len(self.aligned_points)} in memory")
        print(f"📷 Camera trajectory: {len(self.full_camera_trajectory)} poses, {len(self.chunk_end_poses)} chunk end poses")
        
        # Log memory usage if monitoring is enabled
        if self.memory_monitoring:
            self._log_memory_usage()
    
    def _get_visualization_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get subsampled points for visualization."""
        if not self.aligned_points:
            return np.array([]), np.array([])
        
        # Collect all available points
        all_points = []
        all_colors = []
        
        for i, points in enumerate(self.aligned_points):
            if points is not None:
                all_points.append(points)
                all_colors.append(self.aligned_colors[i])
            else:
                # Load from disk if needed
                chunk_data = self._load_chunk_from_disk(i)
                if chunk_data is not None:
                    all_points.append(chunk_data['points'])
                    all_colors.append(chunk_data['colors'])
        
        if not all_points:
            return np.array([]), np.array([])
        
        # Combine all points
        if len(all_points) == 1:
            combined_points = all_points[0]
            combined_colors = all_colors[0]
        else:
            combined_points = np.concatenate(all_points, axis=0)
            combined_colors = np.concatenate(all_colors, axis=0)
        
        # Subsample for visualization (only 10% of points)
        if len(combined_points) > 0:
            subsample_size = max(1000, int(len(combined_points) * self.visualization_subsample_ratio))
            if len(combined_points) > subsample_size:
                indices = np.random.choice(len(combined_points), subsample_size, replace=False)
                combined_points = combined_points[indices]
                combined_colors = combined_colors[indices]
        
        return combined_points, combined_colors
    
    def _log_memory_usage(self):
        """Log current memory usage for monitoring."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
            
            # Count total points in memory
            total_points_in_memory = 0
            for points in self.aligned_points:
                if points is not None:
                    total_points_in_memory += len(points)
            
            print(f"🧠 Memory usage: {memory_mb:.1f} MB, Points in memory: {total_points_in_memory:,}, Camera poses: {len(self.full_camera_trajectory):,}, Chunk end poses: {len(self.chunk_end_poses)}")
            
        except ImportError:
            # psutil not available, skip memory logging
            pass
        except Exception as e:
            print(f"⚠️  Failed to log memory usage: {e}")
    
    def start_background_loader(self, all_image_paths: List):
        """Start background image loading for the entire sequence using chunk-based loading."""
        if self.loader_running:
            return
        
        try:
            # Store the full image paths list
            self.image_paths = all_image_paths
            
            # Calculate target size from first image if not already set
            if self.target_size is None:
                self.target_size = self._calculate_target_size(all_image_paths[0])
            
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
    
    def stop_background_loader(self):
        """Stop the background image loader."""
        if self.loader_running and self.background_loader is not None:
            try:
                self.loader_running = False
                if hasattr(self.background_loader, 'shutdown'):
                    self.background_loader.shutdown()
                
                # Clean up video loader if it exists
                if hasattr(self.background_loader, 'dataset') and hasattr(self.background_loader.dataset, 'video_loader'):
                    if self.background_loader.dataset.video_loader is not None:
                        try:
                            # Close any open video decoders
                            if hasattr(self.background_loader.dataset.video_loader, 'decoders'):
                                for video_path in list(self.background_loader.dataset.video_loader.decoders.keys()):
                                    self.background_loader.dataset.video_loader.close_decoder(video_path)
                        except:
                            pass  # Ignore errors during cleanup
                
                print("🔄 Background image loader stopped")
            except Exception as e:
                print(f"⚠️  Error stopping background loader: {e}")
    
    def get_chunk_images(self, start_idx: int, end_idx: int) -> torch.Tensor:
        """
        Get images for a specific chunk range using background loader.
        
        Args:
            start_idx: Start index of the chunk
            end_idx: End index of the chunk (exclusive)
        
        Returns:
            Tensor of images (N, 3, H, W)
        """
        if not self.loader_running or self.background_loader is None or self.image_paths is None:
            # Fallback to synchronous loading
            return self._load_images_from_paths(self.image_paths[start_idx:end_idx] if self.image_paths else [])
        
        try:
            # Find the chunk index that contains this range
            chunk_idx = None
            for i, (chunk_start, chunk_end) in enumerate(self.background_loader.dataset.chunk_indices):
                if chunk_start == start_idx and chunk_end == end_idx:
                    chunk_idx = i
                    break
            
            if chunk_idx is None:
                # Chunk not found, fallback to synchronous loading
                return self._load_images_from_paths(self.image_paths[start_idx:end_idx])
            
            # Get the chunk from the dataset
            chunk_data = self.background_loader.dataset[chunk_idx]
            return chunk_data['chunk']
            
        except Exception as e:
            print(f"⚠️  Background loading failed, falling back to synchronous: {e}")
            return self._load_images_from_paths(self.image_paths[start_idx:end_idx])
    
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
                args=(self.vis_queue, self.rerun_port, self.max_points_visualization, self.update_interval)
            )
            self.vis_process.start()
            self.vis_running = True
            
            print(f"🎥 Rerun visualization process started on port {self.rerun_port}")
            print(f"🌐 Open http://localhost:{self.rerun_port} in your browser to view the visualization")
            
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
    
    def _update_visualization(self):
        """Send visualization data to the separate process."""
        if not self.vis_running:
            return
        
        # Throttle visualization updates
        current_time = time.time()
        if hasattr(self, '_last_viz_update') and current_time - self._last_viz_update < self.update_interval:
            return
        self._last_viz_update = current_time
        
        try:
            # Get subsampled points for visualization (only 10% of points)
            all_points, all_colors = self._get_visualization_points()
            
            if len(all_points) == 0:
                return
            
            # Use full camera trajectory (always kept in memory)
            camera_positions = np.array(self.full_camera_trajectory) if self.full_camera_trajectory else np.array([])
            camera_orientations = np.array(self.full_camera_orientations) if self.full_camera_orientations else np.array([])
            
            # Prepare current frame
            current_frame = None
            if hasattr(self, 'current_chunk_result') and self.current_chunk_result is not None:
                if 'images' in self.current_chunk_result:
                    # Get the latest image
                    images_tensor = self.current_chunk_result['images']
                    
                    # Handle different tensor shapes
                    if len(images_tensor.shape) == 5:  # (B, N, C, H, W) - batch dimension still present
                        latest_image = images_tensor[0, -1]  # Remove batch, get last frame: (C, H, W)
                    elif len(images_tensor.shape) == 4:  # (N, C, H, W) - batch already removed
                        latest_image = images_tensor[-1]  # Last frame: (C, H, W)
                    else:
                        latest_image = images_tensor  # Already single frame
                    
                    # Convert to RGB format for display (CHW -> HWC)
                    if latest_image.shape[0] == 3:  # CHW format
                        image_rgb = latest_image.permute(1, 2, 0).numpy()
                    else:
                        image_rgb = latest_image.numpy()
                    
                    # Normalize to [0, 255] range
                    if image_rgb.max() <= 1.0:
                        image_rgb = (image_rgb * 255).astype(np.uint8)
                    
                    current_frame = image_rgb
            
            # Prepare frame info and statistics
            frame_info = f"Frame: {self.stats['total_frames']}, Chunk: {self.stats['total_chunks']}"
            stats_text = f"Chunks: {self.stats['total_chunks']}, Frames: {self.stats['total_frames']}, Points: {len(all_points)}"
            
            # Send data to visualization process
            viz_data = {
                'points': all_points,
                'colors': all_colors,
                'camera_positions': camera_positions,
                'camera_orientations': camera_orientations,
                'chunk_end_poses': np.array(self.chunk_end_poses) if self.chunk_end_poses else np.array([]),  # Send chunk end poses for linestrips
                'current_frame': current_frame,
                'chunk_start_frame': self.current_chunk_start_frame,  # Send current chunk start frame
                'chunk_end_frame': self.current_chunk_end_frame,  # Send current chunk end frame
                'frame_info': frame_info,
                'stats_text': stats_text
            }
            
            # Non-blocking put to avoid blocking the main process
            try:
                self.vis_queue.put_nowait(viz_data)
            except mp.queues.Full:
                # Queue is full, skip this update
                pass
            
        except Exception as e:
            print(f"⚠️  Error sending visualization data: {e}")
    
    def process_chunk_online(self, image_paths: List, start_idx: int = None, end_idx: int = None) -> Dict:
        """
        Process a single chunk online and immediately align it with previous chunks.
        
        Args:
            image_paths: List of image paths or (video_path, frame_idx) tuples
            start_idx: Start index in the full sequence (for background loading)
            end_idx: End index in the full sequence (for background loading)
        
        Returns:
            Dictionary containing processing results and alignment info
        """
        start_time = time.time()
        
        # Extract timestamps for this chunk
        chunk_timestamps = self._extract_timestamps_from_paths(image_paths)
        self.timestamps.extend(chunk_timestamps)
        
        # Load and process chunk
        load_start = time.time()
        chunk_result = self._process_chunk_from_paths(image_paths, start_idx, end_idx)
        load_time = time.time() - load_start
        
        # Generate camera IDs for this chunk
        chunk_idx = len(self.chunk_results)
        start_frame = chunk_idx * (self.chunk_length - self.overlap)
        chunk_camera_ids = list(range(start_frame + 1, start_frame + len(image_paths) + 1))
        
        # Store chunk result
        self.chunk_results.append(chunk_result)
        self.camera_ids.append(chunk_camera_ids)
        
        # Track current chunk for frame display
        self.current_chunk_result = chunk_result
        
        # Align with previous chunks
        alignment_start = time.time()
        aligned_chunk = self._align_chunk_online(chunk_result, chunk_camera_ids)
        alignment_time = time.time() - alignment_start
        
        # Extract timing information from alignment result
        timing = aligned_chunk.get('timing', {})
        correspondence_time = timing.get('correspondence', 0.0)
        sim3_time = timing.get('sim3', 0.0)
        optimize_time = timing.get('optimize', 0.0)
        transform_time = timing.get('transform', 0.0)
        color_time = timing.get('color', 0.0)
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats['total_chunks'] += 1
        self.stats['total_frames'] += len(image_paths)
        self.stats['processing_times'].append(processing_time)
        self.stats['alignment_times'].append(alignment_time)
        
        # # Verify coordinate system consistency
        # if self.aligned_points:
        #     all_points = torch.from_numpy(np.concatenate(self.aligned_points, axis=0))
        #     all_poses = torch.from_numpy(np.concatenate(self.aligned_camera_poses, axis=0))
        #     self._verify_coordinate_system(all_points, all_poses)
        
        # Update visualization
        viz_start = time.time()
        if self.vis_running:
            self._update_visualization()
        viz_time = time.time() - viz_start
        
        # Print progress with detailed timing
        fps = len(image_paths) / processing_time if processing_time > 0 else 0
        avg_fps = self.stats['total_frames'] / sum(self.stats['processing_times']) if self.stats['processing_times'] else 0
        
        print(f"📊 Chunk {self.stats['total_chunks']}: {len(image_paths)} frames in {processing_time:.2f}s ({fps:.1f} FPS)")
        print(f"   Load: {load_time:.2f}s, Inference: {self.stats['inference_times'][-1]:.2f}s, Postprocess: {self.stats['postprocess_times'][-1]:.2f}s")
        
        # Print alignment timing if available
        if len(self.chunk_results) > 1:
            print(f"   Alignment: {alignment_time:.2f}s (Corresp: {correspondence_time:.2f}s, SIM3: {sim3_time:.2f}s, Optimize: {optimize_time:.2f}s, Transform: {transform_time:.2f}s)")
        else:
            print(f"   Alignment: {alignment_time:.2f}s (first chunk)")
        
        print(f"   Colors: {color_time:.2f}s, Visualization: {viz_time:.2f}s")
        print(f"   Total points: {len(self.aligned_points[-1]) if self.aligned_points else 0}, Average FPS: {avg_fps:.1f}")
        
        return {
            'chunk_result': chunk_result,
            'aligned_chunk': aligned_chunk,
            'processing_time': processing_time,
            'alignment_time': alignment_time,
            'camera_ids': chunk_camera_ids,
            'timestamps': chunk_timestamps
        }
    
    def process_chunks_with_background_loader(self) -> List[Dict]:
        """
        Process all chunks using the background loader for optimal performance.
        This method iterates through all chunks defined in the background loader.
        
        Returns:
            List of processing results for each chunk
        """
        if not self.loader_running or self.background_loader is None:
            raise ValueError("Background loader not started")
        
        results = []
        chunk_count = 0
        total_chunks = len(self.background_loader.dataset)
        
        print(f"\n🔄 Starting chunk-based processing with {total_chunks} chunks...")
        print("=" * 60)
        
        try:
            for chunk_data in self.background_loader:
                chunk_count += 1
                start_idx = chunk_data['start_idx'].item()  # Extract from tensor
                end_idx = chunk_data['end_idx'].item()
                chunk_images = chunk_data['chunk']  # Already correct shape
                chunk_paths = chunk_data['chunk_paths'][0]  # Extract from list of lists
                
                print(f"\n📦 Processing chunk {chunk_count}/{total_chunks}: frames {start_idx + 1}-{end_idx}")
                
                # Process chunk directly with loaded images
                result = self._process_chunk_with_images(chunk_images, chunk_paths, start_idx, end_idx)
                results.append(result)
                
                # Small delay to allow visualization to update
                if self.vis_running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print(f"\n⚠️  Processing interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            raise
        
        return results
    
    def _process_chunk_with_images(self, chunk_images: torch.Tensor, chunk_paths: List, 
                                  start_idx: int, end_idx: int) -> Dict:
        """
        Process a chunk using pre-loaded images.
        
        Args:
            chunk_images: Pre-loaded images tensor (N, 3, H, W)
            chunk_paths: List of image paths for this chunk
            start_idx: Start index in the full sequence
            end_idx: End index in the full sequence
        
        Returns:
            Dictionary containing processing results and alignment info
        """
        start_time = time.time()
        
        # Extract timestamps for this chunk
        # chunk_paths contains lists of paths, so we need to extract the actual paths
        actual_paths = []
        for path_item in chunk_paths:
            if isinstance(path_item, list):
                actual_paths.extend(path_item)
            else:
                actual_paths.append(path_item)
        
        chunk_timestamps = self._extract_timestamps_from_paths(actual_paths)
        self.timestamps.extend(chunk_timestamps)
        
        # Model inference
        inference_start = time.time()
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        print(f"chunk_images.shape: {chunk_images.shape}")
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                result = self.model(chunk_images.to(self.device))  # Add batch dimension
        inference_time = time.time() - inference_start
        
        # Process masks
        postprocess_start = time.time()
        masks = torch.sigmoid(result['conf'][..., 0]) > 0.1
        non_edge = ~depth_edge(result['local_points'][..., 2], rtol=0.03)
        masks = torch.logical_and(masks, non_edge)[0]
        
        # Move all results to CPU to save GPU memory
        cpu_transfer_start = time.time()
        cpu_result = {
            'points': result['points'][0].cpu(),
            'local_points': result['local_points'][0].cpu(),
            'camera_poses': result['camera_poses'][0].cpu(),
            'conf': result['conf'][0].cpu(),
            'masks': masks.cpu(),
            'image_paths': chunk_paths,
            'images': chunk_images.squeeze(0).cpu()  # Remove batch dimension and cache images for colorization
        }
        cpu_transfer_time = time.time() - cpu_transfer_start
        postprocess_time = time.time() - postprocess_start
        
        # Store timing info
        self.stats.setdefault('load_times', []).append(0.0)  # No load time since images are pre-loaded
        self.stats.setdefault('inference_times', []).append(inference_time)
        self.stats.setdefault('postprocess_times', []).append(postprocess_time)
        self.stats.setdefault('cpu_transfer_times', []).append(cpu_transfer_time)
        
        # Generate camera IDs for this chunk
        chunk_idx = len(self.chunk_results)
        start_frame = chunk_idx * (self.chunk_length - self.overlap)
        chunk_camera_ids = list(range(start_frame + 1, start_frame + len(chunk_paths) + 1))
        
        # Store chunk result
        self.chunk_results.append(cpu_result)
        self.camera_ids.append(chunk_camera_ids)
        
        # Track current chunk for frame display
        self.current_chunk_result = cpu_result
        
        # Align with previous chunks
        alignment_start = time.time()
        aligned_chunk = self._align_chunk_online(cpu_result, chunk_camera_ids)
        alignment_time = time.time() - alignment_start
        
        # Extract timing information from alignment result
        timing = aligned_chunk.get('timing', {})
        correspondence_time = timing.get('correspondence', 0.0)
        sim3_time = timing.get('sim3', 0.0)
        optimize_time = timing.get('optimize', 0.0)
        transform_time = timing.get('transform', 0.0)
        color_time = timing.get('color', 0.0)
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats['total_chunks'] += 1
        self.stats['total_frames'] += len(chunk_paths)
        self.stats['processing_times'].append(processing_time)
        self.stats['alignment_times'].append(alignment_time)
        
        # Update visualization
        viz_start = time.time()
        if self.vis_running:
            self._update_visualization()
        viz_time = time.time() - viz_start
        
        # Print progress with detailed timing
        fps = len(chunk_paths) / processing_time if processing_time > 0 else 0
        avg_fps = self.stats['total_frames'] / sum(self.stats['processing_times']) if self.stats['processing_times'] else 0
        
        print(f"📊 Chunk {self.stats['total_chunks']}: {len(chunk_paths)} frames in {processing_time:.2f}s ({fps:.1f} FPS)")
        print(f"   Inference: {inference_time:.2f}s, Postprocess: {postprocess_time:.2f}s")
        
        # Print alignment timing if available
        if len(self.chunk_results) > 1:
            print(f"   Alignment: {alignment_time:.2f}s (Corresp: {correspondence_time:.2f}s, SIM3: {sim3_time:.2f}s, Optimize: {optimize_time:.2f}s, Transform: {transform_time:.2f}s)")
        else:
            print(f"   Alignment: {alignment_time:.2f}s (first chunk)")
        
        print(f"   Colors: {color_time:.2f}s, Visualization: {viz_time:.2f}s")
        print(f"   Total points: {len(self.aligned_points[-1]) if self.aligned_points else 0}, Average FPS: {avg_fps:.1f}")
        
        return {
            'chunk_result': cpu_result,
            'aligned_chunk': aligned_chunk,
            'processing_time': processing_time,
            'alignment_time': alignment_time,
            'camera_ids': chunk_camera_ids,
            'timestamps': chunk_timestamps
        }
    
    def _extract_colors_from_chunk(self, chunk_result: Dict) -> torch.Tensor:
        """Extract colors from chunk result."""
        print(chunk_result['images'].shape)
        return chunk_result['images'].permute(0, 2, 3, 1).reshape(-1, 3).cpu()
    
    def _align_chunk_online(self, chunk_result: Dict, chunk_camera_ids: List[int]) -> Dict:
        """
        Align a new chunk with previously processed chunks.
        
        Args:
            chunk_result: Result from processing the new chunk
            chunk_camera_ids: Camera IDs for the new chunk
        
        Returns:
            Dictionary containing aligned chunk data
        """
        if len(self.chunk_results) == 1:
            # First chunk, no alignment needed
            aligned_points = chunk_result['points']
            aligned_poses = chunk_result['camera_poses']
            color_start = time.time()
            aligned_colors = self._extract_colors_from_chunk(chunk_result)
            color_time = time.time() - color_start
            print(f"color_time: {color_time:.3f}s (using cached images)")
            
            # Store aligned data
            self.aligned_points.append(aligned_points.reshape(-1, 3).cpu().numpy())
            self.aligned_camera_poses.append(aligned_poses.cpu().numpy())
            self.aligned_camera_ids.extend(chunk_camera_ids)
            self.aligned_colors.append(aligned_colors.cpu().numpy())
            
            # Store camera trajectory data (always kept in memory)
            poses_np = aligned_poses.cpu().numpy()
            for pose in poses_np:
                camera_pos = pose[:3, 3]  # Translation part
                camera_rot = pose[:3, :3]  # Rotation part
                self.full_camera_trajectory.append(camera_pos)
                self.full_camera_orientations.append(camera_rot)
            
            # Store end pose of this chunk for linestrip visualization
            if len(poses_np) > 0:
                end_pose = poses_np[-1]  # Last pose of the chunk
                self.chunk_end_poses.append(end_pose[:3, 3])  # Store position only
            
            # Store end frame from this chunk for visualization
            if 'images' in chunk_result:
                # Get the last frame of the chunk
                images_tensor = chunk_result['images']
                print(f"📷 Images tensor shape: {images_tensor.shape}")
                
                # Handle different tensor shapes
                if len(images_tensor.shape) == 5:  # (B, N, C, H, W) - batch dimension still present
                    end_frame = images_tensor[0, -1]  # Remove batch, get last frame: (C, H, W)
                    print(f"📷 End frame shape (5D->3D): {end_frame.shape}")
                elif len(images_tensor.shape) == 4:  # (N, C, H, W) - batch already removed
                    end_frame = images_tensor[-1]  # Last frame: (C, H, W)
                    print(f"📷 End frame shape (4D->3D): {end_frame.shape}")
                else:
                    end_frame = images_tensor  # Already single frame
                    print(f"📷 End frame shape (already 3D): {end_frame.shape}")
                
                # Convert to RGB format for display (CHW -> HWC)
                if end_frame.shape[0] == 3:  # CHW format
                    frame_rgb = end_frame.permute(1, 2, 0).numpy()
                    print(f"📷 Frame RGB shape (CHW->HWC): {frame_rgb.shape}")
                else:
                    frame_rgb = end_frame.numpy()
                    print(f"📷 Frame RGB shape (direct): {frame_rgb.shape}")
                
                # Normalize to [0, 255] range
                if frame_rgb.max() <= 1.0:
                    frame_rgb = (frame_rgb * 255).astype(np.uint8)
                
                print(f"📷 Final frame shape: {frame_rgb.shape}, dtype: {frame_rgb.dtype}")
                self.current_chunk_end_frame = frame_rgb
            
            # Manage memory after storing new chunk
            self._manage_memory()
            
            # Set as current reference chunk (like offline version)
            self.current_reference_chunk = {
                'points': aligned_points,
                'camera_poses': aligned_poses,
                'conf': chunk_result['conf'],
                'masks': chunk_result['masks']
            }
            self.current_reference_camera_ids = chunk_camera_ids
            
            return {
                'points': aligned_points,
                'camera_poses': aligned_poses,
                'colors': aligned_colors,
                'transformation': np.eye(4),
                'timing': {
                    'correspondence': 0.0,
                    'sim3': 0.0,
                    'optimize': 0.0,
                    'transform': 0.0,
                    'color': color_time
                }
            }
        
        # Align with current reference chunk (like offline version)
        current_chunk = self.current_reference_chunk
        current_camera_ids = self.current_reference_camera_ids
        
        # Find overlapping cameras
        overlap_cameras_current = current_camera_ids[-self.overlap:] if len(current_camera_ids) >= self.overlap else current_camera_ids
        overlap_cameras_next = chunk_camera_ids[:self.overlap] if len(chunk_camera_ids) >= self.overlap else chunk_camera_ids
        
        common_camera_ids = set(overlap_cameras_current) & set(overlap_cameras_next)
        
        if len(common_camera_ids) >= 2:
            # Find corresponding points
            correspondence_start = time.time()
            corresponding_points1, corresponding_points2 = find_corresponding_points(
                current_chunk['points'], chunk_result['points'],
                current_camera_ids, chunk_camera_ids,
                conf1=current_chunk.get('conf', None),
                conf2=chunk_result.get('conf', None),
                subsample_factor=self.correspondence_subsample_factor,  # Use configurable subsampling
                conf_threshold=self.conf_threshold,
                use_robust_filtering=self.use_robust_alignment
            )
            correspondence_time = time.time() - correspondence_start
            
            if len(corresponding_points1) >= 10:
                # Estimate SIM3 transformation from 3D points using robust method
                sim3_start = time.time()
                
                # Choose robust estimation method based on configuration and number of points
                if self.use_robust_alignment:
                    if self.robust_alignment_method == 'ransac' or (self.robust_alignment_method == 'auto' and len(corresponding_points1) >= 50):
                        # Use RANSAC for large point sets or when explicitly requested
                        sim3_transform = estimate_sim3_transformation_robust(
                            corresponding_points2, corresponding_points1,
                            max_iterations=50,  # Reduced for speed
                            inlier_threshold=0.05,  # 5cm threshold
                            min_inlier_ratio=0.4
                        )
                        print(f"  🔧 Using RANSAC SIM3 estimation with {len(corresponding_points1)} points")
                    elif self.robust_alignment_method == 'irls' or (self.robust_alignment_method == 'auto' and len(corresponding_points1) >= 20):
                        # Use IRLS for medium point sets or when explicitly requested
                        sim3_transform = estimate_sim3_transformation_robust_irls(
                            corresponding_points2, corresponding_points1,
                            max_iterations=15,
                            convergence_threshold=1e-5
                        )
                        print(f"  🔧 Using IRLS SIM3 estimation with {len(corresponding_points1)} points")
                    else:
                        # Use standard estimation for small point sets
                        sim3_transform = estimate_sim3_transformation(corresponding_points2, corresponding_points1)
                        print(f"  🔧 Using standard SIM3 estimation with {len(corresponding_points1)} points")
                else:
                    # Use standard estimation when robust alignment is disabled
                    sim3_transform = estimate_sim3_transformation(corresponding_points2, corresponding_points1)
                    print(f"  🔧 Using standard SIM3 estimation with {len(corresponding_points1)} points")
                
                sim3_time = time.time() - sim3_start
                
                # Optimize SIM3 transformation using overlapping camera poses
                optimize_start = time.time()
                optimized_transform = self._optimize_sim3_transformation(
                    sim3_transform, current_chunk, chunk_result, 
                    current_camera_ids, chunk_camera_ids, common_camera_ids
                )
                optimize_time = time.time() - optimize_start

                transformation = optimized_transform.get_matrix()
                
                print(f"  Optimized SIM3 transformation with scale={optimized_transform.s:.3f}")
                
                # Apply transformation
                transform_start = time.time()
                transformed_points = self._apply_transformation_to_points(
                    chunk_result['points'], transformation
                )
                transformed_poses = self._apply_transformation_to_poses(
                    chunk_result['camera_poses'], transformation
                )
                transform_time = time.time() - transform_start
                
                # Extract colors for transformed points
                color_start = time.time()
                aligned_colors = self._extract_colors_from_chunk(chunk_result)
                color_time = time.time() - color_start
                
                # Store aligned data (excluding overlap)
                overlap_size = self.overlap
                self.aligned_points.append(transformed_points[overlap_size:].reshape(-1, 3).cpu().numpy())
                self.aligned_camera_poses.append(transformed_poses[overlap_size:].cpu().numpy())
                self.aligned_camera_ids.extend(chunk_camera_ids[overlap_size:])
                self.aligned_colors.append(aligned_colors[overlap_size:].reshape(-1, 3).cpu().numpy())
                
                # Store camera trajectory data (always kept in memory) - excluding overlap
                poses_np = transformed_poses[overlap_size:].cpu().numpy()
                for pose in poses_np:
                    camera_pos = pose[:3, 3]  # Translation part
                    camera_rot = pose[:3, :3]  # Rotation part
                    self.full_camera_trajectory.append(camera_pos)
                    self.full_camera_orientations.append(camera_rot)
                
                # Store start and end frames from this chunk for visualization
                if 'images' in chunk_result:
                    # Get the first and last frames of the chunk
                    images_tensor = chunk_result['images']
                    
                    # Handle different tensor shapes
                    if len(images_tensor.shape) == 5:  # (B, N, C, H, W) - batch dimension still present
                        start_frame = images_tensor[0, 0]  # Remove batch, get first frame: (C, H, W)
                        end_frame = images_tensor[0, -1]  # Remove batch, get last frame: (C, H, W)
                    elif len(images_tensor.shape) == 4:  # (N, C, H, W) - batch already removed
                        start_frame = images_tensor[0]  # First frame: (C, H, W)
                        end_frame = images_tensor[-1]  # Last frame: (C, H, W)
                    else:
                        start_frame = images_tensor  # Already single frame
                        end_frame = images_tensor  # Already single frame
                    
                    # Convert to RGB format for display (CHW -> HWC)
                    if start_frame.shape[0] == 3:  # CHW format
                        start_rgb = start_frame.permute(1, 2, 0).numpy()
                    else:
                        start_rgb = start_frame.numpy()
                    
                    if end_frame.shape[0] == 3:  # CHW format
                        end_rgb = end_frame.permute(1, 2, 0).numpy()
                    else:
                        end_rgb = end_frame.numpy()
                    
                    # Normalize to [0, 255] range
                    if start_rgb.max() <= 1.0:
                        start_rgb = (start_rgb * 255).astype(np.uint8)
                    if end_rgb.max() <= 1.0:
                        end_rgb = (end_rgb * 255).astype(np.uint8)
                    
                    self.current_chunk_start_frame = start_rgb
                    self.current_chunk_end_frame = end_rgb
            
                # Manage memory after storing new chunk
                self._manage_memory()
                
                # Update current reference chunk for next iteration (like offline version)
                self.current_reference_chunk = {
                    'points': transformed_points,
                    'camera_poses': transformed_poses,
                    'conf': chunk_result['conf'],
                    'masks': chunk_result['masks']
                }
                self.current_reference_camera_ids = chunk_camera_ids
                
                return {
                    'points': transformed_points,
                    'camera_poses': transformed_poses,
                    'colors': aligned_colors,
                    'transformation': transformation,
                    'timing': {
                        'correspondence': correspondence_time,
                        'sim3': sim3_time,
                        'optimize': optimize_time,
                        'transform': transform_time,
                        'color': color_time
                    }
                }
            else:
                print(f"⚠️  Warning: Insufficient corresponding points ({len(corresponding_points1)}), using identity transformation")
        else:
            print(f"⚠️  Warning: Insufficient overlapping cameras ({len(common_camera_ids)}), using identity transformation")
    
        # Fallback: no transformation
        aligned_points = chunk_result['points']
        aligned_poses = chunk_result['camera_poses']
        color_start = time.time()
        aligned_colors = self._extract_colors_from_chunk(chunk_result)
        color_time = time.time() - color_start
        print(f"color_time: {color_time:.3f}s (using cached images)")
        
        # Store aligned data
        self.aligned_points.append(aligned_points.reshape(-1, 3).cpu().numpy())
        self.aligned_camera_poses.append(aligned_poses.cpu().numpy())
        self.aligned_camera_ids.extend(chunk_camera_ids)
        self.aligned_colors.append(aligned_colors.reshape(-1, 3).cpu().numpy())
        
        # Store camera trajectory data (always kept in memory)
        poses_np = aligned_poses.cpu().numpy()
        for pose in poses_np:
            camera_pos = pose[:3, 3]  # Translation part
            camera_rot = pose[:3, :3]  # Rotation part
            self.full_camera_trajectory.append(camera_pos)
            self.full_camera_orientations.append(camera_rot)
        
        # Store end pose of this chunk for linestrip visualization
        if len(poses_np) > 0:
            end_pose = poses_np[-1]  # Last pose of the chunk
            self.chunk_end_poses.append(end_pose[:3, 3])  # Store position only
        
        # Store start and end frames from this chunk for visualization
        if 'images' in chunk_result:
            # Get the first and last frames of the chunk
            images_tensor = chunk_result['images']
            
            # Handle different tensor shapes
            if len(images_tensor.shape) == 5:  # (B, N, C, H, W) - batch dimension still present
                start_frame = images_tensor[0, 0]  # Remove batch, get first frame: (C, H, W)
                end_frame = images_tensor[0, -1]  # Remove batch, get last frame: (C, H, W)
            elif len(images_tensor.shape) == 4:  # (N, C, H, W) - batch already removed
                start_frame = images_tensor[0]  # First frame: (C, H, W)
                end_frame = images_tensor[-1]  # Last frame: (C, H, W)
            else:
                start_frame = images_tensor  # Already single frame
                end_frame = images_tensor  # Already single frame
            
            # Convert to RGB format for display (CHW -> HWC)
            if start_frame.shape[0] == 3:  # CHW format
                start_rgb = start_frame.permute(1, 2, 0).numpy()
            else:
                start_rgb = start_frame.numpy()
            
            if end_frame.shape[0] == 3:  # CHW format
                end_rgb = end_frame.permute(1, 2, 0).numpy()
            else:
                end_rgb = end_frame.numpy()
            
            # Normalize to [0, 255] range
            if start_rgb.max() <= 1.0:
                start_rgb = (start_rgb * 255).astype(np.uint8)
            if end_rgb.max() <= 1.0:
                end_rgb = (end_rgb * 255).astype(np.uint8)
            
            self.current_chunk_start_frame = start_rgb
            self.current_chunk_end_frame = end_rgb
        
        # Manage memory after storing new chunk
        self._manage_memory()
        
        # Update current reference chunk for next iteration (like offline version)
        self.current_reference_chunk = {
            'points': aligned_points,
            'camera_poses': aligned_poses,
            'conf': chunk_result['conf'],
            'masks': chunk_result['masks']
        }
        self.current_reference_camera_ids = chunk_camera_ids
        
        return {
            'points': aligned_points,
            'camera_poses': aligned_poses,
            'colors': aligned_colors,
            'transformation': np.eye(4),
            'timing': {
                'correspondence': 0.0,
                'sim3': 0.0,
                'optimize': 0.0,
                'transform': 0.0,
                'color': color_time
            }
        }
    
    def _create_fallback_colors(self, num_points: int) -> torch.Tensor:
        """Create fallback colors when image extraction fails."""
        colors = torch.zeros(num_points, 3)
        for i in range(num_points):
            hue = (i / num_points) % 1.0
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors[i] = torch.tensor([r, g, b])
        return colors
    
    def _process_chunk_from_paths(self, image_paths: List, start_idx: int = None, end_idx: int = None) -> Dict:
        """
        Process a single chunk of images from file paths.
        
        Args:
            image_paths: List of image paths or (video_path, frame_idx) tuples
            start_idx: Start index in the full sequence (for background loading)
            end_idx: End index in the full sequence (for background loading)
        
        Returns:
            Dictionary containing model outputs (moved to CPU)
        """
        # Load images for this chunk using background loader if available
        load_start = time.time()
        if self.loader_running and start_idx is not None and end_idx is not None:
            chunk_images = self.get_chunk_images(start_idx, end_idx)
        else:
            chunk_images = self._load_images_from_paths(image_paths)
        load_time = time.time() - load_start
        
        # Model inference
        inference_start = time.time()
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                result = self.model(chunk_images[None].to(self.device))  # Add batch dimension
        inference_time = time.time() - inference_start
        
        # Process masks
        postprocess_start = time.time()
        masks = torch.sigmoid(result['conf'][..., 0]) > 0.1
        non_edge = ~depth_edge(result['local_points'][..., 2], rtol=0.03)
        masks = torch.logical_and(masks, non_edge)[0]
        
        # Move all results to CPU to save GPU memory
        cpu_transfer_start = time.time()
        cpu_result = {
            'points': result['points'][0].cpu(),
            'local_points': result['local_points'][0].cpu(),
            'camera_poses': result['camera_poses'][0].cpu(),
            'conf': result['conf'][0].cpu(),
            'masks': masks.cpu(),
            'image_paths': image_paths,
            'images': chunk_images.squeeze(0).cpu()  # Remove batch dimension and cache images for colorization
        }
        cpu_transfer_time = time.time() - cpu_transfer_start
        postprocess_time = time.time() - postprocess_start
        
        # Store timing info
        self.stats.setdefault('load_times', []).append(load_time)
        self.stats.setdefault('inference_times', []).append(inference_time)
        self.stats.setdefault('postprocess_times', []).append(postprocess_time)
        self.stats.setdefault('cpu_transfer_times', []).append(cpu_transfer_time)
        
        return cpu_result
    
    def _load_images_from_paths(self, image_paths: List) -> torch.Tensor:
        """
        Load images from file paths and resize to uniform target size.
        
        Args:
            image_paths: List of image paths or (video_path, frame_idx) tuples
        
        Returns:
            Tensor of images (N, 3, H, W) on device
        """
        # Calculate target size from first image if not already set
        if self.target_size is None:
            self.target_size = self._calculate_target_size(image_paths[0])
        
        images = []
        
        for path_info in image_paths:
            if isinstance(path_info, tuple):
                # Video frame
                video_path, frame_idx = path_info
                from pi3.utils.basic import load_video_frame
                
                # Load frame using torchcodec (with fallback to OpenCV)
                image = load_video_frame(video_path, frame_idx, self.target_size, use_torchcodec=True)
                
                # Apply undistortion if maps are provided
                if self.undistortion_maps is not None:
                    # Convert tensor to numpy for undistortion
                    image_np = image.permute(1, 2, 0).numpy()  # CHW to HWC
                    undistorted_np = self.undistortion_maps.undistort_image(image_np, self.target_size)
                    # Convert back to tensor
                    image = torch.from_numpy(undistorted_np).float() / 255.0
                    image = image.permute(2, 0, 1)  # HWC to CHW
            else:
                # Use PIL for image loading
                from PIL import Image
                import torchvision.transforms as transforms
                
                pil_image = Image.open(path_info).convert('RGB')
                
                # Apply undistortion if maps are provided
                if self.undistortion_maps is not None:
                    # Convert PIL to numpy for undistortion
                    image_np = np.array(pil_image)
                    undistorted_np = self.undistortion_maps.undistort_image(image_np, self.target_size)
                    # Convert back to tensor
                    image = torch.from_numpy(undistorted_np).float() / 255.0
                    image = image.permute(2, 0, 1)  # HWC to CHW
                else:
                    # Use PIL transforms for regular loading
                    transform = transforms.Compose([
                        transforms.Resize(self.target_size),  # (height, width)
                        transforms.ToTensor()
                    ])
                    image = transform(pil_image)
            images.append(image)
        
        # Stack images
        images_tensor = torch.stack(images)
        return images_tensor
    
    def _calculate_target_size(self, first_image_path) -> tuple:
        """
        Calculate target size based on the first image.
        
        Args:
            first_image_path: Path to the first image or (video_path, frame_idx) tuple
        
        Returns:
            Tuple of (height, width) for target size
        """
        import math
        from PIL import Image
        
        # Load first image to get original dimensions
        if isinstance(first_image_path, tuple):
            # Video frame
            video_path, frame_idx = first_image_path
            from pi3.utils.basic import load_video_frame
            
            # Load frame using torchcodec (with fallback to OpenCV)
            frame_tensor = load_video_frame(video_path, frame_idx, use_torchcodec=True)
            
            # Convert tensor to PIL Image
            frame_np = frame_tensor.permute(1, 2, 0).numpy()  # CHW to HWC
            pil_image = Image.fromarray((frame_np * 255).astype(np.uint8), mode='RGB')
        else:
            # Image file
            pil_image = Image.open(first_image_path).convert('RGB')
        
        # Calculate target size
        W_orig, H_orig = pil_image.size
        scale = math.sqrt(self.pixel_limit / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target, H_target = W_orig * scale, H_orig * scale
        k, m = round(W_target / 14), round(H_target / 14)
        while (k * 14) * (m * 14) > self.pixel_limit:
            if k / m > W_target / H_target: k -= 1
            else: m -= 1
        TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
        
        return (TARGET_H, TARGET_W)  # Return as (height, width)
    
    def _apply_transformation_to_points(self, points: torch.Tensor, transformation: np.ndarray) -> torch.Tensor:
        """Apply similarity transformation (SIM3) to 3D points."""
        points_homog = homogenize_points(points)
        transformation_tensor = torch.from_numpy(transformation).to(torch.float32)
        transformed_points_homog = (transformation_tensor @ points_homog.view(-1, 4).T).T
        transformed_points_homog = transformed_points_homog.view(points.shape[0], points.shape[1], points.shape[2], 4)
        return transformed_points_homog[:,:,:,:3]
    
    def _apply_transformation_to_poses(self, poses: torch.Tensor, transformation: np.ndarray) -> torch.Tensor:
        """Apply similarity transformation (SIM3) to camera poses."""
        transformed_poses = []
        transformation_tensor = torch.from_numpy(transformation).to(torch.float32)
        
        for pose in poses:
            transformed_pose = transformation_tensor @ pose
            transformed_poses.append(transformed_pose)
        
        return torch.stack(transformed_poses)
    
    def _compute_rigid_transformation(self, source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        """
        Compute optimal rigid transformation using Procrustes analysis.
        
        Args:
            source_points: Source points (N, 3)
            target_points: Target points (N, 3)
        
        Returns:
            4x4 transformation matrix
        """
        # Center the points
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        
        centered_source = source_points - source_centroid
        centered_target = target_points - target_centroid
        
        # Compute rotation using SVD
        H = centered_source.T @ centered_target
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation matrix (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = target_centroid - R @ source_centroid
        
        # Build 4x4 transformation matrix
        transformation = np.eye(4)
        transformation[:3, :3] = R
        transformation[:3, 3] = t
        
        return transformation
    
    def _optimize_sim3_transformation(self, initial_transform: SIM3Transformation, 
                                    current_chunk: Dict, next_chunk: Dict,
                                    current_camera_ids: List[int], next_camera_ids: List[int],
                                    common_camera_ids: set) -> SIM3Transformation:
        """
        Optimize SIM3 transformation by minimizing pose alignment error.
        
        Args:
            initial_transform: Initial SIM3 transformation from point correspondence
            current_chunk: Current chunk data
            next_chunk: Next chunk data
            current_camera_ids: Camera IDs in current chunk
            next_camera_ids: Camera IDs in next chunk
            common_camera_ids: Set of overlapping camera IDs
        
        Returns:
            Optimized SIM3Transformation
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            print("  Warning: scipy not available, using initial transformation")
            return initial_transform
        
        # Get overlapping camera poses
        current_poses = []
        next_poses = []
        
        for camera_id in sorted(common_camera_ids):
            # Find camera index in current chunk
            current_idx = current_camera_ids.index(camera_id)
            current_poses.append(current_chunk['camera_poses'][current_idx].numpy())
            
            # Find camera index in next chunk
            next_idx = next_camera_ids.index(camera_id)
            next_poses.append(next_chunk['camera_poses'][next_idx].numpy())
        
        current_poses = np.stack(current_poses)  # (N, 4, 4)
        next_poses = np.stack(next_poses)        # (N, 4, 4)
        
        # Extract camera positions (translation part)
        current_positions = current_poses[:, :3, 3]  # (N, 3)
        next_positions = next_poses[:, :3, 3]        # (N, 3)
        
        # Extract camera orientations (rotation part)
        current_rotations = current_poses[:, :3, :3]  # (N, 3, 3)
        next_rotations = next_poses[:, :3, :3]        # (N, 3, 3)
        
        def residual_function(params):
            """Residual function for least squares optimization."""
            # Unpack parameters: [scale, rx, ry, rz, tx, ty, tz]
            scale = params[0]
            rodrigues = params[1:4]
            translation = params[4:7]
            
            # Convert Rodrigues parameters to rotation matrix
            R = rodrigues_to_rotation_matrix(rodrigues)
            
            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = scale * R
            T[:3, 3] = translation
            
            residuals = []
            
            # Weighting factors for balancing position and rotation
            pos_weight = 1.0
            rot_weight = 0.1  # Rotation residuals are typically smaller in magnitude
            
            # Apply transformation to next chunk poses
            for i, pose in enumerate(next_poses):
                transformed_pose = T @ pose
                
                # Position residuals (3 components per camera) - weighted
                pos_residual = pos_weight * (current_positions[i] - transformed_pose[:3, 3])
                residuals.extend(pos_residual)
                
                # Rotation residuals using Rodrigues representation - weighted
                relative_rot = current_rotations[i] @ transformed_pose[:3, :3].T
                relative_rodrigues = rotation_matrix_to_rodrigues(relative_rot)
                rot_residual = rot_weight * relative_rodrigues
                residuals.extend(rot_residual)
            
            return np.array(residuals)
        
        # Initial parameters from the initial transformation
        initial_scale = initial_transform.s
        initial_rotation = initial_transform.R
        initial_translation = initial_transform.t
        
        # Convert rotation matrix to Rodrigues parameters
        initial_rodrigues = rotation_matrix_to_rodrigues(initial_rotation)
        
        initial_params = np.concatenate([
            [initial_scale],
            initial_rodrigues,
            initial_translation
        ])
        
        # Optimize using least squares
        print(f"  Optimizing SIM3 transformation with {len(common_camera_ids)} overlapping cameras using least squares...")
        
        try:
            from scipy.optimize import least_squares
            
            
            result = least_squares(
                residual_function,
                initial_params,
                method='lm',
                max_nfev=100,  # Maximum function evaluations
                ftol=1e-6,     # Function tolerance
                xtol=1e-6,     # Parameter tolerance
                verbose=0
            )
            
            # Convert least_squares result to minimize-like format
            class ResultWrapper:
                def __init__(self, ls_result):
                    self.success = ls_result.success
                    self.x = ls_result.x
                    self.fun = np.sum(ls_result.fun ** 2)  # Sum of squared residuals
                    self.message = ls_result.message
            
            result = ResultWrapper(result)
            
        except ImportError:
            print("  Warning: scipy.optimize.least_squares not available, using initial transformation")
            return initial_transform
        
        if result.success:
            # Extract optimized parameters
            optimized_scale = result.x[0]
            optimized_rodrigues = result.x[1:4]
            optimized_translation = result.x[4:7]
            
            # Convert back to rotation matrix
            optimized_rotation = rodrigues_to_rotation_matrix(optimized_rodrigues)
            
            optimized_transform = SIM3Transformation(
                optimized_scale, optimized_rotation, optimized_translation
            )
            
            # Calculate final residuals for diagnostics
            final_residuals = residual_function(result.x)
            pos_residuals = final_residuals[::6]  # Every 6th residual (position components)
            rot_residuals = final_residuals[3::6]  # Every 6th residual starting from 3rd (rotation components)
            
            print(f"  Optimization successful. Final cost: {result.fun:.6f}")
            print(f"  Position RMS: {np.sqrt(np.mean(pos_residuals**2)):.6f}")
            print(f"  Rotation RMS: {np.sqrt(np.mean(rot_residuals**2)):.6f}")
            return optimized_transform
        else:
            print(f"  Optimization failed: {result.message}, using initial transformation")
            return initial_transform
    
    def _verify_coordinate_system(self, points: torch.Tensor, camera_poses: torch.Tensor) -> bool:
        """
        Verify coordinate system consistency.
        
        Args:
            points: 3D points tensor
            camera_poses: Camera poses tensor
        
        Returns:
            True if coordinate system is consistent
        """
        try:
            # Check if camera poses are reasonable (not too far from origin)
            camera_positions = camera_poses[:, :3, 3]
            max_distance = torch.max(torch.norm(camera_positions, dim=1))
            
            if max_distance > 1000:  # Arbitrary threshold
                print(f"  Warning: Large camera distances detected (max: {max_distance:.2f})")
                return False
            
            # Check if points are reasonable (not too far from cameras)
            if len(points) > 0:
                point_distances = torch.norm(points, dim=-1)
                max_point_distance = torch.max(point_distances)
                
                if max_point_distance > 1000:  # Arbitrary threshold
                    print(f"  Warning: Large point distances detected (max: {max_point_distance:.2f})")
                    return False
            
            return True
            
        except Exception as e:
            print(f"  Warning: Error during coordinate system verification: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        if not self.stats['processing_times']:
            return self.stats
        
        avg_processing_time = np.mean(self.stats['processing_times'])
        avg_alignment_time = np.mean(self.stats['alignment_times'])
        total_processing_time = sum(self.stats['processing_times'])
        overall_fps = self.stats['total_frames'] / total_processing_time if total_processing_time > 0 else 0
        
        return {
            **self.stats,
            'avg_processing_time': avg_processing_time,
            'avg_alignment_time': avg_alignment_time,
            'total_processing_time': total_processing_time,
            'overall_fps': overall_fps
        }
    
    def print_timing_summary(self):
        """Print detailed timing summary."""
        if not self.stats['processing_times']:
            print("No timing data available")
            return
        
        print("\n" + "="*60)
        print("📊 DETAILED TIMING SUMMARY")
        print("="*60)
        
        # Overall statistics
        total_time = sum(self.stats['processing_times'])
        total_frames = self.stats['total_frames']
        overall_fps = total_frames / total_time if total_time > 0 else 0
        
        print(f"🎯 Overall Performance:")
        print(f"   • Total chunks: {self.stats['total_chunks']}")
        print(f"   • Total frames: {total_frames}")
        print(f"   • Total time: {total_time:.2f}s")
        print(f"   • Overall FPS: {overall_fps:.1f}")
        
        # Average times per chunk
        if self.stats.get('load_times'):
            avg_load = sum(self.stats['load_times']) / len(self.stats['load_times'])
            avg_inference = sum(self.stats['inference_times']) / len(self.stats['inference_times'])
            avg_postprocess = sum(self.stats['postprocess_times']) / len(self.stats['postprocess_times'])
            avg_cpu_transfer = sum(self.stats['cpu_transfer_times']) / len(self.stats['cpu_transfer_times'])
            avg_alignment = sum(self.stats['alignment_times']) / len(self.stats['alignment_times'])
            
            print(f"\n⏱️  Average Times per Chunk:")
            print(f"   • Image loading: {avg_load:.3f}s ({avg_load/total_time*100:.1f}%)")
            print(f"   • Model inference: {avg_inference:.3f}s ({avg_inference/total_time*100:.1f}%)")
            print(f"   • Post-processing: {avg_postprocess:.3f}s ({avg_postprocess/total_time*100:.1f}%)")
            print(f"   • CPU transfer: {avg_cpu_transfer:.3f}s ({avg_cpu_transfer/total_time*100:.1f}%)")
            print(f"   • Alignment: {avg_alignment:.3f}s ({avg_alignment/total_time*100:.1f}%)")
        
        # Bottleneck analysis
        if self.stats.get('load_times'):
            max_load = max(self.stats['load_times'])
            max_inference = max(self.stats['inference_times'])
            max_alignment = max(self.stats['alignment_times'])
            
            print(f"\n🚨 Bottleneck Analysis:")
            print(f"   • Slowest image loading: {max_load:.3f}s")
            print(f"   • Slowest inference: {max_inference:.3f}s")
            print(f"   • Slowest alignment: {max_alignment:.3f}s")
            
            # Identify the main bottleneck
            bottlenecks = [
                ("Image Loading", max_load),
                ("Model Inference", max_inference),
                ("Alignment", max_alignment)
            ]
            main_bottleneck = max(bottlenecks, key=lambda x: x[1])
            print(f"   • Main bottleneck: {main_bottleneck[0]} ({main_bottleneck[1]:.3f}s)")
        
        print("="*60)
    
    def save_final_result(self, save_path: str, max_points: int = 1000000):
        """
        Save the final aligned trajectory result.
        
        Args:
            save_path: Path to save the .ply file
            max_points: Maximum number of points to save
        """
        if not self.aligned_points:
            print("No points to save")
            return
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Combine all aligned points and colors
        all_points = np.vstack(self.aligned_points)
        all_colors = np.vstack(self.aligned_colors)
        
        # Subsample if needed
        if len(all_points) > max_points:
            indices = np.random.choice(len(all_points), max_points, replace=False)
            all_points = all_points[indices]
            all_colors = all_colors[indices]
        
        # Save point cloud
        print(f"💾 Saving final trajectory to: {save_path}")
        write_ply(torch.from_numpy(all_points), torch.from_numpy(all_colors), save_path, max_points=max_points)
        print(f"✅ Saved trajectory with {len(all_points)} points from {self.stats['total_chunks']} chunks")
        
        # Save camera poses using full trajectory
        if self.full_camera_trajectory:
            camera_positions = np.array(self.full_camera_trajectory)
            camera_colors = np.full((len(camera_positions), 3), [1.0, 0.0, 0.0])
            
            camera_filename = save_path.replace('.ply', '_camera_poses.ply')
            write_ply(torch.from_numpy(camera_positions), torch.from_numpy(camera_colors), camera_filename)
            print(f"📷 Saved {len(camera_positions)} camera poses to: {camera_filename}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.stop_background_loader()
            self.stop_visualization()
            
            # Clean up disk cache if it exists
            if hasattr(self, 'cache_dir') and self.cache_dir is not None:
                try:
                    import shutil
                    if os.path.exists(self.cache_dir):
                        shutil.rmtree(self.cache_dir)
                        print(f"🧹 Cleaned up disk cache: {self.cache_dir}")
                except Exception as e:
                    print(f"⚠️  Failed to clean up disk cache: {e}")
        except:
            pass  # Ignore errors during cleanup
    
    def cleanup_disk_cache(self):
        """Manually clean up disk cache files."""
        if hasattr(self, 'cache_dir') and self.cache_dir is not None:
            try:
                import shutil
                if os.path.exists(self.cache_dir):
                    shutil.rmtree(self.cache_dir)
                    print(f"🧹 Cleaned up disk cache: {self.cache_dir}")
                    self.cache_dir = None
            except Exception as e:
                print(f"⚠️  Failed to clean up disk cache: {e}")
    
    def save_trajectory_tum(self, save_path: str, timestamps: List[float] = None):
        """
        Save the final aligned trajectory in TUM format.
        
        Args:
            save_path: Path to save the TUM format trajectory file
            timestamps: Optional list of timestamps for each pose (if None, will use stored timestamps or frame indices)
        """
        if not self.full_camera_trajectory:
            print("No camera trajectory available to save")
            return
        
        try:
            from scipy.spatial.transform import Rotation
            
            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Use stored timestamps if available, otherwise use provided timestamps or frame indices
            if timestamps is None:
                if self.timestamps and len(self.timestamps) >= len(self.full_camera_trajectory):
                    timestamps = self.timestamps[:len(self.full_camera_trajectory)]
                else:
                    timestamps = list(range(len(self.full_camera_trajectory)))
            
            print(f"💾 Saving trajectory in TUM format to: {save_path}")
            print(f"   Using {len(timestamps)} timestamps for {len(self.full_camera_trajectory)} poses")
            
            with open(save_path, 'w') as f:
                # TUM format header
                f.write("# timestamp tx ty tz qx qy qz qw\n")
                
                for i, (camera_pos, camera_rot) in enumerate(zip(self.full_camera_trajectory, self.full_camera_orientations)):
                    # Get timestamp (use stored timestamp if available, otherwise use frame index)
                    t = timestamps[i] if i < len(timestamps) else float(i)
                    
                    # Extract translation (x, y, z)
                    x, y, z = camera_pos
                    
                    # Convert rotation matrix to quaternion
                    # Note: scipy uses quaternion format [x, y, z, w]
                    quat = Rotation.from_matrix(camera_rot).as_quat()
                    qx, qy, qz, qw = quat
                    
                    # Write TUM format line: timestamp tx ty tz qx qy qz qw
                    # Use full precision for timestamp to match ground truth format
                    f.write(f"{t:.9f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
            
            print(f"✅ Saved trajectory with {len(self.full_camera_trajectory)} poses to: {save_path}")
            
        except ImportError:
            print("❌ Error: scipy.spatial.transform.Rotation not available")
        except Exception as e:
            print(f"❌ Error saving trajectory: {e}") 
    
    def _extract_timestamps_from_paths(self, image_paths: List) -> List[float]:
        """
        Extract timestamps from image paths or video frames.
        
        Args:
            image_paths: List of image paths or (video_path, frame_idx) tuples
        
        Returns:
            List of timestamps in nanoseconds
        """
        timestamps = []
        
        for path_info in image_paths:
            if isinstance(path_info, tuple):
                # Video frame - calculate timestamp from frame index and video metadata
                video_path, frame_idx = path_info
                timestamp = self._get_video_frame_timestamp(video_path, frame_idx)
                timestamps.append(timestamp)
            else:
                # Image file - extract timestamp from filename
                timestamp = self._extract_timestamp_from_filename(path_info)
                timestamps.append(timestamp)
        
        return timestamps
    
    def _get_video_frame_timestamp(self, video_path: str, frame_idx: int) -> float:
        """
        Get timestamp for a video frame based on frame index and video metadata.
        
        Args:
            video_path: Path to video file
            frame_idx: Frame index
        
        Returns:
            Timestamp in nanoseconds
        """
        # Cache video metadata to avoid repeated loading
        if not hasattr(self, '_video_metadata_cache'):
            self._video_metadata_cache = {}
        
        if video_path not in self._video_metadata_cache:
            try:
                # Try to get metadata using torchcodec
                if TORCHCODEC_AVAILABLE:
                    from torchcodec.decoders import VideoDecoder
                    decoder = VideoDecoder(video_path, device="cpu")
                    metadata = decoder.metadata
                    self._video_metadata_cache[video_path] = {
                        'fps': metadata.average_fps,
                        'duration': metadata.duration_seconds,
                        'total_frames': metadata.num_frames
                    }
                    del decoder
                else:
                    # Fallback to OpenCV
                    import cv2
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = total_frames / fps if fps > 0 else 0
                    cap.release()
                    
                    self._video_metadata_cache[video_path] = {
                        'fps': fps,
                        'duration': duration,
                        'total_frames': total_frames
                    }
            except Exception as e:
                print(f"Warning: Could not get video metadata for {video_path}: {e}")
                # Use default values
                self._video_metadata_cache[video_path] = {
                    'fps': 30.0,
                    'duration': 0.0,
                    'total_frames': 0
                }
        
        metadata = self._video_metadata_cache[video_path]
        fps = metadata['fps']
        
        if fps > 0:
            # Calculate timestamp in seconds, then convert to nanoseconds
            timestamp_seconds = frame_idx / fps
            timestamp_nanoseconds = timestamp_seconds * 1e9
            return timestamp_nanoseconds
        else:
            # Fallback: use frame index as timestamp (in nanoseconds)
            return float(frame_idx) * 1e9
    
    def _extract_timestamp_from_filename(self, image_path: str) -> float:
        """
        Extract timestamp from image filename.
        Assumes filename contains a nanosecond timestamp.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Timestamp in nanoseconds
        """
        import os
        import re
        
        filename = os.path.basename(image_path)
        
        # Try to extract timestamp from filename
        # Common patterns: 1403636580838555648.jpg, 1403636580838555648.000000000.jpg, etc.
        timestamp_patterns = [
            r'(\d{16,19})',  # 16-19 digit timestamp
            r'(\d{10,13})',  # 10-13 digit timestamp (seconds or milliseconds)
        ]
        
        for pattern in timestamp_patterns:
            match = re.search(pattern, filename)
            if match:
                timestamp_str = match.group(1)
                timestamp = float(timestamp_str)
                
                # If it's a short timestamp (10-13 digits), assume it's in seconds and convert to nanoseconds
                if len(timestamp_str) <= 13:
                    timestamp *= 1e9
                
                return timestamp
        
        # If no timestamp found, use file modification time as fallback
        try:
            mtime = os.path.getmtime(image_path)
            return mtime * 1e9  # Convert to nanoseconds
        except:
            # Last resort: use a default timestamp
            return 0.0
        
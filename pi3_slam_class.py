import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge, homogenize_points
import scipy.optimize
from scipy.spatial.distance import cdist


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


def estimate_sim3_transformation(points1: np.ndarray, points2: np.ndarray) -> SIM3Transformation:
    """
    Estimate SIM3 transformation between two sets of corresponding points.
    
    Args:
        points1: First set of points (N, 3)
        points2: Second set of points (N, 3)
    
    Returns:
        SIM3Transformation object
    """
    print(f"SIM3 estimation: points1 shape {points1.shape}, points2 shape {points2.shape}")
    
    if len(points1) < 3 or len(points2) < 3:
        raise ValueError(f"Need at least 3 corresponding points, got {len(points1)} and {len(points2)}")
    
    # Ensure both point sets have the same number of points
    if len(points1) != len(points2):
        min_len = min(len(points1), len(points2))
        points1 = points1[:min_len]
        points2 = points2[:min_len]
        print(f"Warning: Truncated points to {min_len} for SIM3 estimation")
    
    # Double-check shapes before proceeding
    print(f"After truncation: points1 shape {points1.shape}, points2 shape {points2.shape}")
    
    # Center the points
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)
    
    centered_points1 = points1 - centroid1
    centered_points2 = points2 - centroid2
    
    print(f"Centered points shapes: {centered_points1.shape}, {centered_points2.shape}")
    
    # Compute scale
    scale1 = np.sqrt(np.sum(centered_points1 ** 2, axis=1))
    scale2 = np.sqrt(np.sum(centered_points2 ** 2, axis=1))
    
    # Use median scale ratio for robustness
    scale_ratios = scale2 / (scale1 + 1e-8)
    s = np.median(scale_ratios)
    
    # Estimate rotation using SVD
    print(f"Computing H matrix: {centered_points1.T.shape} @ {centered_points2.shape}")
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
                            conf_threshold: float = 0.5, threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find corresponding points between two chunks based on camera IDs, using confidence filtering.
    
    Args:
        points1: Points from first chunk (N, 3)
        points2: Points from second chunk (M, 3)
        camera_ids1: Camera IDs for first chunk
        camera_ids2: Camera IDs for second chunk
        conf1: Confidence values for first chunk (N, 1) or (N, H, W, 1)
        conf2: Confidence values for second chunk (M, 1) or (M, H, W, 1)
        conf_threshold: Confidence threshold for filtering points (default: 0.5)
        threshold: Distance threshold for correspondence
    
    Returns:
        Tuple of corresponding points (N_corr, 3), (N_corr, 3)
    """
    # Find common camera IDs
    common_ids = set(camera_ids1) & set(camera_ids2)
    
    if len(common_ids) == 0:
        return np.array([]), np.array([])
    
    print(f"Common camera IDs: {common_ids}")
    
    # Collect all corresponding points
    all_corr_points1 = []
    all_corr_points2 = []
    
    for cam_id in common_ids:
        idx1 = camera_ids1.index(cam_id)
        idx2 = camera_ids2.index(cam_id)
        
        print(f"Processing camera ID {cam_id}: idx1={idx1}, idx2={idx2}")
        
        # Get points for this camera
        pts1 = points1[idx1].view(-1, 3).cpu().numpy()  # (H*W, 3)
        pts2 = points2[idx2].view(-1, 3).cpu().numpy()  # (H*W, 3)
        
        print(f"Camera {cam_id}: pts1 shape {pts1.shape}, pts2 shape {pts2.shape}")
        
        # Find valid points (non-zero)
        valid1 = np.linalg.norm(pts1, axis=1) > 1e-6
        valid2 = np.linalg.norm(pts2, axis=1) > 1e-6
        
        # Apply confidence filtering if confidence values are provided
        if conf1 is not None:
            # Get confidence values for this camera
            cam_conf1 = conf1[idx1].view(-1, 1).cpu().numpy()  # (H, W, 1) or (H*W, 1)
            if len(cam_conf1.shape) == 3:
                cam_conf1 = cam_conf1.reshape(-1)  # Flatten to (H*W,)
            elif len(cam_conf1.shape) == 2:
                cam_conf1 = cam_conf1.reshape(-1)  # Flatten to (H*W,)
            
            # Apply sigmoid and threshold
            cam_conf1 = 1 / (1 + np.exp(-cam_conf1))  # sigmoid
            high_conf1 = cam_conf1 > conf_threshold
            valid1 = np.logical_and(valid1, high_conf1)
        
        if conf2 is not None:
            # Get confidence values for this camera
            cam_conf2 = conf2[idx2].view(-1, 1).cpu().numpy()  # (H, W, 1) or (H*W, 1)
            if len(cam_conf2.shape) == 3:
                cam_conf2 = cam_conf2.reshape(-1)  # Flatten to (H*W,)
            elif len(cam_conf2.shape) == 2:
                cam_conf2 = cam_conf2.reshape(-1)  # Flatten to (H*W,)
            
            # Apply sigmoid and ttshreshold
            cam_conf2 = 1 / (1 + np.exp(-cam_conf2))  # sigmoid
            high_conf2 = cam_conf2 > conf_threshold
            valid2 = np.logical_and(valid2, high_conf2)
        
        if np.sum(valid1) > 0 and np.sum(valid2) > 0:
            valid_pts = np.logical_and(valid1, valid2)

            points1_selected = pts1[valid_pts]
            points2_selected = pts2[valid_pts]
                
            all_corr_points1.extend(points1_selected)
            all_corr_points2.extend(points2_selected)
    
    if not all_corr_points1:
        return np.array([]), np.array([])
    
    # Convert to numpy arrays
    all_points1 = np.array(all_corr_points1)
    all_points2 = np.array(all_corr_points2)
    
    print(f"After collecting: all_points1 shape {all_points1.shape}, all_points2 shape {all_points2.shape}")
    
    # Ensure we have the same number of points from both chunks
    min_total = min(len(all_points1), len(all_points2))
    if min_total > 0:
        result1 = all_points1[:min_total]
        result2 = all_points2[:min_total]
        print(f"Final result shapes: {result1.shape}, {result2.shape}")
        return result1, result2
    else:
        return np.array([]), np.array([])


class Pi3SLAM:
    """
    Pi3SLAM class for processing longer trajectories in chunks with overlap.
    """
    
    def __init__(self, model: Pi3, chunk_length: int = 10, overlap: int = 3, device: str = 'cuda', conf_threshold: float = 0.5, undistortion_maps=None):
        """
        Initialize Pi3SLAM.
        
        Args:
            model: Pre-trained Pi3 model
            chunk_length: Number of frames per chunk
            overlap: Number of overlapping frames between chunks
            device: Device to run inference on
            conf_threshold: Confidence threshold for filtering points in correspondence search (default: 0.5)
            target_size: Target image size (height, width) for resizing (calculated automatically from first image)
        """
        self.model = model
        self.chunk_length = chunk_length
        self.overlap = overlap
        self.device = device
        self.conf_threshold = conf_threshold
        
        # Store results for each chunk
        self.chunk_results = []
        self.camera_ids = []
        
        # Timestamp tracking
        self.timestamps = []
        
        # Calculate target size based on first image (will be set during first chunk processing)
        self.target_size = None
        self.pixel_limit = 255000  # Same as in pi3/utils/basic.py
        
        # Undistortion support
        self.undistortion_maps = undistortion_maps
        
    def process_sequence(self, data_path: str) -> Dict:
        """
        Process a sequence of images/video in chunks.
        
        Args:
            data_path: Path to image directory or video file
            interval: Sampling interval for frames
        
        Returns:
            Dictionary containing aligned trajectory results
        """
        print(f"Loading sequence from: {data_path}")
        
        # Get list of image files instead of loading all images
        if data_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # For video, we'll need to extract frames
            from pi3.utils.basic import get_video_frame_count
            
            total_frames = get_video_frame_count(data_path, use_torchcodec=True)
            
            # Create frame indices
            frame_indices = list(range(0, total_frames, 1))
            image_paths = [(data_path, idx) for idx in frame_indices]  # (video_path, frame_idx)
        else:
            # For image directory, get file paths
            import os
            import glob
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            image_paths = []
            
            for ext in image_extensions:
                image_paths.extend(glob.glob(os.path.join(data_path, ext)))
                image_paths.extend(glob.glob(os.path.join(data_path, ext.upper())))
            
            image_paths.sort()
        
        if len(image_paths) == 0:
            raise ValueError("No images found in the sequence")
        
        print(f"Found {len(image_paths)} images")
        print(f"Processing in chunks of {self.chunk_length} with {self.overlap} overlap")
        
        # Extract timestamps for all images
        print("Extracting timestamps from image paths...")
        self.timestamps = self._extract_timestamps_from_paths(image_paths)
        print(f"Extracted {len(self.timestamps)} timestamps")
        
        # Store data path and image paths for later loading
        self.data_path = data_path
        self.image_paths = image_paths
        
        # Process chunks with progress tracking
        chunk_results = []
        camera_ids = []
        
        # Calculate total number of chunks
        total_chunks = 0
        for chunk_idx in range(0, len(image_paths), self.chunk_length - self.overlap):
            end_idx = min(chunk_idx + self.chunk_length, len(image_paths))
            if end_idx - chunk_idx >= 2:
                total_chunks += 1
        
        print(f"Processing {total_chunks} chunks...")
        
        # Import tqdm for progress bar
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            print("tqdm not available, using simple progress output")
        
        import time
        chunk_times = []
        
        chunk_idx = 0
        while chunk_idx < len(image_paths):
            end_idx = min(chunk_idx + self.chunk_length, len(image_paths))
            chunk_image_paths = image_paths[chunk_idx:end_idx]
            
            if len(chunk_image_paths) < 2:
                chunk_idx += self.chunk_length - self.overlap
                continue
            
            # Start timing
            start_time = time.time()
            
            # Create progress description
            chunk_num = len(chunk_results) + 1
            progress_desc = f"Chunk {chunk_num}/{total_chunks}: frames {chunk_idx + 1}-{end_idx}"
            
            if use_tqdm:
                # Use tqdm progress bar
                with tqdm(total=1, desc=progress_desc, leave=False) as pbar:
                    # Load and process this chunk
                    chunk_result = self._process_chunk_from_paths(chunk_image_paths)
                    pbar.update(1)
            else:
                # Simple progress output
                print(f"Processing {progress_desc}")
                chunk_result = self._process_chunk_from_paths(chunk_image_paths)
            
            # End timing and calculate metrics
            end_time = time.time()
            chunk_time = end_time - start_time
            chunk_times.append(chunk_time)
            
            # Calculate FPS for this chunk
            num_frames = len(chunk_image_paths)
            fps = num_frames / chunk_time if chunk_time > 0 else 0
            
            # Calculate average FPS so far
            total_frames_so_far = sum(len(ids) for ids in camera_ids)
            total_time_so_far = sum(chunk_times)
            avg_fps = total_frames_so_far / total_time_so_far if total_time_so_far > 0 else 0
            
            # Display timing information
            if use_tqdm:
                tqdm.write(f"  ✓ {progress_desc} - {chunk_time:.2f}s ({fps:.1f} FPS) - Avg: {avg_fps:.1f} FPS")
            else:
                print(f"  ✓ Completed in {chunk_time:.2f}s ({fps:.1f} FPS) - Avg: {avg_fps:.1f} FPS")
            
            chunk_results.append(chunk_result)
            
            # Generate camera IDs for this chunk (sequential starting from chunk_idx + 1)
            chunk_camera_ids = list(range(chunk_idx + 1, end_idx + 1))
            camera_ids.append(chunk_camera_ids)
            
            # Move to next chunk
            chunk_idx += self.chunk_length - self.overlap
        
        self.chunk_results = chunk_results
        self.camera_ids = camera_ids
        
        # Print final timing summary
        if chunk_times:
            total_time = sum(chunk_times)
            total_frames = sum(len(ids) for ids in camera_ids)
            overall_fps = total_frames / total_time if total_time > 0 else 0
            avg_chunk_time = sum(chunk_times) / len(chunk_times)
            min_chunk_time = min(chunk_times)
            max_chunk_time = max(chunk_times)
            
            print(f"\n📊 Processing Summary:")
            print(f"  • Total chunks processed: {len(chunk_results)}")
            print(f"  • Total frames: {total_frames}")
            print(f"  • Total processing time: {total_time:.2f}s")
            print(f"  • Overall FPS: {overall_fps:.1f}")
            print(f"  • Average chunk time: {avg_chunk_time:.2f}s")
            print(f"  • Fastest chunk: {min_chunk_time:.2f}s")
            print(f"  • Slowest chunk: {max_chunk_time:.2f}s")
            
            # Estimate time for different sequence lengths
            if overall_fps > 0:
                print(f"\n⏱️  Time Estimates:")
                for frames in [100, 500, 1000, 5000]:
                    estimated_time = frames / overall_fps
                    print(f"  • {frames} frames: {estimated_time:.1f}s ({estimated_time/60:.1f}min)")
        
        # Align chunks using sequential alignment
        print("\n🔄 Aligning chunks...")
        align_start_time = time.time()
        aligned_result = self._align_chunks()
        align_time = time.time() - align_start_time
        print(f"  ✓ Chunk alignment completed in {align_time:.2f}s")
        
        # Add image paths to result for later coloring
        aligned_result['image_paths'] = image_paths
        
        return aligned_result
    
    def _process_chunk_from_paths(self, image_paths: List) -> Dict:
        """
        Process a single chunk of images from file paths.
        
        Args:
            image_paths: List of image paths or (video_path, frame_idx) tuples
        
        Returns:
            Dictionary containing model outputs (moved to CPU)
        """
        # Load images for this chunk
        chunk_images = self._load_images_from_paths(image_paths)
        
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                result = self.model(chunk_images[None].to(self.device))  # Add batch dimension
        
        # Process masks
        masks = torch.sigmoid(result['conf'][..., 0]) > 0.1
        non_edge = ~depth_edge(result['local_points'][..., 2], rtol=0.03)
        masks = torch.logical_and(masks, non_edge)[0]
        
        # Move all results to CPU to save GPU memory
        return {
            'points': result['points'][0].cpu(),
            'local_points': result['local_points'][0].cpu(),
            'camera_poses': result['camera_poses'][0].cpu(),
            'conf': result['conf'][0].cpu(),  # Store confidence values
            'masks': masks.cpu(),
            'image_paths': image_paths  # Store image paths for later coloring
        }
    
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
                # Image file
                from PIL import Image
                import torchvision.transforms as transforms
                
                # Apply undistortion if maps are provided
                if self.undistortion_maps is not None:
                    # Load with PIL and convert to numpy for undistortion
                    pil_image = Image.open(path_info).convert('RGB')
                    image_np = np.array(pil_image)
                    
                    # Apply undistortion
                    image = self.undistortion_maps.undistort_image(image_np, self.target_size)
                    
                    # Convert to tensor
                    image = torch.from_numpy(image).float() / 255.0
                    image = image.permute(2, 0, 1)  # HWC to CHW
                else:
                    # Use PIL for regular loading
                    image = Image.open(path_info).convert('RGB')
                    transform = transforms.Compose([
                        transforms.Resize(self.target_size),  # (height, width)
                        transforms.ToTensor()
                    ])
                    image = transform(image)
            
            images.append(image)
        
        # Stack images
        images_tensor = torch.stack(images)
        return images_tensor
    
    def _load_all_images_from_paths(self, image_paths: List, subsample_factor: int = 1) -> torch.Tensor:
        """
        Load all images from file paths for coloring and resize to uniform target size.
        
        Args:
            image_paths: List of image paths or (video_path, frame_idx) tuples
        
        Returns:
            Tensor of images (N, 3, H, W) on CPU
        """
        # Use calculated target size
        if self.target_size is None:
            self.target_size = self._calculate_target_size(image_paths[0])

        image_paths = image_paths[::subsample_factor]
        
        print(f"Loading {len(image_paths)} images for coloring with target size {self.target_size}...")
        
        # Import tqdm for progress bar
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
        
        import time
        start_time = time.time()
        images = []
        
        # Create progress bar or simple counter
        if use_tqdm:
            pbar = tqdm(total=len(image_paths), desc="Loading images", unit="img")
        else:
            print("Loading images...")
        
        for i, path_info in enumerate(image_paths):
            if isinstance(path_info, tuple):
                # Video frame
                video_path, frame_idx = path_info
                from pi3.utils.basic import load_video_frame
                
                # Load frame using torchcodec (with fallback to OpenCV)
                image = load_video_frame(video_path, frame_idx, self.target_size, use_torchcodec=True)
            else:
                # Image file
                from PIL import Image
                import torchvision.transforms as transforms
                
                image = Image.open(path_info).convert('RGB')
                transform = transforms.Compose([
                    transforms.Resize(self.target_size),  # (height, width)
                    transforms.ToTensor()
                ])
                image = transform(image)
            
            images.append(image)
            
            # Update progress
            if use_tqdm:
                pbar.update(1)
            elif i % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(image_paths) - i - 1) / rate if rate > 0 else 0
                print(f"  Loaded {i+1}/{len(image_paths)} images ({rate:.1f} img/s, ETA: {eta:.1f}s)")
        
        if use_tqdm:
            pbar.close()
        
        # Stack images and keep on CPU for coloring
        images_tensor = torch.stack(images)
        total_time = time.time() - start_time
        load_rate = len(images) / total_time if total_time > 0 else 0
        
        print(f"✓ Loaded {len(images)} images in {total_time:.2f}s ({load_rate:.1f} img/s)")
        print(f"  Final tensor shape: {images_tensor.shape}")
        
        return images_tensor
    
    def _calculate_target_size(self, first_image_path) -> tuple:
        """
        Calculate target size based on the first image using the exact logic from pi3/utils/basic.py.
        
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
        
        # --- Use exact logic from pi3/utils/basic.py ---
        # This is necessary to ensure all tensors have the same dimensions for stacking.
        W_orig, H_orig = pil_image.size
        scale = math.sqrt(self.pixel_limit / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target, H_target = W_orig * scale, H_orig * scale
        k, m = round(W_target / 14), round(H_target / 14)
        while (k * 14) * (m * 14) > self.pixel_limit:
            if k / m > W_target / H_target: k -= 1
            else: m -= 1
        TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
        print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")
        
        return (TARGET_H, TARGET_W)  # Return as (height, width) for consistency
    
    def _process_chunk(self, images: torch.Tensor) -> Dict:
        """
        Process a single chunk of images (legacy method for compatibility).
        
        Args:
            images: Tensor of images (N, 3, H, W)
        
        Returns:
            Dictionary containing model outputs (moved to CPU)
        """
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                result = self.model(images[None])  # Add batch dimension
        
        # Process masks
        masks = torch.sigmoid(result['conf'][..., 0]) > 0.1
        non_edge = ~depth_edge(result['local_points'][..., 2], rtol=0.03)
        masks = torch.logical_and(masks, non_edge)[0]
        
        # Move all results to CPU to save GPU memory
        return {
            'points': result['points'][0].cpu(),
            'local_points': result['local_points'][0].cpu(),
            'camera_poses': result['camera_poses'][0].cpu(),
            'conf': result['conf'][0].cpu(),  # Store confidence values
            'masks': masks.cpu()
        }
    
    def _align_chunks(self) -> Dict:
        """
        Align all chunks using simple sequential alignment.
        Each chunk is aligned to the previous one using overlapping cameras.
        
        Returns:
            Dictionary containing aligned trajectory
        """
        if len(self.chunk_results) == 0:
            return {}
        
        if len(self.chunk_results) == 1:
            # Single chunk, no alignment needed
            return self._merge_single_chunk(self.chunk_results[0], self.camera_ids[0])
        
        print("Aligning chunks using sequential alignment...")

        # Start with the first chunk as reference
        aligned_confidences = []
        aligned_masks = []
        aligned_points = []
        aligned_camera_poses = []
        aligned_camera_ids = []

        current_chunk = self.chunk_results[0]
        current_camera_ids = self.camera_ids[0]
        
        # Add first chunk
        aligned_points.append(current_chunk['points'])
        aligned_camera_poses.append(current_chunk['camera_poses'])
        aligned_camera_ids.extend(current_camera_ids)
        aligned_confidences.append(current_chunk['conf'])
        aligned_masks.append(current_chunk['masks'])
        
        # Align subsequent chunks
        for i in range(1, len(self.chunk_results)):
            next_chunk = self.chunk_results[i]
            next_camera_ids = self.camera_ids[i]
            
            print(f"Aligning chunk {i+1} to chunk {i}...")
            
            # Find overlapping cameras between current and next chunk
            overlap_cameras_current = current_camera_ids[-self.overlap:] if len(current_camera_ids) >= self.overlap else current_camera_ids
            overlap_cameras_next = next_camera_ids[:self.overlap] if len(next_camera_ids) >= self.overlap else next_camera_ids
            
            # Find common camera IDs
            common_camera_ids = set(overlap_cameras_current) & set(overlap_cameras_next)
            
            if len(common_camera_ids) >= 2:  # Need at least 2 cameras for alignment
                print(f"  Found {len(common_camera_ids)} overlapping cameras: {sorted(common_camera_ids)}")
                
                # Find corresponding 3D points between overlapping cameras
                current_points = current_chunk['points']
                next_points = next_chunk['points']
                
                # Find corresponding points using camera IDs
                corresponding_points1, corresponding_points2 = find_corresponding_points(
                    current_points, next_points,
                    current_camera_ids, next_camera_ids,
                    conf1=current_chunk.get('conf', None),
                    conf2=next_chunk.get('conf', None),
                    conf_threshold=self.conf_threshold
                )
                
                if len(corresponding_points1) >= 10:  # Need sufficient points for robust estimation
                    print(f"  Found {len(corresponding_points1)} corresponding 3D points for transformation estimation")
                    
                    # Estimate SIM3 transformation from 3D points
                    sim3_transform = estimate_sim3_transformation(corresponding_points2, corresponding_points1)
                    
                    # Optimize SIM3 transformation using overlapping camera poses
                    optimized_transform = self._optimize_sim3_transformation(
                        sim3_transform, current_chunk, next_chunk, 
                        current_camera_ids, next_camera_ids, common_camera_ids
                    )
                    transformation = optimized_transform.get_matrix()
                    
                    print(f"  Optimized SIM3 transformation with scale={optimized_transform.s:.3f}")
                else:
                    print(f"  Warning: Insufficient corresponding points ({len(corresponding_points1)}), using identity transformation")
                    transformation = np.eye(4)
                
                # Apply transformation to next chunk
                transformed_points = self._apply_transformation_to_points(
                    next_chunk['points'], transformation
                )
                
                transformed_poses = self._apply_transformation_to_poses(
                    next_chunk['camera_poses'], transformation
                )
                
                # Add transformed chunk (excluding overlap)
                overlap_size = self.overlap
                aligned_points.append(transformed_points[overlap_size:])
                aligned_camera_poses.append(transformed_poses[overlap_size:])
                aligned_camera_ids.extend(next_camera_ids[overlap_size:])
                aligned_confidences.append(next_chunk['conf'][overlap_size:])
                aligned_masks.append(next_chunk['masks'][overlap_size:])
                
                # Update current chunk for next iteration
                current_chunk = {
                    'points': transformed_points,
                    'camera_poses': transformed_poses,
                    'conf': next_chunk['conf'],
                    'masks': next_chunk['masks']
                }
                current_camera_ids = next_camera_ids
                
            else:
                print(f"  Warning: Insufficient overlapping cameras ({len(common_camera_ids)}), skipping alignment")
                # Add chunk without transformation
                aligned_points.append(next_chunk['points'])
                aligned_camera_poses.append(next_chunk['camera_poses'])
                aligned_camera_ids.extend(next_camera_ids)
                aligned_confidences.append(next_chunk['conf'])
                aligned_masks.append(next_chunk['masks'])
        
        # Merge all aligned chunks
        all_points = torch.cat(aligned_points, dim=0)
        all_camera_poses = torch.cat(aligned_camera_poses, dim=0)
        all_confidences = torch.cat(aligned_confidences, dim=0)
        all_masks = torch.cat(aligned_masks, dim=0)
        
        # Verify coordinate system consistency
        print("Verifying coordinate system consistency...")
        self._verify_coordinate_system(all_points, all_camera_poses)
        
        return {
            'points': all_points,
            'camera_poses': all_camera_poses,
            'camera_ids': aligned_camera_ids,
            'num_chunks': len(self.chunk_results),
            'chunk_results': self.chunk_results,  # Store individual chunks for separate saving
            'conf': all_confidences,
            'masks': all_masks
        }
    
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
    
    def _apply_transformation_to_points(self, points: torch.Tensor, transformation: np.ndarray) -> torch.Tensor:
        """
        Apply similarity transformation (SIM3) to 3D points.
        
        Args:
            points: 3D points tensor (N, 3) - on CPU
            transformation: 4x4 SIM3 transformation matrix (includes scale)
        
        Returns:
            Transformed points tensor (on CPU)
        """
        points_homog = homogenize_points(points)  # Shape: (N, 4)
        # Create transformation tensor on CPU
        transformation_tensor = torch.from_numpy(transformation).to(torch.float32)
        # Apply SIM3 transformation: T @ points_homog.T
        transformed_points_homog = (transformation_tensor @ points_homog.view(-1, 4).T).T  # Shape: (N, 4)
        transformed_points_homog = transformed_points_homog.view(points.shape[0], points.shape[1], points.shape[2], 4)
        # Return only the 3D coordinates (homogeneous division is handled by the transformation matrix)
        return transformed_points_homog[:,:,:,:3]
    
    def _apply_transformation_to_poses(self, poses: torch.Tensor, transformation: np.ndarray) -> torch.Tensor:
        """
        Apply similarity transformation (SIM3) to camera poses.
        
        Args:
            poses: Camera poses tensor (N, 4, 4) - on CPU
            transformation: 4x4 SIM3 transformation matrix (includes scale)
        
        Returns:
            Transformed poses tensor (on CPU)
        """
        transformed_poses = []
        transformation_tensor = torch.from_numpy(transformation).to(torch.float32)
        
        for pose in poses:
            # Apply SIM3 transformation: new_pose = transformation @ pose
            # This properly handles the scale component of the similarity transformation
            transformed_pose = transformation_tensor @ pose
            transformed_poses.append(transformed_pose)
        
        return torch.stack(transformed_poses)
    
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
        
        def objective_function(params):
            """Objective function to minimize pose alignment error."""
            # Unpack parameters: [scale, rx, ry, rz, tx, ty, tz]
            # where rx, ry, rz are rotation angles (Euler angles)
            scale = params[0]
            rotation_angles = params[1:4]
            translation = params[4:7]
            
            # Convert Euler angles to rotation matrix
            from scipy.spatial.transform import Rotation
            R = Rotation.from_euler('xyz', rotation_angles).as_matrix()
            
            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = scale * R
            T[:3, 3] = translation
            
            # Apply transformation to next chunk poses
            transformed_next_poses = []
            for pose in next_poses:
                transformed_pose = T @ pose
                transformed_next_poses.append(transformed_pose)
            transformed_next_poses = np.stack(transformed_next_poses)
            
            # Compute alignment error
            position_error = np.mean(np.linalg.norm(
                current_positions - transformed_next_poses[:, :3, 3], axis=1
            ))
            
            # Compute rotation error (Frobenius norm of rotation difference)
            rotation_error = 0
            for i in range(len(current_rotations)):
                rot_diff = current_rotations[i] @ transformed_next_poses[i, :3, :3].T
                rotation_error += np.linalg.norm(rot_diff - np.eye(3), 'fro')
            rotation_error /= len(current_rotations)
            
            # Total error (weighted combination)
            total_error = position_error + 0.1 * rotation_error
            
            return total_error
        
        # Initial parameters from the initial transformation
        initial_scale = initial_transform.s
        initial_rotation = initial_transform.R
        initial_translation = initial_transform.t
        
        # Convert rotation matrix to Euler angles
        from scipy.spatial.transform import Rotation
        initial_euler = Rotation.from_matrix(initial_rotation).as_euler('xyz')
        
        initial_params = np.concatenate([
            [initial_scale],
            initial_euler,
            initial_translation
        ])
        
        # Optimize
        print(f"  Optimizing SIM3 transformation with {len(common_camera_ids)} overlapping cameras...")
        result = minimize(
            objective_function, 
            initial_params, 
            method='L-BFGS-B',
            options={'maxiter': 100, 'disp': False}
        )
        
        if result.success:
            # Extract optimized parameters
            optimized_scale = result.x[0]
            optimized_euler = result.x[1:4]
            optimized_translation = result.x[4:7]
            
            # Convert back to rotation matrix
            optimized_rotation = Rotation.from_euler('xyz', optimized_euler).as_matrix()
            
            optimized_transform = SIM3Transformation(
                optimized_scale, optimized_rotation, optimized_translation
            )
            
            print(f"  Optimization successful. Final error: {result.fun:.6f}")
            return optimized_transform
        else:
            print(f"  Optimization failed: {result.message}, using initial transformation")
            return initial_transform
    
    def _merge_chunks_without_optimization(self) -> Dict:
        """
        Merge chunks without any optimization when pose graph optimization fails.
        Simply concatenates chunks without alignment.
        
        Returns:
            Dictionary containing merged trajectory
        """
        print("Merging chunks without optimization...")
        
        # Simply concatenate all chunks
        aligned_points = []
        aligned_camera_poses = []
        aligned_camera_ids = []
        
        for chunk_idx, (chunk_result, camera_ids) in enumerate(zip(self.chunk_results, self.camera_ids)):
            if chunk_idx == 0:
                # Add first chunk completely
                aligned_points.append(chunk_result['points'])
                aligned_camera_poses.append(chunk_result['camera_poses'])
                aligned_camera_ids.extend(camera_ids)
            else:
                # Add subsequent chunks excluding overlap
                overlap_size = self.overlap
                aligned_points.append(chunk_result['points'][overlap_size:])
                aligned_camera_poses.append(chunk_result['camera_poses'][overlap_size:])
                aligned_camera_ids.extend(camera_ids[overlap_size:])
        
        # Merge all chunks
        all_points = torch.cat(aligned_points, dim=0)
        all_camera_poses = torch.cat(aligned_camera_poses, dim=0)
        
        return {
            'points': all_points,
            'camera_poses': all_camera_poses,
            'camera_ids': aligned_camera_ids,
            'num_chunks': len(self.chunk_results),
            'chunk_results': self.chunk_results  # Store individual chunks for separate saving
        }
    

    
    def _merge_single_chunk(self, chunk_result: Dict, camera_ids: List[int]) -> Dict:
        """Merge a single chunk result."""
        return {
            'points': chunk_result['points'],
            'camera_poses': chunk_result['camera_poses'],
            'camera_ids': camera_ids,
            'num_chunks': 1
        }
    
    def save_result(self, result: Dict, save_path: str, max_points: int = 1000000,
                   save_chunks: bool = False, save_camera_poses: bool = False, use_chunk_colors: bool = False,
                   conf_threshold: float = 0.5, subsample_factor: int = 4):
        """
        Save the aligned trajectory result with colors.
        
        Args:
            result: Result dictionary from process_sequence
            save_path: Path to save the .ply file
            max_points: Maximum number of points to save
            images: Original images tensor (N, 3, H, W) for coloring (optional, will use result['original_images'] if available)
            save_chunks: Whether to save individual chunks separately (deprecated)
            save_camera_poses: Whether to save camera poses as colored points
            use_chunk_colors: Whether to use chunk-based colors instead of image colors
            conf_threshold: Confidence threshold for filtering points
            subsample_factor: Factor to subsa[mple points (take every nth point)
        """
        if 'points' not in result:
            print("No points to save")
            return
        
        # Filter points by confidence and subsample
        points = result['points'][::subsample_factor]
        masks = result['masks'][::subsample_factor]
        conf = result['conf'][::subsample_factor]

        # Load images from paths for coloring
        image_paths = result.get('image_paths', [])
        if image_paths:
            print("Loading images from paths for coloring...")
            images = self._load_all_images_from_paths(image_paths, subsample_factor)
        else:
            print("Warning: No image paths available, using fallback colors")
            images = None
        
        # Flatten confidence tensor to match points shape
        if conf is not None:
            conf_flat = conf.flatten()  # Shape: [100*378*672]
            points_flat = points.reshape(-1, 3)  # Shape: [100*378*672, 3]
            masks_flat = masks.flatten()  # Shape: [100*378*672]
            
            # Filter by confidence
            high_conf_mask = conf_flat > conf_threshold
            points = points_flat[high_conf_mask]
            masks = masks_flat[high_conf_mask]
            conf = conf_flat[high_conf_mask]
            print(f"Filtered points by confidence: {len(points_flat)} -> {len(points)} points")
        else:
            print(f"Warning: No confidence values available")
            points = points.reshape(-1, 3)  # Flatten points anyway
        
        # Extract colors from images
        if images is not None:
            rgb_flat = images.permute(0, 2, 3, 1).reshape(-1, 3)  # Shape: [100*378*672, 3]
            if conf is not None:
                rgb = rgb_flat[high_conf_mask]
            else:
                rgb = rgb_flat
        else:
            # Fallback colors
            rgb = torch.full((len(points), 3), 0.5)  # Gray color


        print(f"Saving aligned trajectory to: {save_path}")
        write_ply(points, rgb, save_path, max_points=max_points)
        print(f"Saved trajectory with {len(points)} points from {result['num_chunks']} chunks")
        
        # Save camera poses as colored points (always save)
        self._save_camera_poses(result, save_path)
    
    def save_trajectory_tum(self, save_path: str, result: Dict, timestamps: List[float] = None):
        """
        Save the aligned trajectory in TUM format.
        
        Args:
            save_path: Path to save the TUM format trajectory file
            result: Result dictionary from process_sequence
            timestamps: Optional list of timestamps for each pose (if None, will use stored timestamps or frame indices)
        """
        if 'camera_poses' not in result:
            print("No camera poses available to save")
            return
        
        try:
            from scipy.spatial.transform import Rotation
            
            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            camera_poses = result['camera_poses']
            camera_ids = result.get('camera_ids', [])
            
            # Use stored timestamps if available, otherwise use provided timestamps or frame indices
            if timestamps is None:
                if self.timestamps and len(self.timestamps) >= len(camera_poses):
                    timestamps = self.timestamps[:len(camera_poses)]
                else:
                    timestamps = list(range(len(camera_poses)))
            
            print(f"💾 Saving trajectory in TUM format to: {save_path}")
            print(f"   Using {len(timestamps)} timestamps for {len(camera_poses)} poses")
            
            with open(save_path, 'w') as f:
                # TUM format header
                f.write("# timestamp tx ty tz qx qy qz qw\n")
                
                for i, pose in enumerate(camera_poses):
                    # Get timestamp (use stored timestamp if available, otherwise use frame index)
                    t = timestamps[i] if i < len(timestamps) else float(i)
                    
                    # Convert pose tensor to numpy
                    pose_np = pose.numpy()
                    
                    # Extract translation (x, y, z)
                    x, y, z = pose_np[:3, 3]
                    
                    # Extract rotation matrix and convert to quaternion
                    rotation_matrix = pose_np[:3, :3]
                    quat = Rotation.from_matrix(rotation_matrix).as_quat()
                    qx, qy, qz, qw = quat
                    
                    # Write TUM format line: timestamp tx ty tz qx qy qz qw
                    # Use full precision for timestamp to match ground truth format
                    f.write(f"{t:.9f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
            
            print(f"✅ Saved trajectory with {len(camera_poses)} poses to: {save_path}")
            
        except ImportError:
            print("❌ Error: scipy.spatial.transform.Rotation not available")
        except Exception as e:
            print(f"❌ Error saving trajectory: {e}")
    
    def _verify_coordinate_system(self, points: torch.Tensor, camera_poses: torch.Tensor) -> bool:
        """
        Verify that the coordinate system is consistent.
        
        Args:
            points: World coordinates of points (N, 3)
            camera_poses: Camera-to-world transformations (N, 4, 4)
        
        Returns:
            True if coordinate system is consistent
        """
        if len(points) == 0 or len(camera_poses) == 0:
            return True
            
        print(f"Verifying {len(camera_poses)} camera poses...")
        
        # Check that camera poses are valid transformations
        for i, pose in enumerate(camera_poses):
            pose_np = pose.numpy()
            R = pose_np[:3, :3]
            t = pose_np[:3, 3]
            
            # Check if rotation matrix is orthogonal
            RRt = R @ R.T
            if not np.allclose(RRt, np.eye(3), atol=1e-3):
                print(f"Warning: Camera pose {i} has non-orthogonal rotation matrix")
                return False
                
            # Check if determinant is 1 (proper rotation)
            det = np.linalg.det(R)
            if not np.isclose(det, 1.0, atol=1e-3):
                print(f"Warning: Camera pose {i} has improper rotation (det={det:.6f})")
                return False
        
        print("Coordinate system verification passed - all camera poses are valid transformations")
        return True
    
    def _save_individual_chunks(self, result: Dict, base_save_path: str, images: torch.Tensor = None):
        """
        Save individual chunks as separate files.
        
        Args:
            result: Result dictionary from process_sequence
            base_save_path: Base path for saving (will append chunk indices)
            images: Original images tensor for coloring
        """
        if 'chunk_results' not in result:
            print("No individual chunks available for saving")
            return
        
        print("Saving individual chunks...")
        
        # Generate different colors for each chunk
        chunk_colors = self._generate_chunk_colors(len(self.chunk_results))
        
        for i, chunk_result in enumerate(self.chunk_results):
            chunk_points = chunk_result['points']
            
            # Create colored point cloud for this chunk
            if images is not None:
                # Use actual colors from images for this chunk
                # Get the correct image indices for this chunk
                start_img_idx = self.camera_ids[i][0] - 1  # Convert to 0-based index
                end_img_idx = self.camera_ids[i][-1]  # Keep 1-based for slicing
                chunk_images = images[start_img_idx:end_img_idx]
                
                chunk_rgb = self._extract_colors_from_images(
                    chunk_points, 
                    chunk_images, 
                    self.camera_ids[i]
                )
            else:
                # Use chunk color
                chunk_rgb = torch.ones_like(chunk_points) * chunk_colors[i]
            
            # Create filename for this chunk
            chunk_filename = base_save_path.replace('.ply', f'_chunk_{i+1:02d}.ply')
            
            print(f"Saving chunk {i+1} to: {chunk_filename}")
            write_ply(chunk_points, chunk_rgb, chunk_filename, max_points=1000000)
        
        print(f"Saved {len(self.chunk_results)} individual chunks")
    
    def _save_camera_poses(self, result: Dict, base_save_path: str):
        """
        Save camera poses as colored points in red.
        
        Args:
            result: Result dictionary from process_sequence
            base_save_path: Base path for saving (will append '_camera_poses')
        """
        if 'camera_poses' not in result:
            print("No camera poses available for saving")
            return
        
        camera_poses = result['camera_poses']
        camera_ids = result.get('camera_ids', [])
        
        print("Saving camera poses as red points...")
        
        # Extract camera positions (translation part of camera-to-world transformation)
        camera_positions = []
        
        for i, pose in enumerate(camera_poses):
            pose_np = pose.numpy()
            camera_pos = pose_np[:3, 3]  # Translation part
            camera_positions.append(camera_pos)
        
        if camera_positions:
            camera_positions = np.array(camera_positions)
            # All camera poses in red
            camera_colors = np.full((len(camera_positions), 3), [1.0, 0.0, 0.0])  # Red color
            
            # Create filename for camera poses
            camera_filename = base_save_path.replace('.ply', '_camera_poses.ply')
            
            print(f"Saving {len(camera_positions)} camera poses to: {camera_filename}")
            write_ply(torch.from_numpy(camera_positions), torch.from_numpy(camera_colors), camera_filename)
        else:
            print("No camera positions to save")
    
    def _generate_chunk_colors(self, num_chunks: int) -> List[List[float]]:
        """
        Generate distinct colors for each chunk.
        
        Args:
            num_chunks: Number of chunks
        
        Returns:
            List of RGB colors for each chunk
        """
        import colorsys
        
        colors = []
        for i in range(num_chunks):
            # Generate colors using HSV color space for better distribution
            hue = i / num_chunks
            saturation = 0.8
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(list(rgb))
        
        return colors
    
    def _find_chunk_for_camera_id(self, camera_id: int) -> int:
        """
        Find which chunk a camera ID belongs to.
        
        Args:
            camera_id: Camera ID to find
        
        Returns:
            Chunk index (0-based)
        """
        for i, chunk_camera_ids in enumerate(self.camera_ids):
            if camera_id in chunk_camera_ids:
                return i
        return -1  # Not found
    
    def _extract_colors_from_images(self, points: torch.Tensor, images: torch.Tensor, camera_ids: List[int]) -> torch.Tensor:
        """
        Extract RGB colors from images for each point using chunk-based approach.
        
        Args:
            points: Point cloud tensor (N, 3) - on CPU
            images: Images tensor (N, 3, H, W) - on CPU
            camera_ids: List of camera IDs for each point
        
        Returns:
            RGB colors tensor (N, 3) - on CPU
        """
        H, W = images.shape[-2:]
        
        # Initialize RGB tensor
        rgb = torch.zeros_like(points)
        
        print(f"Color extraction: {len(points)} points, {len(images)} images, {H}x{W} resolution")
        print(f"Camera IDs: {camera_ids}")
        
        # For chunked processing, we need to map points to their original images
        # Each chunk has its own set of images, so we need to handle this properly
        
        # Calculate points per image in this chunk
        points_per_image = H * W
        num_images = len(images)
        
        points = points.reshape(-1, 3)
        # Check if the number of points matches what we expect
        expected_points = num_images * points_per_image
        if len(points) != expected_points:
            print(f"Warning: Expected {expected_points} points but got {len(points)}")
            print("Using fallback color extraction...")
            return self._extract_colors_fallback(points, images, camera_ids)
        
        # Map each point to its corresponding image and pixel
        for i in range(len(points)):
            # Determine which image this point belongs to
            img_idx = i // points_per_image
            if img_idx < num_images:
                # Get the pixel index within the image
                pixel_idx = i % points_per_image
                
                # Convert pixel index to (h, w) coordinates
                h = pixel_idx // W
                w = pixel_idx % W
                
                # Get the image
                img = images[img_idx]  # (3, H, W)
                
                # Get the color at this pixel
                if h < H and w < W:
                    rgb[i] = img[:, h, w]  # (3,) - RGB values
        
        # Normalize colors to [0, 1] range if they're not already
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        
        print(f"Color extraction complete: RGB range [{rgb.min():.3f}, {rgb.max():.3f}]")
        return rgb
    
    def _extract_colors_from_chunked_images(self, points: torch.Tensor, result: Dict) -> torch.Tensor:
        """
        Extract RGB colors from chunked images for merged point cloud.
        Uses local_points and masks to map sparse points to image pixels.
        
        Args:
            points: Merged point cloud tensor (N, 3)
            result: Result dictionary containing chunk_results
        
        Returns:
            RGB colors tensor (N, 3)
        """
        print("Extracting colors from chunked images using sparse point mapping...")
        
        # Initialize RGB tensor
        rgb = torch.zeros_like(points)
        
        # Track current point index
        current_point_idx = 0
        
        # Process each chunk
        for chunk_idx, chunk_result in enumerate(self.chunk_results):
            chunk_points = chunk_result['points']
            chunk_local_points = chunk_result['local_points']
            chunk_masks = chunk_result['masks']
            chunk_images = chunk_result.get('original_images', None)
            chunk_camera_ids = self.camera_ids[chunk_idx]
            
            if chunk_images is None:
                print(f"Warning: No original images for chunk {chunk_idx}, using fallback colors")
                chunk_rgb = self._extract_colors_fallback(chunk_points, torch.zeros(1, 3, 64, 64), chunk_camera_ids)
            else:
                # Extract colors using local_points and masks for proper pixel mapping
                chunk_rgb = self._extract_colors_from_sparse_points(
                    chunk_points, chunk_local_points, chunk_masks, chunk_images, chunk_camera_ids
                )
            
            # Copy colors to the appropriate section of the merged RGB tensor
            num_chunk_points = len(chunk_points)
            if current_point_idx + num_chunk_points <= len(rgb):
                rgb[current_point_idx:current_point_idx + num_chunk_points] = chunk_rgb
                current_point_idx += num_chunk_points
            else:
                print(f"Warning: Point index mismatch in chunk {chunk_idx}")
                break
        
        print(f"Chunked color extraction complete: RGB range [{rgb.min():.3f}, {rgb.max():.3f}]")
        return rgb
    
    def _extract_colors_from_sparse_points(self, points: torch.Tensor, local_points: torch.Tensor, 
                                         masks: torch.Tensor, images: torch.Tensor, 
                                         camera_ids: List[int]) -> torch.Tensor:
        """
        Extract RGB colors from images for sparse points using local_points and masks.
        
        Args:
            points: Sparse point cloud tensor (N, 3) - on CPU
            local_points: Local point coordinates (num_images, H, W, 3) - on CPU
            masks: Point masks (num_images, H, W) - on CPU
            images: Images tensor (num_images, 3, H, W) - on CPU
            camera_ids: List of camera IDs
        
        Returns:
            RGB colors tensor (N, 3) - on CPU
        """
        print(f"Extracting colors from sparse points: {len(points)} points, {len(images)} images")
        
        # Initialize RGB tensor
        rgb = torch.zeros_like(points)
        
        # Get image dimensions
        num_images, C, H, W = images.shape
        
        # For each point, find which image and pixel it corresponds to
        for point_idx in range(len(points)):
            point = points[point_idx]
            
            # Find the closest local point across all images
            min_dist = float('inf')
            best_img_idx = 0
            best_h = 0
            best_w = 0
            
            for img_idx in range(num_images):
                # Get local points and mask for this image
                img_local_points = local_points[img_idx]  # (H, W, 3)
                img_mask = masks[img_idx]  # (H, W)
                
                # Only consider valid points (where mask is True)
                valid_mask = img_mask.bool()
                
                if valid_mask.any():
                    # Get valid local points
                    valid_local_points = img_local_points[valid_mask]  # (num_valid, 3)
                    valid_positions = torch.nonzero(valid_mask)  # (num_valid, 2) - (h, w)
                    
                    # Find closest local point to our world point
                    distances = torch.norm(valid_local_points - point, dim=1)
                    min_dist_idx = torch.argmin(distances)
                    
                    if distances[min_dist_idx] < min_dist:
                        min_dist = distances[min_dist_idx]
                        best_img_idx = img_idx
                        best_h, best_w = valid_positions[min_dist_idx]
            
            # Get color from the best matching pixel
            if min_dist < float('inf'):
                img = images[best_img_idx]  # (3, H, W)
                rgb[point_idx] = img[:, best_h, best_w]
            else:
                # Fallback: use a default color
                rgb[point_idx] = torch.tensor([0.5, 0.5, 0.5])
        
        # Normalize colors to [0, 1] range if they're not already
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        
        print(f"Sparse color extraction complete: RGB range [{rgb.min():.3f}, {rgb.max():.3f}]")
        return rgb
    
    def _extract_colors_fallback(self, points: torch.Tensor, images: torch.Tensor, camera_ids: List[int]) -> torch.Tensor:
        """
        Fallback color extraction method when point count doesn't match expected.
        
        Args:
            points: Point cloud tensor (N, 3)
            images: Images tensor (N, 3, H, W)
            camera_ids: List of camera IDs for each point
        
        Returns:
            RGB colors tensor (N, 3)
        """
        print("Using fallback color extraction method")
        
        # Create a simple color gradient based on point index
        rgb = torch.zeros_like(points)
        
        for i in range(len(points)):
            # Create a color based on point index
            hue = (i / len(points)) % 1.0
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            rgb[i] = torch.tensor([r, g, b])
        
        return rgb 
    
    def _create_chunk_based_colors(self, points: torch.Tensor, camera_ids: List[int]) -> torch.Tensor:
        """
        Create colors based on chunk membership instead of image colors.
        
        Args:
            points: Point cloud tensor (N, 3) - on CPU
            camera_ids: List of camera IDs for each point
        
        Returns:
            RGB colors tensor (N, 3) - on CPU
        """
        print("Creating chunk-based colors...")
        
        rgb = torch.zeros_like(points)
        chunk_colors = self._generate_chunk_colors(len(self.chunk_results))
        
        # Calculate points per camera
        points_per_camera = len(points) // len(camera_ids) if camera_ids else len(points)
        
        for i, cam_id in enumerate(camera_ids):
            # Find which chunk this camera belongs to
            chunk_idx = self._find_chunk_for_camera_id(cam_id)
            
            # Get color for this chunk
            if chunk_idx < len(chunk_colors):
                color = torch.tensor(chunk_colors[chunk_idx], device=points.device)
            else:
                color = torch.tensor([1.0, 0.0, 0.0], device=points.device)  # Red as fallback
            
            # Assign this color to all points from this camera
            start_idx = i * points_per_camera
            end_idx = min((i + 1) * points_per_camera, len(points))
            rgb[start_idx:end_idx] = color
        
        return rgb 
    
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
                from pi3.utils.basic import TORCHCODEC_AVAILABLE
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
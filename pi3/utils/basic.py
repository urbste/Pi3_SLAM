import os
import os.path as osp
import math
import cv2
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import natsort
import psutil
import time
from typing import Optional, Tuple, List
import open3d as o3d

# Import undistortion utilities
from .undistortion import UndistortionMaps, create_undistortion_maps_from_file, VideoUndistortionLoader

# Try to import torchcodec for video processing
try:
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_AVAILABLE = True
except ImportError:
    TORCHCODEC_AVAILABLE = False
    print("Warning: torchcodec not available!")


def load_images_as_tensor(path='data/truck', interval=1, PIXEL_LIMIT=255000, 
                         undistortion_maps: Optional[UndistortionMaps] = None,
                         max_images: Optional[int] = None):
    """
    Loads images from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, 3, H, W] PyTorch tensor.
    
    Args:
        path: Path to image directory or video file
        interval: Sampling interval for frames
        PIXEL_LIMIT: Maximum number of pixels per image
        undistortion_maps: Optional UndistortionMaps object for applying undistortion
        max_images: Maximum number of images to load (None for no limit)
    """
    sources = [] 
    
    # --- 1. Load image paths or video frames ---
    if osp.isdir(path):
        print(f"Loading images from directory: {path}")
        filenames = natsort.natsorted([x for x in os.listdir(path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Calculate how many images to load based on max_images and interval
        if max_images is not None:
            # Calculate the maximum index to load
            max_index = min(len(filenames), max_images * interval)
            print(f"Limiting to {max_images} images (loading every {interval}th image)")
        else:
            max_index = len(filenames)
        
        for i in range(0, max_index, interval):
            if max_images is not None and len(sources) >= max_images:
                break
            img_path = osp.join(path, filenames[i])
            try:
                sources.append(Image.open(img_path).convert('RGB'))
            except Exception as e:
                print(f"Could not load image {filenames[i]}: {e}")
    elif path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print(f"Loading frames from video: {path}")
        
        # Use torchcodec if available, otherwise fall back to OpenCV
        if TORCHCODEC_AVAILABLE:
            try:
                # Use torchcodec for video loading
                decoder = VideoDecoder(path, device="cpu")
                total_frames = decoder.metadata.num_frames
                
                for frame_idx in range(0, total_frames, interval):
                    if max_images is not None and len(sources) >= max_images:
                        break
                    try:
                        # Load frame using torchcodec
                        frame_tensor = decoder[frame_idx]  # Shape: [C, H, W], uint8
                        
                        # Convert to PIL Image
                        frame_np = frame_tensor.permute(1, 2, 0).numpy()  # CHW to HWC
                        frame_pil = Image.fromarray(frame_np, mode='RGB')
                        sources.append(frame_pil)
                    except Exception as e:
                        print(f"Could not load frame {frame_idx}: {e}")
                        continue
                
                # Clean up decoder
                del decoder
                
            except Exception as e:
                print(f"Error using torchcodec, falling back to OpenCV: {e}")
                # Fall back to OpenCV
                cap = cv2.VideoCapture(path)
                if not cap.isOpened(): 
                    raise IOError(f"Cannot open video file: {path}")
                frame_idx = 0
                while True:
                    if max_images is not None and len(sources) >= max_images:
                        break
                    ret, frame = cap.read()
                    if not ret: break
                    if frame_idx % interval == 0:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        sources.append(Image.fromarray(rgb_frame))
                    frame_idx += 1
                cap.release()
        else:
            # Use OpenCV as fallback
            cap = cv2.VideoCapture(path)
            if not cap.isOpened(): 
                raise IOError(f"Cannot open video file: {path}")
            frame_idx = 0
            while True:
                if max_images is not None and len(sources) >= max_images:
                    break
                ret, frame = cap.read()
                if not ret: break
                if frame_idx % interval == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    sources.append(Image.fromarray(rgb_frame))
                frame_idx += 1
            cap.release()
    else:
        raise ValueError(f"Unsupported path. Must be a directory or a video file: {path}")

    if not sources:
        print("No images found or loaded.")
        return torch.empty(0)

    print(f"Found {len(sources)} images/frames. Processing...")

    # --- 2. Determine a uniform target size for all images based on the first image ---
    # This is necessary to ensure all tensors have the same dimensions for stacking.
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target: k -= 1
        else: m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    # --- 3. Resize images and convert them to tensors in the [0, 1] range ---
    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = transforms.ToTensor()
    
    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            
            # Apply undistortion if maps are provided
            if undistortion_maps is not None:
                # Convert PIL to numpy for undistortion
                img_np = np.array(resized_img)
                undistorted_img = undistortion_maps.undistort_image(img_np, (TARGET_H, TARGET_W))
                # Convert back to PIL
                resized_img = Image.fromarray(undistorted_img)
            
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    return torch.stack(tensor_list, dim=0)


def load_images_from_paths(
    image_paths: List[str],
    PIXEL_LIMIT: int = 255000
) -> torch.Tensor:
    """
    Load a list of image file paths into a single tensor [N, 3, H, W].

    - Supports PNG/JPG/JPEG and any Pillow-readable format
    - Converts grayscale images to RGB
    - Resizes all images to a uniform size (multiple of 14) based on the first image,
      keeping total pixels under PIXEL_LIMIT

    Args:
        image_paths: List of absolute or relative file paths to images
        PIXEL_LIMIT: Maximum number of pixels per image

    Returns:
        torch.Tensor of shape [N, 3, H, W]
    """
    if image_paths is None or len(image_paths) == 0:
        return torch.empty(0)

    sources: List[Image.Image] = []

    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            # Convert grayscale or other modes to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            sources.append(img)
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")

    if not sources:
        print("No valid images found in provided paths.")
        return torch.empty(0)

    # Determine uniform target size from the first image
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target:
            k -= 1
        else:
            m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14

    tensor_list: List[torch.Tensor] = []
    to_tensor_transform = transforms.ToTensor()

    for img_pil in sources:
        try:
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed from provided paths.")
        return torch.empty(0)

    return torch.stack(tensor_list, dim=0)


def load_images_as_tensor_with_undistortion(
    path='data/truck', 
    interval=1, 
    PIXEL_LIMIT=255000,
    cam_dist_path: Optional[str] = None,
    scale: float = 1.0
):
    """
    Loads images with undistortion applied using camera calibration files.
    
    Args:
        path: Path to image directory or video file
        interval: Sampling interval for frames
        PIXEL_LIMIT: Maximum number of pixels per image
        cam_dist_path: Path to distorted camera calibration JSON file
        scale: Scaling factor for camera parameters
    
    Returns:
        Tensor of undistorted images (N, 3, H, W)
    """

    if cam_dist_path is None:
        print("Warning: No camera calibration file provided, loading without undistortion")
        return load_images_as_tensor(path, interval, PIXEL_LIMIT)
    
    # Create undistortion maps
    print(f"Creating undistortion maps from calibration file...")
    undistortion_maps = create_undistortion_maps_from_file(cam_dist_path, scale)
    
    # Load images with undistortion
    return load_images_as_tensor(path, interval, PIXEL_LIMIT, undistortion_maps)


def load_images_as_tensor_with_undistortion_maps(
    path='data/truck', 
    interval=1, 
    PIXEL_LIMIT=255000,
    undistortion_maps: Optional[UndistortionMaps] = None,
    use_torchcodec: bool = True,
    device: str = "cpu"
):
    """
    Loads images with undistortion applied using pre-computed undistortion maps.
    
    Args:
        path: Path to image directory or video file
        interval: Sampling interval for frames
        PIXEL_LIMIT: Maximum number of pixels per image
        undistortion_maps: Pre-computed undistortion maps
        use_torchcodec: Whether to use torchcodec for video processing (if available)
        device: Device to load videos on ("cpu" or "cuda")
    
    Returns:
        Tensor of undistorted images (N, 3, H, W)
    """

    # For video files, use torchcodec if available and requested
    if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) and use_torchcodec:
        try:
            return load_video_as_tensor_with_undistortion_torchcodec(
                path, interval, PIXEL_LIMIT, undistortion_maps, device
            )
        except ImportError:
            print("Warning: torchcodec not available, falling back to OpenCV")
            return load_images_as_tensor(path, interval, PIXEL_LIMIT, undistortion_maps)
    
    return load_images_as_tensor(path, interval, PIXEL_LIMIT, undistortion_maps)


def load_video_as_tensor_with_undistortion_torchcodec(
    video_path: str,
    interval: int = 1,
    PIXEL_LIMIT: int = 255000,
    undistortion_maps: Optional[UndistortionMaps] = None,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Load video frames with undistortion using torchcodec for efficient processing.
    
    Args:
        video_path: Path to the video file
        interval: Sampling interval for frames
        PIXEL_LIMIT: Maximum number of pixels per image
        undistortion_maps: Pre-computed undistortion maps
        device: Device to load videos on ("cpu" or "cuda")
    
    Returns:
        Tensor of undistorted video frames (N, 3, H, W)
    """

    # Create video loader
    video_loader = VideoUndistortionLoader(undistortion_maps, device=device)
    
    try:
        # Get video metadata
        metadata = video_loader.get_video_metadata(video_path)
        total_frames = metadata.num_frames
        
        print(f"Loading video: {video_path}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {metadata.duration_seconds:.2f} seconds")
        print(f"FPS: {metadata.average_fps:.2f}")
        
        # Calculate frame indices
        frame_indices = list(range(0, total_frames, interval))
        
        if not frame_indices:
            print("No frames to load.")
            return torch.empty(0)
        
        print(f"Loading {len(frame_indices)} frames with interval {interval}")
        
        # Load frames with undistortion
        frames_tensor = video_loader.load_and_undistort_frames(
            video_path, frame_indices
        )
        
        # Calculate target size based on first frame
        if frames_tensor.shape[0] > 0:
            H_orig, W_orig = frames_tensor.shape[2], frames_tensor.shape[3]
            scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
            W_target, H_target = W_orig * scale, H_orig * scale
            k, m = round(W_target / 14), round(H_target / 14)
            while (k * 14) * (m * 14) > PIXEL_LIMIT:
                if k / m > W_target / H_target: k -= 1
                else: m -= 1
            TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
            
            print(f"Resizing frames to ({TARGET_W}, {TARGET_H})")
            
            # Resize frames if needed
            if (TARGET_W, TARGET_H) != (W_orig, H_orig):
                frames_tensor = torch.nn.functional.interpolate(
                    frames_tensor, size=(TARGET_H, TARGET_W), mode='bilinear', align_corners=False
                )
        
        print(f"✅ Successfully loaded {frames_tensor.shape[0]} undistorted video frames")
        print(f"   Frame tensor shape: {frames_tensor.shape}")
        print(f"   Value range: [{frames_tensor.min():.3f}, {frames_tensor.max():.3f}]")
        
        return frames_tensor
        
    finally:
        # Clean up video loader
        video_loader.close_decoder(video_path)


def tensor_to_pil(tensor):
    """
    Converts a PyTorch tensor to a PIL image. Automatically moves the channel dimension 
    (if it has size 3) to the last axis before converting.

    Args:
        tensor (torch.Tensor): Input tensor. Expected shape can be [C, H, W], [H, W, C], or [H, W].
    
    Returns:
        PIL.Image: The converted PIL image.
    """
    if torch.is_tensor(tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = tensor

    return array_to_pil(array)


def array_to_pil(array):
    """
    Converts a NumPy array to a PIL image. Automatically:
        - Squeezes dimensions of size 1.
        - Moves the channel dimension (if it has size 3) to the last axis.
    
    Args:
        array (np.ndarray): Input array. Expected shape can be [C, H, W], [H, W, C], or [H, W].
    
    Returns:
        PIL.Image: The converted PIL image.
    """
    # Remove singleton dimensions
    array = np.squeeze(array)
    
    # Ensure the array has the channel dimension as the last axis
    if array.ndim == 3 and array.shape[0] == 3:  # If the channel is the first axis
        array = np.transpose(array, (1, 2, 0))  # Move channel to the last axis
    
    # Handle single-channel grayscale images
    if array.ndim == 2:  # [H, W]
        return Image.fromarray((array * 255).astype(np.uint8), mode="L")
    elif array.ndim == 3 and array.shape[2] == 3:  # [H, W, C] with 3 channels
        return Image.fromarray((array * 255).astype(np.uint8), mode="RGB")
    else:
        raise ValueError(f"Unsupported array shape for PIL conversion: {array.shape}")


def rotate_target_dim_to_last_axis(x, target_dim=3):
    shape = x.shape
    axis_to_move = -1
    # Iterate backwards to find the first occurrence from the end 
    # (which corresponds to the last dimension of size 3 in the original order).
    for i in range(len(shape) - 1, -1, -1):
        if shape[i] == target_dim:
            axis_to_move = i
            break

    # 2. If the axis is found and it's not already in the last position, move it.
    if axis_to_move != -1 and axis_to_move != len(shape) - 1:
        # Create the new dimension order.
        dims_order = list(range(len(shape)))
        dims_order.pop(axis_to_move)
        dims_order.append(axis_to_move)
        
        # Use permute to reorder the dimensions.
        ret = x.transpose(*dims_order)
    else:
        ret = x

    return ret


def write_ply(
    xyz,
    rgb=None,
    path='output.ply',
    max_points=None,
    normals=None,
    colors_from_coords=True,
) -> None:
    """
    Write point cloud data to a PLY file using Open3D.
    
    Args:
        xyz (torch.Tensor or np.ndarray): Point coordinates of shape (..., 3)
        rgb (torch.Tensor or np.ndarray, optional): RGB colors of shape (..., 3)
        path (str): Output file path
        max_points (int, optional): If set, randomly sample up to max_points
        normals (torch.Tensor or np.ndarray, optional): Normal vectors of shape (..., 3)
        colors_from_coords (bool): If True and rgb is None, generate colors from coordinates
    """
    # Convert to numpy arrays
    if torch.is_tensor(xyz):
        xyz = xyz.detach().cpu().numpy()
    if torch.is_tensor(rgb):
        rgb = rgb.detach().cpu().numpy()
    if torch.is_tensor(normals):
        normals = normals.detach().cpu().numpy()

    # Reshape to (N, 3)
    xyz = rotate_target_dim_to_last_axis(xyz, 3)
    xyz = xyz.reshape(-1, 3)
    
    if normals is not None:
        normals = rotate_target_dim_to_last_axis(normals, 3)
        normals = normals.reshape(-1, 3)

    # Handle RGB colors
    if rgb is not None:
        rgb = rotate_target_dim_to_last_axis(rgb, 3)
        rgb = rgb.reshape(-1, 3)
        
        # Normalize RGB values to [0, 1] if they're in [0, 255]
        if rgb.max() > 1:
            rgb = rgb / 255.0
    elif colors_from_coords:
        # Generate colors from coordinates using the same method as before
        min_coord = np.min(xyz, axis=0)
        max_coord = np.max(xyz, axis=0)
        normalized_coord = (xyz - min_coord) / (max_coord - min_coord + 1e-8)
        
        hue = 0.7 * normalized_coord[:,0] + 0.2 * normalized_coord[:,1] + 0.1 * normalized_coord[:,2]
        hsv = np.stack([hue, 0.9*np.ones_like(hue), 0.8*np.ones_like(hue)], axis=1)

        c = hsv[:,2:] * hsv[:,1:2]
        x = c * (1 - np.abs( (hsv[:,0:1]*6) % 2 - 1 ))
        m = hsv[:,2:] - c
        
        rgb = np.zeros_like(hsv)
        cond = (0 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 1)
        rgb[cond] = np.hstack([c[cond], x[cond], np.zeros_like(x[cond])])
        cond = (1 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 2)
        rgb[cond] = np.hstack([x[cond], c[cond], np.zeros_like(x[cond])])
        cond = (2 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 3)
        rgb[cond] = np.hstack([np.zeros_like(x[cond]), c[cond], x[cond]])
        cond = (3 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 4)
        rgb[cond] = np.hstack([np.zeros_like(x[cond]), x[cond], c[cond]])
        cond = (4 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 5)
        rgb[cond] = np.hstack([x[cond], np.zeros_like(x[cond]), c[cond]])
        cond = (5 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 6)
        rgb[cond] = np.hstack([c[cond], np.zeros_like(x[cond]), x[cond]])
        rgb = (rgb + m)

    # Random sampling if max_points is specified
    if max_points is not None and xyz.shape[0] > max_points:
        indices = np.random.choice(xyz.shape[0], max_points, replace=False)
        xyz = xyz[indices]
        if rgb is not None:
            rgb = rgb[indices]
        if normals is not None:
            normals = normals[indices]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    # Write to file
    o3d.io.write_point_cloud(path, pcd)


def read_ply(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Read point cloud data from a PLY file using Open3D.
    
    Args:
        path (str): Input file path
        
    Returns:
        Tuple of (xyz, rgb, normals) where:
        - xyz: Point coordinates of shape (N, 3)
        - rgb: RGB colors of shape (N, 3) or None
        - normals: Normal vectors of shape (N, 3) or None
    """
    pcd = o3d.io.read_point_cloud(path)
    
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors) if pcd.has_colors() else None
    normals = np.asarray(pcd.normals) if pcd.has_normals() else None
    
    return xyz, rgb, normals


def extract_colors_from_images(imgs: torch.Tensor, points: torch.Tensor, patch_size: int = 14) -> torch.Tensor:
    """
    Extract colors from input images and map them to 3D points.
    
    Args:
        imgs: Input images tensor of shape (B, N, 3, H, W)
        points: 3D points tensor of shape (B, N, H, W, 3)
        patch_size: Patch size used by the model (default: 14)
    
    Returns:
        colors: RGB colors tensor of shape (B*N*H*W, 3)
    """
    B, N, C, H, W = imgs.shape
    Bh, Nh, Hh, Wh, _ = points.shape
    
    # Reshape images to (B*N, 3, H, W)
    imgs_flat = imgs.reshape(B*N, C, H, W)

    # Resize images to match the patch grid
    imgs_resized = torch.nn.functional.interpolate(
        imgs_flat, 
        size=(Hh, Wh), 
        mode='bilinear', 
        align_corners=False
    )
    
    # Reshape to (B*N, 3, Wh*Wh)
    colors = imgs_resized.reshape(B*N, 3, -1).permute(0, 2, 1)  # (B*N, Wh*Wh, 3)
    
    # Flatten to (B*N*Wh*Wh, 3)
    colors = colors.reshape(-1, 3)
    
    return colors


def filter_points_by_confidence(points: torch.Tensor, confidence: torch.Tensor, 
                               threshold: float = 0.7) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter points by confidence threshold.
    
    Args:
        points: 3D points tensor of shape (..., 3)
        confidence: Confidence tensor of shape (...)
        threshold: Confidence threshold (default: 0.7)
    
    Returns:
        Tuple of (filtered_points, filtered_confidence, mask):
        - filtered_points: Points with confidence > threshold
        - filtered_confidence: Confidence values for filtered points
        - mask: Boolean mask indicating which points were kept
    """
    # Flatten tensors
    points_flat = points.reshape(-1, 3)
    conf_flat = confidence.reshape(-1)
    
    # Create confidence mask
    mask = conf_flat > threshold
    
    # Apply mask
    filtered_points = points_flat[mask]
    filtered_confidence = conf_flat[mask]
    
    return filtered_points, filtered_confidence, mask

def create_camera_trajectory(camera_poses: torch.Tensor, 
                           trajectory_scale: float = 0.1,
                           show_orientations: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create camera trajectory points and colors for visualization.
    
    Args:
        camera_poses: Camera pose matrices of shape (B, N, 4, 4)
        trajectory_scale: Scale factor for camera orientation arrows
        show_orientations: Whether to show camera orientations as arrows
    
    Returns:
        Tuple of (trajectory_points, trajectory_colors):
        - trajectory_points: 3D points for camera positions and orientations
        - trajectory_colors: RGB colors for trajectory visualization
    """
    B, N, _, _ = camera_poses.shape
    
    # Extract camera positions (translation part)
    camera_positions = camera_poses[:, :, :3, 3]  # Shape: (B, N, 3)
    camera_positions = camera_positions.reshape(-1, 3)  # Shape: (B*N, 3)
    
    # Create camera position colors (red for visibility)
    camera_colors = torch.zeros_like(camera_positions)
    camera_colors[:, 0] = 1.0  # Red
    
    trajectory_points = [camera_positions]
    trajectory_colors = [camera_colors]

    # Combine all trajectory elements
    all_trajectory_points = torch.cat(trajectory_points, dim=0)
    all_trajectory_colors = torch.cat(trajectory_colors, dim=0)
    
    return all_trajectory_points, all_trajectory_colors

def random_sample_points(points: torch.Tensor, colors: torch.Tensor, 
                        max_points_per_image: int = 1000, 
                        random_seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly sample a fixed number of points per image to reduce point cloud size.
    
    Args:
        points: 3D points tensor of shape (B, N, H, W, 3)
        colors: RGB colors tensor of shape (B*N*H*W, 3)
        max_points_per_image: Maximum number of points to sample per image
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (sampled_points, sampled_colors, sample_mask):
        - sampled_points: Randomly sampled points
        - sampled_colors: Colors for sampled points
        - sample_mask: Boolean mask indicating which points were sampled
    """
    import numpy as np
    
    B, N, H, W, _ = points.shape
    
    # Flatten points to (B*N*H*W, 3)
    points_flat = points.reshape(-1, 3)
    
    # Calculate points per image
    points_per_image = H * W
    total_points = B * N * points_per_image
    
    # Determine how many points to sample per image
    points_to_sample = min(max_points_per_image, points_per_image)
    total_samples = B * N * points_to_sample
    
    print(f"Random sampling: {points_to_sample} points per image from {points_per_image} available")
    print(f"Total reduction: {total_points} → {total_samples} ({total_samples/total_points*100:.1f}%)")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Create sampling indices
    sample_indices = []
    for i in range(B * N):
        # Get indices for this image
        start_idx = i * points_per_image
        
        # Randomly sample indices for this image
        image_indices = np.random.choice(
            points_per_image, 
            size=points_to_sample, 
            replace=False
        )
        
        # Convert to global indices
        global_indices = start_idx + image_indices
        sample_indices.extend(global_indices)
    
    sample_indices = np.array(sample_indices)
    
    # Sample points and colors
    sampled_points = points_flat[sample_indices]
    sampled_colors = colors[sample_indices]
    
    # Create mask for sampled points (on same device as input points)
    sample_mask = torch.zeros(total_points, dtype=torch.bool, device=points.device)
    sample_mask[sample_indices] = True
    
    return sampled_points, sampled_colors, sample_mask


def get_memory_usage():
    """Get current memory usage in MB."""
    if torch.cuda.is_available():
        # GPU memory
        gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**2  # MB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        return {
            'gpu_allocated': gpu_memory,
            'gpu_max_allocated': gpu_memory_max,
            'gpu_reserved': gpu_memory_reserved,
            'gpu_available': (torch.cuda.get_device_properties(0).total_memory / 1024**2) - gpu_memory_reserved
        }
    else:
        # CPU memory
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024**2  # MB
        return {
            'cpu_memory': cpu_memory,
            'cpu_available': psutil.virtual_memory().available / 1024**2
        }


def print_memory_usage(stage_name: str = ""):
    """Print current memory usage."""
    memory = get_memory_usage()
    
    if torch.cuda.is_available():
        print(f"🔧 Memory Usage {stage_name}:")
        print(f"   GPU Allocated: {memory['gpu_allocated']:.1f} MB")
        print(f"   GPU Reserved: {memory['gpu_reserved']:.1f} MB")
        print(f"   GPU Available: {memory['gpu_available']:.1f} MB")
        print(f"   GPU Max Used: {memory['gpu_max_allocated']:.1f} MB")
    else:
        print(f"🔧 Memory Usage {stage_name}:")
        print(f"   CPU Memory: {memory['cpu_memory']:.1f} MB")
        print(f"   CPU Available: {memory['cpu_available']:.1f} MB")


class MemoryProfiler:
    """Context manager for profiling memory usage."""
    
    def __init__(self, stage_name: str = ""):
        self.stage_name = stage_name
        self.start_memory = None
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = get_memory_usage()
        print(f"🚀 Starting {self.stage_name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_memory = get_memory_usage()
        
        duration = end_time - self.start_time
        
        print(f"✅ Completed {self.stage_name} in {duration:.2f}s")
        
        if torch.cuda.is_available():
            memory_diff = end_memory['gpu_allocated'] - self.start_memory['gpu_allocated']
            print(f"   Memory change: {memory_diff:+.1f} MB")
            print(f"   Peak memory: {end_memory['gpu_max_allocated']:.1f} MB")
        else:
            memory_diff = end_memory['cpu_memory'] - self.start_memory['cpu_memory']
            print(f"   Memory change: {memory_diff:+.1f} MB")


def load_video_frame_torchcodec(video_path: str, frame_idx: int, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Load a single video frame using torchcodec.
    
    Args:
        video_path: Path to the video file
        frame_idx: Frame index to load
        target_size: Optional target size (height, width) for resizing
    
    Returns:
        Tensor of the frame (3, H, W) in [0, 1] range
    """
    if not TORCHCODEC_AVAILABLE:
        raise ImportError("torchcodec not available. Install with: pip install torchcodec")
    
    try:
        # Load frame using torchcodec
        decoder = VideoDecoder(video_path, device="cpu")
        frame_tensor = decoder[frame_idx]  # Shape: [C, H, W], uint8
        
        # Convert to float [0, 1] range
        frame_tensor = frame_tensor.float() / 255.0
        
        # Resize if target size is specified
        if target_size is not None:
            frame_tensor = torch.nn.functional.interpolate(
                frame_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False
            ).squeeze(0)
        
        # Clean up decoder
        del decoder
        
        return frame_tensor
        
    except Exception as e:
        raise RuntimeError(f"Error loading frame {frame_idx} from {video_path}: {e}")


def load_video_frame_opencv(video_path: str, frame_idx: int, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Load a single video frame using OpenCV (fallback method).
    
    Args:
        video_path: Path to the video file
        frame_idx: Frame index to load
        target_size: Optional target size (height, width) for resizing
    
    Returns:
        Tensor of the frame (3, H, W) in [0, 1] range
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize if target size is specified
    if target_size is not None:
        frame_rgb = cv2.resize(frame_rgb, (target_size[1], target_size[0]))  # (width, height)
    
    # Convert to tensor
    frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
    frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC to CHW
    
    return frame_tensor


def load_video_frame(video_path: str, frame_idx: int, target_size: Optional[Tuple[int, int]] = None, 
                    use_torchcodec: bool = True) -> torch.Tensor:
    """
    Load a single video frame using torchcodec if available, otherwise OpenCV.
    
    Args:
        video_path: Path to the video file
        frame_idx: Frame index to load
        target_size: Optional target size (height, width) for resizing
        use_torchcodec: Whether to use torchcodec (if available)
    
    Returns:
        Tensor of the frame (3, H, W) in [0, 1] range
    """
    if use_torchcodec and TORCHCODEC_AVAILABLE:
        try:
            return load_video_frame_torchcodec(video_path, frame_idx, target_size)
        except Exception as e:
            print(f"Warning: torchcodec failed, falling back to OpenCV: {e}")
            return load_video_frame_opencv(video_path, frame_idx, target_size)
    else:
        return load_video_frame_opencv(video_path, frame_idx, target_size)


def get_video_frame_count(video_path: str, use_torchcodec: bool = True) -> int:
    """
    Get the total number of frames in a video file.
    
    Args:
        video_path: Path to the video file
        use_torchcodec: Whether to use torchcodec (if available)
    
    Returns:
        Total number of frames
    """
    if use_torchcodec and TORCHCODEC_AVAILABLE:
        try:
            decoder = VideoDecoder(video_path, device="cpu")
            frame_count = decoder.metadata.num_frames
            del decoder
            return frame_count
        except Exception as e:
            print(f"Warning: torchcodec failed, falling back to OpenCV: {e}")
    
    # Fall back to OpenCV
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count
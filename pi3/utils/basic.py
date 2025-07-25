import os
import os.path as osp
import math
import cv2
from PIL import Image
import torch
from torchvision import transforms
from plyfile import PlyData, PlyElement
import numpy as np
import natsort
from typing import Optional, Tuple, Dict, Any

# Import undistortion utilities
from .undistortion import UndistortionMaps, create_undistortion_maps_from_file, create_undistortion_maps_from_files, UndistortionImageLoader, VideoUndistortionLoader

# Try to import torchcodec for video processing
try:
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_AVAILABLE = True
except ImportError:
    TORCHCODEC_AVAILABLE = False
    print("Warning: torchcodec not available!")


def load_images_as_tensor(path='data/truck', interval=1, PIXEL_LIMIT=255000, 
                         undistortion_maps: Optional[UndistortionMaps] = None):
    """
    Loads images from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, 3, H, W] PyTorch tensor.
    
    Args:
        path: Path to image directory or video file
        interval: Sampling interval for frames
        PIXEL_LIMIT: Maximum number of pixels per image
        undistortion_maps: Optional UndistortionMaps object for applying undistortion
    """
    sources = [] 
    
    # --- 1. Load image paths or video frames ---
    if osp.isdir(path):
        print(f"Loading images from directory: {path}")
        filenames = natsort.natsorted([x for x in os.listdir(path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for i in range(0, len(filenames), interval):
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
    max_points=None,  # New parameter
) -> None:
    """
    Write point cloud data to a PLY file.
    
    Args:
        xyz (torch.Tensor or np.ndarray): Point coordinates of shape (..., 3)
        rgb (torch.Tensor or np.ndarray, optional): RGB colors of shape (..., 3)
        path (str): Output file path
        max_points (int, optional): If set, randomly sample up to max_points
    """
    if torch.is_tensor(xyz):
        xyz = xyz.detach().cpu().numpy()

    if torch.is_tensor(rgb):
        rgb = rgb.detach().cpu().numpy()

    if rgb is not None and rgb.max() > 1:
        rgb = rgb / 255.

    xyz = rotate_target_dim_to_last_axis(xyz, 3)
    xyz = xyz.reshape(-1, 3)

    if rgb is not None:
        rgb = rotate_target_dim_to_last_axis(rgb, 3)
        rgb = rgb.reshape(-1, 3)

    # Add random sampling if max_points is specified
    if max_points is not None and xyz.shape[0] > max_points:
        indices = np.random.choice(xyz.shape[0], max_points, replace=False)
        xyz = xyz[indices]
        if rgb is not None:
            rgb = rgb[indices]
    
    if rgb is None:
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

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb * 255), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


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
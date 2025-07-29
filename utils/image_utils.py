"""
Image loading and processing utilities for Pi3SLAM.
"""

import torch
import numpy as np
import math
from typing import List, Tuple
from PIL import Image
import torchvision.transforms as transforms


def calculate_target_size(first_image_path, pixel_limit: int = 255000) -> Tuple[int, int]:
    """
    Calculate target size based on the first image.
    
    Args:
        first_image_path: Path to the first image or (video_path, frame_idx) tuple
        pixel_limit: Maximum number of pixels to allow
    
    Returns:
        Tuple of (height, width) for target size
    """
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
    scale = math.sqrt(pixel_limit / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > W_target / H_target: k -= 1
        else: m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    
    return (TARGET_H, TARGET_W)  # Return as (height, width)


def load_images_from_paths(image_paths: List, target_size: Tuple[int, int], 
                          undistortion_maps=None) -> torch.Tensor:
    """
    Load images from file paths and resize to uniform target size.
    
    Args:
        image_paths: List of image paths or (video_path, frame_idx) tuples
        target_size: Target size (height, width) for resizing
        undistortion_maps: Optional UndistortionMaps object for applying undistortion
    
    Returns:
        Tensor of images (N, 3, H, W) on device
    """
    images = []
    
    for path_info in image_paths:
        if isinstance(path_info, tuple):
            # Video frame
            video_path, frame_idx = path_info
            from pi3.utils.basic import load_video_frame
            
            # Load frame using torchcodec (with fallback to OpenCV)
            image = load_video_frame(video_path, frame_idx, target_size, use_torchcodec=True)
            
            # Apply undistortion if maps are provided
            if undistortion_maps is not None:
                # Convert tensor to numpy for undistortion
                image_np = image.permute(1, 2, 0).numpy()  # CHW to HWC
                undistorted_np = undistortion_maps.undistort_image(image_np, target_size)
                # Convert back to tensor
                image = torch.from_numpy(undistorted_np).float() / 255.0
                image = image.permute(2, 0, 1)  # HWC to CHW
        else:
            # Use PIL for image loading
            pil_image = Image.open(path_info).convert('RGB')
            
            # Apply undistortion if maps are provided
            if undistortion_maps is not None:
                # Convert PIL to numpy for undistortion
                image_np = np.array(pil_image)
                undistorted_np = undistortion_maps.undistort_image(image_np, target_size)
                # Convert back to tensor
                image = torch.from_numpy(undistorted_np).float() / 255.0
                image = image.permute(2, 0, 1)  # HWC to CHW
            else:
                # Use PIL transforms for regular loading
                transform = transforms.Compose([
                    transforms.Resize(target_size),  # (height, width)
                    transforms.ToTensor()
                ])
                image = transform(pil_image)
        images.append(image)
    
    # Stack images
    images_tensor = torch.stack(images)
    return images_tensor


def load_single_image(image_path: str, target_size: Tuple[int, int], device: str = 'cpu') -> torch.Tensor:
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to the image file
        target_size: Target size (width, height)
        device: Device to load tensor on
    
    Returns:
        Preprocessed image tensor (C, H, W)
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize image
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
        
        # Move to device
        image_tensor = image_tensor.to(device)
        
        return image_tensor
        
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        raise


def extract_colors_from_chunk(chunk_result: dict) -> torch.Tensor:
    """Extract colors from chunk result."""
    print(chunk_result['images'].shape)
    return chunk_result['images'].permute(0, 2, 3, 1).reshape(-1, 3).cpu()


def create_fallback_colors(num_points: int) -> torch.Tensor:
    """Create fallback colors when image extraction fails."""
    colors = torch.zeros(num_points, 3)
    for i in range(num_points):
        hue = (i / num_points) % 1.0
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors[i] = torch.tensor([r, g, b])
    return colors 
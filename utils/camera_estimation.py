"""
Camera parameter estimation utilities for Pi3SLAM.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional

from utils.geometry_torch import recover_focal_shift


def estimate_camera_parameters(
    result: Dict[str, torch.Tensor], 
    downsample_size: Tuple[int, int] = (64, 64)
) -> Dict[str, torch.Tensor]:
    """
    Estimate camera parameters from Pi3 model output.
    
    Args:
        result: Dictionary containing model output with keys:
            - 'local_points': torch.Tensor of shape (B, N, H, W, 3) - local 3D points
            - 'conf': torch.Tensor of shape (B, N, H, W, 1) - confidence scores
        downsample_size: Tuple of (height, width) for downsampling during estimation
    
    Returns:
        Dictionary containing estimated camera parameters:
            - 'intrinsics': torch.Tensor of shape (B, N, 3, 3) - camera intrinsics matrix
            - 'focal': torch.Tensor of shape (B, N) - focal length
            - 'shift': torch.Tensor of shape (B, N) - Z-axis shift
            - 'fx': torch.Tensor of shape (B, N) - focal length in x direction
            - 'fy': torch.Tensor of shape (B, N) - focal length in y direction
            - 'cx': torch.Tensor of shape (B, N) - principal point x coordinate
            - 'cy': torch.Tensor of shape (B, N) - principal point y coordinate
    """
    # Extract points and confidence
    points = result["local_points"]  # Shape: (B, N, H, W, 3)
    masks = torch.sigmoid(result["conf"][..., 0]) > 0.1  # Shape: (B, N, H, W)
    
    # Get original dimensions
    original_height, original_width = points.shape[-3:-1]
    aspect_ratio = original_width / original_height
    
    # Use recover_focal_shift function from MoGe
    focal, shift = recover_focal_shift(points, masks, downsample_size=downsample_size)
    
    # Calculate fx, fy from focal
    fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio * original_width
    fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 * original_height
    
    # Principal point at image center
    cx = torch.full_like(fx, original_width // 2)
    cy = torch.full_like(fy, original_height // 2)
    
    # Create intrinsics matrix using utils3d
    try:
        import utils3d
        intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, cx, cy)
    except ImportError:
        # Fallback if utils3d is not available
        intrinsics = _create_intrinsics_matrix(fx, fy, cx, cy)
    
    return {
        'intrinsics': intrinsics,
        'focal': focal,
        'shift': shift,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy
    }


def _create_intrinsics_matrix(fx: torch.Tensor, fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
    """
    Create camera intrinsics matrix from focal lengths and principal point.
    
    Args:
        fx: Focal length in x direction, shape (B, N)
        fy: Focal length in y direction, shape (B, N)
        cx: Principal point x coordinate, shape (B, N)
        cy: Principal point y coordinate, shape (B, N)
    
    Returns:
        Camera intrinsics matrix of shape (B, N, 3, 3)
    """
    batch_size, num_frames = fx.shape
    
    # Create identity matrix and expand to batch size
    intrinsics = torch.eye(3, device=fx.device, dtype=fx.dtype)
    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0).expand(batch_size, num_frames, 3, 3)
    
    # Fill in the camera parameters
    intrinsics[..., 0, 0] = fx  # fx
    intrinsics[..., 1, 1] = fy  # fy
    intrinsics[..., 0, 2] = cx  # cx
    intrinsics[..., 1, 2] = cy  # cy
    
    return intrinsics


def estimate_camera_parameters_single_chunk(
    chunk_result: Dict[str, torch.Tensor],
    downsample_size: Tuple[int, int] = (64, 64)
) -> Dict[str, torch.Tensor]:
    """
    Estimate camera parameters for a single chunk.
    
    Args:
        chunk_result: Dictionary containing chunk data with keys:
            - 'local_points': torch.Tensor of shape (N, H, W, 3) - local 3D points
            - 'conf': torch.Tensor of shape (N, H, W, 1) - confidence scores
        downsample_size: Tuple of (height, width) for downsampling during estimation
    
    Returns:
        Dictionary containing estimated camera parameters for the chunk
    """
    # Add batch dimension if not present
    if len(chunk_result['local_points'].shape) == 4:  # (N, H, W, 3)
        chunk_result = {
            'local_points': chunk_result['local_points'].unsqueeze(0),  # (1, N, H, W, 3)
            'conf': chunk_result['conf'].unsqueeze(0)  # (1, N, H, W, 1)
        }
    
    # Estimate parameters
    params = estimate_camera_parameters(chunk_result, downsample_size)
    
    # Remove batch dimension for single chunk
    for key, value in params.items():
        if len(value.shape) >= 3:  # Remove batch dimension
            params[key] = value.squeeze(0)
    
    return params


def validate_camera_parameters(
    intrinsics: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor
) -> bool:
    """
    Validate estimated camera parameters.
    
    Args:
        intrinsics: Camera intrinsics matrix of shape (..., 3, 3)
        fx: Focal length in x direction
        fy: Focal length in y direction
        cx: Principal point x coordinate
        cy: Principal point y coordinate
    
    Returns:
        True if parameters are valid, False otherwise
    """
    # Check for reasonable focal lengths (positive and not too large)
    if torch.any(fx <= 0) or torch.any(fy <= 0):
        return False
    
    # Check for reasonable principal points (within image bounds)
    # This would need image dimensions, but for now just check they're positive
    if torch.any(cx < 0) or torch.any(cy < 0):
        return False
    
    # Check that intrinsics matrix is well-formed
    if torch.any(torch.isnan(intrinsics)) or torch.any(torch.isinf(intrinsics)):
        return False
    
    return True


def print_camera_parameters_summary(params: Dict[str, torch.Tensor]) -> None:
    """
    Print a summary of estimated camera parameters.
    
    Args:
        params: Dictionary containing camera parameters from estimate_camera_parameters
    """
    print("📷 Camera Parameters Summary:")
    print(f"   Focal length range: {params['focal'].min().item():.3f} - {params['focal'].max().item():.3f}")
    print(f"   Fx range: {params['fx'].min().item():.1f} - {params['fx'].max().item():.1f}")
    print(f"   Fy range: {params['fy'].min().item():.1f} - {params['fy'].max().item():.1f}")
    print(f"   Principal point (cx, cy): ({params['cx'].mean().item():.1f}, {params['cy'].mean().item():.1f})")
    print(f"   Z-shift range: {params['shift'].min().item():.3f} - {params['shift'].max().item():.3f}") 
"""
Keypoint extraction utilities for Pi3SLAM.

This module provides keypoint extraction capabilities with two main approaches:

1. ALIKEDExtractor: Uses the ALIKED neural network for learned keypoint detection
2. GridKeypointExtractor: Creates a regular grid of keypoints across the image

Both extractors support color interpolation and can be used interchangeably in the SLAM pipeline.

Usage:
    # Create an ALIKED extractor
    extractor = create_keypoint_extractor('aliked', max_num_keypoints=512, detection_threshold=0.005)
    
    # Create a grid-based extractor (spacing calculated automatically)
    extractor = create_keypoint_extractor('grid', max_num_keypoints=512)
    
    # Disable keypoint extraction
    extractor = create_keypoint_extractor('none')
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
try:
    from lightglue import ALIKED
    ALIKED_AVAILABLE = True
except ImportError:
    ALIKED_AVAILABLE = False


class GridKeypointExtractor:
    """
    Grid-based keypoint extractor that creates a regular grid of keypoints.
    
    The grid spacing is automatically calculated based on the image dimensions and
    the desired number of keypoints to ensure optimal coverage.
    """
    
    def __init__(self, max_num_keypoints: int = 512, grid_spacing: int = None, device: str = 'cuda'):
        """
        Initialize grid-based keypoint extractor.
        
        Args:
            max_num_keypoints: Maximum number of keypoints to extract
            grid_spacing: Spacing between grid points in pixels (if None, calculated automatically from max_num_keypoints and image dimensions)
            device: Device to run the model on
        """
        self.device = device
        self.max_num_keypoints = max_num_keypoints
        self.grid_spacing = grid_spacing  # Will be calculated dynamically if None
    
    def _calculate_grid_spacing(self, H: int, W: int) -> int:
        """
        Calculate optimal grid spacing based on image dimensions and desired number of keypoints.
        
        Args:
            H: Image height
            W: Image width
            
        Returns:
            Grid spacing in pixels
        """
        if self.grid_spacing is not None:
            return self.grid_spacing
        
        # Calculate how many grid points we can fit in each dimension
        # We want approximately max_num_keypoints total points
        # For a grid, the number of points is roughly (H/spacing) * (W/spacing)
        # So spacing = sqrt((H * W) / max_num_keypoints)
        
        # Account for margins (we don't place keypoints at the very edges)
        margin = min(H, W) * 0.05  # 5% margin
        effective_H = H - 2 * margin
        effective_W = W - 2 * margin
        
        if effective_H <= 0 or effective_W <= 0:
            # If margins are too large, use center point only
            return max(H, W)
        
        # Calculate spacing to get approximately max_num_keypoints
        spacing = int(np.sqrt((effective_H * effective_W) / self.max_num_keypoints))
        
        # Ensure minimum spacing of 8 pixels and maximum spacing of min(H, W) / 4
        min_spacing = 8
        max_spacing = min(H, W) // 4
        
        spacing = max(min_spacing, min(spacing, max_spacing))
        
        return spacing
    
    def extract(self, images: torch.Tensor, frame_indices: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """
        Extract grid-based keypoints from images.
        
        Args:
            images: Tensor of shape (N, C, H, W) containing images
            frame_indices: Optional list of frame indices to extract from. If None, extracts from all frames.
        
        Returns:
            Dictionary containing:
                - 'keypoints': Tensor of shape (B, N, num_keypoints, 2) - keypoint coordinates
                - 'descriptors': Tensor of shape (B, N, num_keypoints, descriptor_dim) - dummy descriptors
                - 'scores': Tensor of shape (B, N, num_keypoints) - uniform scores
        """
        if len(images.shape) == 5:
            # Handle batch dimension
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)
        else:
            B, N = 1, images.shape[0]
            C, H, W = images.shape[1:]
        
        # Create grid of keypoints
        keypoints_list = []
        descriptors_list = []
        scores_list = []
        
        for i in range(images.shape[0]):
            # Calculate grid spacing for this image
            grid_spacing = self._calculate_grid_spacing(H, W)
            
            # Create grid coordinates with proper bounds checking
            margin = min(H, W) * 0.05  # 5% margin, consistent with _calculate_grid_spacing
            grid_x = torch.arange(margin, W - margin, grid_spacing, device=self.device)
            grid_y = torch.arange(margin, H - margin, grid_spacing, device=self.device)
            
            # Handle case where grid might be empty
            if len(grid_x) == 0 or len(grid_y) == 0:
                # Create a single keypoint at the center if grid is too small
                coords = torch.tensor([[W // 2, H // 2]], device=self.device, dtype=torch.float32)
            else:
                # Create meshgrid
                grid_y_coords, grid_x_coords = torch.meshgrid(grid_y, grid_x, indexing='ij')
                
                # Flatten and stack coordinates
                coords = torch.stack([grid_x_coords.flatten(), grid_y_coords.flatten()], dim=-1)
            
            # Limit to max_num_keypoints if needed
            if len(coords) > self.max_num_keypoints:
                # Randomly sample keypoints
                indices = torch.randperm(len(coords), device=self.device)[:self.max_num_keypoints]
                coords = coords[indices]
            
            keypoints_list.append(coords)
            
            # Create dummy descriptors (zeros)
            num_kpts = len(coords)
            dummy_descriptors = torch.zeros(num_kpts, 128, device=self.device)  # 128-dim descriptors like ALIKED
            descriptors_list.append(dummy_descriptors)
            
            # Create uniform scores
            uniform_scores = torch.ones(num_kpts, device=self.device)
            scores_list.append(uniform_scores)
        
        # Stack results
        keypoints = torch.stack(keypoints_list, dim=0)
        descriptors = torch.stack(descriptors_list, dim=0)
        scores = torch.stack(scores_list, dim=0)
        
        # Reshape if needed
        if len(images.shape) == 5:
            keypoints = keypoints.view(B, N, -1, 2)
            descriptors = descriptors.view(B, N, -1, 128)
            scores = scores.view(B, N, -1)
        
        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'scores': scores
        }
    
    def extract_colors(self, images: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Extract colors for keypoints using bilinear interpolation.
        
        Args:
            images: Tensor of shape (B, N, C, H, W) or (N, C, H, W) containing images
            keypoints: Tensor of shape (B, N, num_keypoints, 2) or (N, num_keypoints, 2) containing keypoint coordinates
        
        Returns:
            Tensor of shape (B, N, num_keypoints, 3) or (N, num_keypoints, 3) containing RGB colors
        """
        # Handle batch dimensions
        if len(images.shape) == 5:
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)
            keypoints = keypoints.view(B * N, -1, 2)
        else:
            B, N = 1, images.shape[0]
            C, H, W = images.shape[1:]
        
        # Extract colors for each frame
        with torch.no_grad():
            colors = self._interpolate_colors(images, keypoints)
       
        # Reshape if needed
        if len(images.shape) == 5:
            colors = colors.view(B, N, -1, 3)
        
        return colors
    
    def _interpolate_colors(self, image: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Interpolate colors for keypoints using bilinear sampling.
        
        Args:
            image: Tensor of shape (C, H, W) containing the image
            keypoints: Tensor of shape (num_keypoints, 2) containing keypoint coordinates
        
        Returns:
            Tensor of shape (num_keypoints, 3) containing RGB colors
        """
        H, W = image.shape[-2:]
        
        # Convert keypoint coordinates to grid coordinates [-1, 1]
        grid_x = (keypoints[:, :, 0] / (W - 1)) * 2 - 1  # Convert to [-1, 1]
        grid_y = (keypoints[:, :, 1] / (H - 1)) * 2 - 1  # Convert to [-1, 1]
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1)  # Shape: (1, 1, N, 2)
        
        # Interpolate colors using grid_sample
        colors = torch.nn.functional.grid_sample(
            image, grid.cpu(),  # Add batch dimension to image
            mode="bilinear", align_corners=False, padding_mode="border"
        )
        
        # Convert to RGB format and return
        colors = colors.squeeze().transpose(1, 2)  # Shape: (N, 3)
        return (colors * 255).detach().cpu().to(torch.uint8)
    
    def extract_with_colors(self, images: torch.Tensor, frame_indices: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """
        Extract keypoints and their colors in one call.
        
        Args:
            images: Tensor of shape (B, N, C, H, W) or (N, C, H, W) containing images
            frame_indices: Optional list of frame indices to extract from
        
        Returns:
            Dictionary containing keypoints, descriptors, scores, and colors
        """
        # Extract keypoints
        if len(images.shape) == 5:
            images = images.squeeze(0)

        keypoint_result = self.extract(images, frame_indices)
        
        # Extract colors
        colors = self.extract_colors(images, keypoint_result['keypoints'])
        
        # Combine results
        result = keypoint_result.copy()
        result['colors'] = colors
        
        return result


class ALIKEDExtractor:
    """
    ALIKED keypoint extractor with color interpolation capabilities.
    """
    
    def __init__(self, max_num_keypoints: int = 512, detection_threshold: float = 0.005, device: str = 'cuda'):
        """
        Initialize ALIKED keypoint extractor.
        
        Args:
            max_num_keypoints: Maximum number of keypoints to extract
            detection_threshold: Detection threshold for keypoints
            device: Device to run the model on
        """
        if not ALIKED_AVAILABLE:
            raise ImportError("ALIKED not available. Please install lightglue: pip install lightglue")
        
        self.device = device
        self.extractor = ALIKED(
            max_num_keypoints=max_num_keypoints, 
            detection_threshold=detection_threshold
        ).to(device).eval()
    
    def extract(self, images: torch.Tensor, frame_indices: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """
        Extract keypoints from images.
        
        Args:
            images: Tensor of shape (N, C, H, W) containing images
            frame_indices: Optional list of frame indices to extract from. If None, extracts from all frames.
        
        Returns:
            Dictionary containing:
                - 'keypoints': Tensor of shape (B, N, num_keypoints, 2) - keypoint coordinates
                - 'descriptors': Tensor of shape (B, N, num_keypoints, descriptor_dim) - keypoint descriptors
                - 'scores': Tensor of shape (B, N, num_keypoints) - keypoint scores
        """

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            results = self.extractor({"image": images.to(self.device)})
        
        keypoints = results["keypoints"]
        descriptors = results["descriptors"]
        scores = results["keypoint_scores"]

        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'scores': scores
        }
    
    def extract_colors(self, images: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Extract colors for keypoints using bilinear interpolation.
        
        Args:
            images: Tensor of shape (B, N, C, H, W) or (N, C, H, W) containing images
            keypoints: Tensor of shape (B, N, num_keypoints, 2) or (N, num_keypoints, 2) containing keypoint coordinates
        
        Returns:
            Tensor of shape (B, N, num_keypoints, 3) or (N, num_keypoints, 3) containing RGB colors
        """
        # Handle batch dimensions
        if len(images.shape) == 5:
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)
            keypoints = keypoints.view(B * N, -1, 2)
        else:
            B, N = 1, images.shape[0]
            C, H, W = images.shape[1:]
        
        # Extract colors for each frame
        with torch.no_grad():
            colors = self._interpolate_colors(images, keypoints)
        
        # Reshape if needed
        if len(images.shape) == 5:
            colors = colors.view(B, N, -1, 3)
        
        return colors
    
    def _interpolate_colors(self, image: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Interpolate colors for keypoints using bilinear sampling.
        
        Args:
            image: Tensor of shape (C, H, W) containing the image
            keypoints: Tensor of shape (num_keypoints, 2) containing keypoint coordinates
        
        Returns:
            Tensor of shape (num_keypoints, 3) containing RGB colors
        """
        H, W = image.shape[-2:]
        
        # Convert keypoint coordinates to grid coordinates [-1, 1]
        grid_x = (keypoints[:, :, 0] / (W - 1)) * 2 - 1  # Convert to [-1, 1]
        grid_y = (keypoints[:, :, 1] / (H - 1)) * 2 - 1  # Convert to [-1, 1]
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1)  # Shape: (1, 1, N, 2)
        
        # Interpolate colors using grid_sample
        colors = torch.nn.functional.grid_sample(
            image, grid.cpu(),  # Add batch dimension to image
            mode="bilinear", align_corners=False, padding_mode="border"
        )
        
        # Convert to RGB format and return
        colors = colors.squeeze().transpose(1, 2)  # Shape: (N, 3)
        return (colors * 255).detach().cpu().to(torch.uint8)
    
    def extract_with_colors(self, images: torch.Tensor, frame_indices: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """
        Extract keypoints and their colors in one call.
        
        Args:
            images: Tensor of shape (B, N, C, H, W) or (N, C, H, W) containing images
            frame_indices: Optional list of frame indices to extract from
        
        Returns:
            Dictionary containing keypoints, descriptors, scores, and colors
        """
        # Extract keypoints
        if len(images.shape) == 5:
            images = images.squeeze(0)

        keypoint_result = self.extract(images, frame_indices)
        
        # Extract colors
        colors = self.extract_colors(images, keypoint_result['keypoints'])
        
        # Combine results
        result = keypoint_result.copy()
        result['colors'] = colors
        
        return result


def create_keypoint_extractor(keypoint_type: str = 'aliked', **kwargs) -> Optional[object]:
    """
    Factory function to create a keypoint extractor based on type.
    
    Args:
        keypoint_type: Type of keypoint extractor ('aliked', 'grid', or 'none')
        **kwargs: Additional arguments for the extractor
            - For 'aliked': max_num_keypoints, detection_threshold, device
            - For 'grid': max_num_keypoints, device (grid_spacing is calculated automatically)
    
    Returns:
        Keypoint extractor instance or None if type is 'none'
    """
    if keypoint_type.lower() == 'aliked':
        if not ALIKED_AVAILABLE:
            print("⚠️  Warning: ALIKED not available, falling back to grid-based keypoints")
            return GridKeypointExtractor(**kwargs)
        return ALIKEDExtractor(**kwargs)
    elif keypoint_type.lower() == 'grid':
        # Remove grid_spacing from kwargs if present (it's calculated automatically now)
        kwargs.pop('grid_spacing', None)
        return GridKeypointExtractor(**kwargs)
    elif keypoint_type.lower() == 'none':
        return None
    else:
        raise ValueError(f"Unknown keypoint type: {keypoint_type}. Must be 'aliked', 'grid', or 'none'") 
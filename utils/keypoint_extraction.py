"""
Keypoint extraction utilities for Pi3SLAM.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from lightglue import ALIKED


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

        with torch.no_grad():
            results = self.extractor({"image": images.to(self.device)})
        
        keypoints = results["keypoints"]
        descriptors = results["descriptors"]
        scores = results["keypoint_scores"]

        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'scores': scores
        }
    
    def extract_colors(self, images: torch.Tensor, keypoints: torch.Tensor) -> np.ndarray:
        """
        Extract colors for keypoints using bilinear interpolation.
        
        Args:
            images: Tensor of shape (B, N, C, H, W) or (N, C, H, W) containing images
            keypoints: Tensor of shape (B, N, num_keypoints, 2) or (N, num_keypoints, 2) containing keypoint coordinates
        
        Returns:
            Array of shape (B, N, num_keypoints, 3) or (N, num_keypoints, 3) containing RGB colors
        """
        # Extract colors for each frame
        with torch.no_grad():
            colors = self._interpolate_colors(images, keypoints)

        return colors
    
    def _interpolate_colors(self, image: torch.Tensor, keypoints: torch.Tensor) -> np.ndarray:
        """
        Interpolate colors for keypoints using bilinear sampling.
        
        Args:
            image: Tensor of shape (C, H, W) containing the image
            keypoints: Tensor of shape (num_keypoints, 2) containing keypoint coordinates
        
        Returns:
            Array of shape (num_keypoints, 3) containing RGB colors
        """
        H, W = image.shape[-2:]
        
        # Convert keypoint coordinates to grid coordinates [-1, 1]
        grid_x = (keypoints[:, :, 0] / (W - 1)) * 2 - 1  # Convert to [-1, 1]
        grid_y = (keypoints[:, :, 1] / (H - 1)) * 2 - 1  # Convert to [-1, 1]
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1).cpu()  # Shape: (1, 1, N, 2)
        
        # Interpolate colors using grid_sample
        colors = torch.nn.functional.grid_sample(
            image, grid,  # Add batch dimension to image
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
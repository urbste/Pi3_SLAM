"""
Correspondence finding utilities for Pi3SLAM alignment.
"""

import torch
import numpy as np
from typing import List, Tuple
import open3d as o3d


def find_corresponding_points(points1: torch.Tensor, points2: torch.Tensor, 
                            camera_ids1: List[int], camera_ids2: List[int],
                            conf1: torch.Tensor = None, conf2: torch.Tensor = None,
                            subsample_factor: int = 4,
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
    all_corr_conf1 = []
    all_corr_conf2 = []

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
            conf1_selected = cam_conf1[valid_pts]
            conf2_selected = cam_conf2[valid_pts]

            # Extend lists efficiently
            all_corr_points1.extend(points1_selected)
            all_corr_points2.extend(points2_selected)
            all_corr_conf1.extend(conf1_selected)
            all_corr_conf2.extend(conf2_selected)
    
    if not all_corr_points1:
        return np.array([]), np.array([]), np.array([])
    
    # Convert to numpy arrays efficiently
    all_points1 = np.array(all_corr_points1)
    all_points2 = np.array(all_corr_points2)
    all_conf1 = np.array(all_corr_conf1)
    all_conf2 = np.array(all_corr_conf2)
    
    # Ensure equal number of points
    min_total = min(len(all_points1), len(all_points2))
    if min_total > 0:
        all_points1 = all_points1[:min_total]
        all_points2 = all_points2[:min_total]
        all_conf1 = all_conf1[:min_total]
        all_conf2 = all_conf2[:min_total]
        
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
        # return als confidencce for all_points1
        return all_points1, all_points2, (all_conf1 + all_conf2) / 2
    else:
        return np.array([]), np.array([]), np.array([])
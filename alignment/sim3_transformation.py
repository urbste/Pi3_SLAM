"""
SIM3 transformation utilities for Pi3SLAM alignment.
"""

import numpy as np
from typing import Tuple
from utils.geometry_utils import rodrigues_to_rotation_matrix, rotation_matrix_to_rodrigues


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
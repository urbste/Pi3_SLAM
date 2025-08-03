"""
Geometry utility functions for Pi3SLAM.
"""

import numpy as np
import torch
from typing import Tuple


class SIM3Transformation:
    """SIM3 transformation class (scale, rotation, translation)."""
    
    def __init__(self, s: float = 1.0, R: np.ndarray = None, t: np.ndarray = None):
        """
        Initialize SIM3 transformation.
        
        Args:
            s: Scale factor
            R: Rotation matrix (3, 3)
            t: Translation vector (3,)
        """
        self.s = s
        self.R = R if R is not None else np.eye(3)
        self.t = t if t is not None else np.zeros(3)
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform 3D points using SIM3 transformation."""
        return self.s * (points @ self.R.T) + self.t
    
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
        matrix[:3, :3] = self.s * self.R
        matrix[:3, 3] = self.t
        return matrix


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


def homogenize_points(points: torch.Tensor) -> torch.Tensor:
    """
    Convert 3D points to homogeneous coordinates.
    
    Args:
        points: 3D points tensor (..., 3)
    
    Returns:
        Homogeneous points tensor (..., 4)
    """
    shape = points.shape[:-1]
    ones = torch.ones(*shape, 1, device=points.device, dtype=points.dtype)
    return torch.cat([points, ones], dim=-1)


def apply_transformation_to_points(points: torch.Tensor, transformation: np.ndarray) -> torch.Tensor:
    """Apply similarity transformation (SIM3) to 3D points."""
    points_homog = homogenize_points(points)
    transformation_tensor = torch.from_numpy(transformation).to(torch.float32)
    transformed_points_homog = (transformation_tensor @ points_homog.view(-1, 4).T).T
    transformed_points_homog = transformed_points_homog.view(points.shape[0], points.shape[1], 4)
    return transformed_points_homog[:,:,:3]


def apply_transformation_to_poses(poses: torch.Tensor, transformation: np.ndarray) -> torch.Tensor:
    """Apply similarity transformation (SIM3) to camera poses."""
    transformed_poses = []
    transformation_tensor = torch.from_numpy(transformation).to(torch.float32)
    
    for pose in poses:
        transformed_pose = transformation_tensor @ pose
        transformed_poses.append(transformed_pose)
    
    return torch.stack(transformed_poses)


def compute_rigid_transformation(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
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


def verify_coordinate_system(points: torch.Tensor, camera_poses: torch.Tensor) -> bool:
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
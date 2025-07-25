"""
Undistortion utility functions for Pi3SLAM.
"""

import os
from typing import Optional, Dict, Any
from pi3.utils.undistortion import (
    UndistortionMaps, 
    create_undistortion_maps_from_file,
    VideoUndistortionLoader
)


def create_undistortion_maps(cam_dist_path: str) -> Optional[UndistortionMaps]:
    """
    Create undistortion maps from camera calibration file.
    All distortion parameters will be set to zero automatically.
    
    Args:
        cam_dist_path: Path to camera calibration file
    
    Returns:
        UndistortionMaps object or None if creation fails
    """
    if not os.path.exists(cam_dist_path):
        print(f"❌ Camera calibration file not found: {cam_dist_path}")
        return None
    
    try:
        print(f"🔧 Creating undistortion maps from file: {cam_dist_path}")
        # Always use scale=1.0 and let the UndistortionMaps class handle setting distortions to zero
        undistortion_maps = create_undistortion_maps_from_file(cam_dist_path, scale=1.0)
        
        print("✅ Undistortion maps created successfully")
        return undistortion_maps
        
    except Exception as e:
        print(f"❌ Failed to create undistortion maps: {e}")
        print("Continuing without undistortion...")
        return None


def create_undistortion_maps_from_calibration_json(cam_dist_json: Dict[str, Any]) -> Optional[UndistortionMaps]:
    """
    Create undistortion maps from camera calibration JSON data.
    All distortion parameters will be set to zero automatically.
    
    Args:
        cam_dist_json: Camera calibration JSON data
    
    Returns:
        UndistortionMaps object or None if creation fails
    """
    try:
        print(f"🔧 Creating undistortion maps from JSON data...")
        # Always use scale=1.0 and let the UndistortionMaps class handle setting distortions to zero
        undistortion_maps = create_undistortion_maps_from_file(cam_dist_json, scale=1.0)
        print("✅ Undistortion maps created successfully from JSON")
        return undistortion_maps
        
    except Exception as e:
        print(f"❌ Failed to create undistortion maps from JSON: {e}")
        print("Continuing without undistortion...")
        return None


def create_video_undistortion_loader(undistortion_maps: UndistortionMaps, 
                                   device: str = "cpu") -> Optional[VideoUndistortionLoader]:
    """
    Create a video undistortion loader for efficient video processing.
    
    Args:
        undistortion_maps: UndistortionMaps object
        device: Device to use for processing
    
    Returns:
        VideoUndistortionLoader object or None if creation fails
    """
    try:
        video_loader = VideoUndistortionLoader(undistortion_maps, device=device)
        print(f"✅ Video undistortion loader created successfully on {device}")
        return video_loader
        
    except Exception as e:
        print(f"❌ Failed to create video undistortion loader: {e}")
        print("Continuing without video undistortion loader...")
        return None


def validate_undistortion_maps(undistortion_maps: UndistortionMaps) -> bool:
    """
    Validate that undistortion maps are properly initialized.
    
    Args:
        undistortion_maps: UndistortionMaps object to validate
    
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check if the undistortion maps object has the required attributes
        if not hasattr(undistortion_maps, 'cam_dist') or not hasattr(undistortion_maps, 'cam_undist'):
            print("❌ Undistortion maps missing required camera objects")
            return False
        
        # Check if maps can be computed
        test_maps = undistortion_maps.get_maps()
        if test_maps is None or len(test_maps) != 2:
            print("❌ Failed to compute undistortion maps")
            return False
        
        print("✅ Undistortion maps validation successful")
        return True
        
    except Exception as e:
        print(f"❌ Undistortion maps validation failed: {e}")
        return False 
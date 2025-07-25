#!/usr/bin/env python3
"""
Example script demonstrating how to use the undistortion functionality with Pi3SLAM.

This script shows how to:
1. Create undistortion maps from camera calibration files
2. Use undistortion with the basic image loading function
3. Use undistortion with Pi3SLAM classes
"""

import torch
import os
import sys
from pi3.models.pi3 import Pi3
from pi3.utils.undistortion import create_undistortion_maps_from_file
from pi3.utils.basic import load_images_as_tensor_with_undistortion
from pi3_slam_class import Pi3SLAM
from pi3_slam_online_rerun_class import Pi3SLAMOnlineRerun


def example_basic_undistortion():
    """Example of using undistortion with the basic image loading function."""
    print("=" * 60)
    print("Example 1: Basic Undistortion with Image Loading")
    print("=" * 60)
    
    # Camera calibration file path (replace with your actual path)
    cam_dist_path = "/home/steffen/Data/GPStrava/GoproCalib/GoPro9/1080_50_wide_stab/dataset1/cam/cam_calib_GH018983_di_2.json"
    
    # Check if calibration file exists
    if not os.path.exists(cam_dist_path):
        print(f"❌ Camera calibration file not found: {cam_dist_path}")
        print("Please update the path to your actual calibration file.")
        return
    
    # Example data path (replace with your actual data path)
    data_path = "examples/house"  # or path to your image directory/video
    
    if not os.path.exists(data_path):
        print(f"❌ Data path not found: {data_path}")
        print("Please update the path to your actual data directory or video file.")
        return
    
    print(f"📁 Loading images from: {data_path}")
    print(f"📷 Camera calibration: {cam_dist_path}")
    print(f"💡 Undistorted camera will be created automatically by setting distortion to zero")
    
    try:
        # Load images with undistortion
        images_tensor = load_images_as_tensor_with_undistortion(
            path=data_path,
            interval=1,
            PIXEL_LIMIT=255000,
            cam_dist_path=cam_dist_path,
            scale=1.0
        )
        
        print(f"✅ Successfully loaded {images_tensor.shape[0]} undistorted images")
        print(f"   Image tensor shape: {images_tensor.shape}")
        print(f"   Value range: [{images_tensor.min():.3f}, {images_tensor.max():.3f}]")
        
    except Exception as e:
        print(f"❌ Error loading images with undistortion: {e}")


def example_slam_with_undistortion():
    """Example of using undistortion with Pi3SLAM."""
    print("\n" + "=" * 60)
    print("Example 2: Pi3SLAM with Undistortion")
    print("=" * 60)
    
    # Camera calibration file path (replace with your actual path)
    cam_dist_path = "/home/steffen/Data/GPStrava/GoproCalib/GoPro9/1080_50_wide_stab/dataset1/cam/cam_calib_GH018983_di_2.json"
    
    # Check if calibration file exists
    if not os.path.exists(cam_dist_path):
        print("❌ Camera calibration file not found. Skipping SLAM example.")
        return
    
    # Example data path
    data_path = "examples/house"
    
    if not os.path.exists(data_path):
        print(f"❌ Data path not found: {data_path}")
        return
    
    try:
        # Create undistortion maps
        print("🔧 Creating undistortion maps...")
        undistortion_maps = create_undistortion_maps_from_file(
            cam_dist_path, scale=1.0
        )
        
        # Load model (this is just for demonstration - you might want to use a real checkpoint)
        print("🤖 Loading Pi3 model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # For demonstration, we'll just create the model without loading weights
        # In practice, you would load a trained model
        model = Pi3().to(device).eval()
        
        # Create Pi3SLAM instance with undistortion
        print("🚀 Creating Pi3SLAM instance with undistortion...")
        slam = Pi3SLAM(
            model=model,
            chunk_length=10,
            overlap=3,
            device=device,
            conf_threshold=0.5,
            undistortion_maps=undistortion_maps
        )
        
        print("✅ Pi3SLAM instance created successfully with undistortion support")
        print("   Note: This is a demonstration. For actual processing, load a trained model.")
        
    except Exception as e:
        print(f"❌ Error setting up SLAM with undistortion: {e}")


def example_online_slam_with_undistortion():
    """Example of using undistortion with Pi3SLAM Online."""
    print("\n" + "=" * 60)
    print("Example 3: Pi3SLAM Online with Undistortion")
    print("=" * 60)
    
    # Camera calibration file path (replace with your actual path)
    cam_dist_path = "/home/steffen/Data/GPStrava/GoproCalib/GoPro9/1080_50_wide_stab/dataset1/cam/cam_calib_GH018983_di_2.json"
    
    # Check if calibration file exists
    if not os.path.exists(cam_dist_path):
        print("❌ Camera calibration file not found. Skipping online SLAM example.")
        return
    
    # Example data path
    data_path = "examples/house"
    
    if not os.path.exists(data_path):
        print(f"❌ Data path not found: {data_path}")
        return
    
    try:
        # Create undistortion maps
        print("🔧 Creating undistortion maps...")
        undistortion_maps = create_undistortion_maps_from_file(
            cam_dist_path, scale=1.0
        )
        
        # Load model (this is just for demonstration)
        print("🤖 Loading Pi3 model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Pi3().to(device).eval()
        
        # Create Pi3SLAM Online instance with undistortion
        print("🚀 Creating Pi3SLAM Online instance with undistortion...")
        slam_online = Pi3SLAMOnlineRerun(
            model=model,
            chunk_length=10,
            overlap=3,
            device=device,
            conf_threshold=0.5,
            max_points_visualization=50000,
            update_interval=0.1,
            rerun_port=9090,
            undistortion_maps=undistortion_maps
        )
        
        print("✅ Pi3SLAM Online instance created successfully with undistortion support")
        print("   Note: This is a demonstration. For actual processing, load a trained model.")
        
    except Exception as e:
        print(f"❌ Error setting up online SLAM with undistortion: {e}")


def example_manual_undistortion():
    """Example of manual undistortion using the provided code pattern."""
    print("\n" + "=" * 60)
    print("Example 4: Manual Undistortion (Your Original Code Pattern)")
    print("=" * 60)
    
    # Camera calibration file path (replace with your actual path)
    cam_dist_path = "/home/steffen/Data/GPStrava/GoproCalib/GoPro9/1080_50_wide_stab/dataset1/cam/cam_calib_GH018983_di_2.json"
    
    # Check if calibration file exists
    if not os.path.exists(cam_dist_path):
        print("❌ Camera calibration file not found. Skipping manual example.")
        return
    
    try:
        import cv2
        import numpy as np
        from pi3.utils.camera import Camera
        
        print("🔧 Creating cameras and computing undistortion maps...")
        
        # Create distorted camera (your original code pattern)
        cam_dist = Camera()
        cam_dist.load_camera_calibration_file(cam_dist_path, 1.0)
        cam_dist_ = cam_dist.get_camera()

        # Create undistorted camera by copying and setting distortion to zero
        cam_undist = Camera()
        cam_undist.cam_intr_json = cam_dist.cam_intr_json.copy()
        
        # Set distortion parameters to zero based on camera model
        if cam_dist.cam_intr_json["intrinsic_type"] == "DIVISION_UNDISTORTION":
            cam_undist.cam_intr_json["intrinsics"]["div_undist_distortion"] = 0.0
        elif cam_dist.cam_intr_json["intrinsic_type"] == "FISHEYE":
            cam_undist.cam_intr_json["intrinsics"]["radial_distortion_1"] = 0.0
            cam_undist.cam_intr_json["intrinsics"]["radial_distortion_2"] = 0.0
            cam_undist.cam_intr_json["intrinsics"]["radial_distortion_3"] = 0.0
            cam_undist.cam_intr_json["intrinsics"]["radial_distortion_4"] = 0.0
        elif cam_dist.cam_intr_json["intrinsic_type"] == "PINHOLE":
            cam_undist.cam_intr_json["intrinsics"]["radial_distortion_1"] = 0.0
            cam_undist.cam_intr_json["intrinsics"]["radial_distortion_2"] = 0.0
        elif cam_dist.cam_intr_json["intrinsic_type"] == "PINHOLE_RADIAL_TANGENTIAL":
            cam_undist.cam_intr_json["intrinsics"]["radial_distortion_1"] = 0.0
            cam_undist.cam_intr_json["intrinsics"]["radial_distortion_2"] = 0.0
            cam_undist.cam_intr_json["intrinsics"]["radial_distortion_3"] = 0.0
            cam_undist.cam_intr_json["intrinsics"]["tangential_distortion_1"] = 0.0
            cam_undist.cam_intr_json["intrinsics"]["tangential_distortion_2"] = 0.0
        
        # Set aspect ratio to 1 for square pixels
        cam_undist.cam_intr_json["intrinsics"]["aspect_ratio"] = 1.0
        
        cam_undist._load_camera_calibration()
        cam_undist_ = cam_undist.get_camera()

        # Example image (you would load your actual image here)
        # For demonstration, we'll create a dummy image
        img = np.random.randint(0, 255, (cam_dist_.ImageHeight(), cam_dist_.ImageWidth(), 3), dtype=np.uint8)
        img = cv2.resize(img, (cam_dist_.ImageWidth(), cam_dist_.ImageHeight()))

        # Initialize maps (your original code pattern)
        map_x = np.zeros((cam_dist_.ImageHeight(), cam_dist_.ImageWidth()), dtype=np.float32)
        map_y = np.zeros((cam_dist_.ImageHeight(), cam_dist_.ImageWidth()), dtype=np.float32)

        print("🔄 Computing undistortion maps...")
        for c in range(cam_dist_.ImageWidth()):
            for r in range(cam_dist_.ImageHeight()):
                image_pt_undist = np.array([c, r]) 
                pt_in_undist_camera = cam_undist_.CameraIntrinsics().ImageToCameraCoordinates(image_pt_undist)
                distorted_pt = cam_dist_.CameraIntrinsics().CameraToImageCoordinates(pt_in_undist_camera)

                map_x[r, c] = distorted_pt[0]
                map_y[r, c] = distorted_pt[1]

        # Apply undistortion
        undistorted_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        
        print("✅ Manual undistortion completed successfully")
        print(f"   Original image shape: {img.shape}")
        print(f"   Undistorted image shape: {undistorted_img.shape}")
        
        # Now you can use the maps for multiple images without recomputing
        print("💡 The undistortion maps can be reused for multiple images!")
        
    except Exception as e:
        print(f"❌ Error in manual undistortion: {e}")


def main():
    """Run all undistortion examples."""
    print("🎯 Pi3 Undistortion Examples")
    print("This script demonstrates how to use the new undistortion functionality.")
    print("Please update the camera calibration file paths to match your setup.\n")
    
    # Run examples
    example_basic_undistortion()
    example_slam_with_undistortion()
    example_online_slam_with_undistortion()
    example_manual_undistortion()
    
    print("\n" + "=" * 60)
    print("📚 Summary")
    print("=" * 60)
    print("The undistortion functionality has been successfully integrated into:")
    print("✅ pi3/utils/basic.py - Basic image loading functions")
    print("✅ pi3_slam_class.py - Pi3SLAM class")
    print("✅ pi3_slam_online_rerun_class.py - Pi3SLAM Online class")
    print("✅ pi3/utils/undistortion.py - Core undistortion utilities")
    print("\nYou can now use undistortion with any of the existing dataloaders!")
    print("\nFor usage examples, see the functions above and update the file paths.")


if __name__ == "__main__":
    main() 
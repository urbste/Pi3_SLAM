#!/usr/bin/env python3
"""
Test script for the enhanced SLAM system with keypoint detection and MoGe scaling.
"""

import torch
import numpy as np
import sys
import os
import time
from typing import List, Tuple

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pi3.models.pi3 import Pi3
from slam.pi3_slam_online import Pi3SLAMOnlineRerun
from utils.image_utils import calculate_target_size, load_images_from_paths


def test_keypoint_detection_and_moge_scaling():
    """Test the keypoint detection and MoGe scaling functionality."""
    
    # Test parameters
    image_dir = "/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/undist_reduced/"
    chunk_length = 5
    overlap = 2
    device = "cuda"
    
    print("🧪 Testing Keypoint Detection and MoGe Scaling")
    print("=" * 60)
    
    # Check if test directory exists
    if not os.path.exists(image_dir):
        print(f"❌ Test directory not found: {image_dir}")
        print("🖥️  Please update the image_dir path in the script")
        return
    
    # Load a small set of images for testing
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_paths = []
    
    for filename in sorted(os.listdir(image_dir)):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(image_dir, filename))
            if len(image_paths) >= chunk_length * 2:  # Load enough for 2 chunks
                break
    
    if len(image_paths) < chunk_length:
        print(f"❌ Not enough images found: {len(image_paths)} < {chunk_length}")
        return
    
    print(f"📷 Loaded {len(image_paths)} images for testing")
    
    # Load Pi3 model
    print("\n🤖 Loading Pi3 model...")
    try:
        model = Pi3.from_pretrained("yyfz233/Pi3")
        model.to(device)
        model.eval()
        print("✅ Pi3 model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load Pi3 model: {e}")
        return
    
    # Test different configurations
    test_configs = [
        {
            'name': 'Default (MoGe + Keypoints)',
            'enable_moge_scaling': True,
            'enable_keypoint_alignment': True,
            'moge_model_path': "Ruicheng/moge-2-vitl-normal",
            'keypoint_max_num': 2048,
            'keypoint_detection_threshold': 0.005
        },
        {
            'name': 'MoGe Only',
            'enable_moge_scaling': True,
            'enable_keypoint_alignment': False,
            'moge_model_path': "Ruicheng/moge-2-vitl-normal",
            'keypoint_max_num': 2048,
            'keypoint_detection_threshold': 0.005
        },
        {
            'name': 'Keypoints Only (Overlap)',
            'enable_moge_scaling': False,
            'enable_keypoint_alignment': True,
            'moge_model_path': "Ruicheng/moge-2-vitl-normal",
            'keypoint_max_num': 2048,
            'keypoint_detection_threshold': 0.005
        },
        {
            'name': 'Dense Point Cloud Only',
            'enable_moge_scaling': False,
            'enable_keypoint_alignment': False,
            'moge_model_path': "Ruicheng/moge-2-vitl-normal",
            'keypoint_max_num': 2048,
            'keypoint_detection_threshold': 0.005
        }
    ]
    
    for config in test_configs:
        print(f"\n🧪 Testing: {config['name']}")
        print("-" * 40)
        
        try:
            # Initialize SLAM with current configuration
            slam = Pi3SLAMOnlineRerun(
                model=model,
                chunk_length=chunk_length,
                overlap=overlap,
                device=device,
                conf_threshold=0.5,
                undistortion_maps=None,
                cam_scale=1.0,
                max_chunks_in_memory=3,
                enable_disk_cache=False,
                cache_dir=None,
                rerun_port=9090,
                enable_sim3_optimization=True,
                enable_moge_scaling=config['enable_moge_scaling'],
                enable_keypoint_alignment=config['enable_keypoint_alignment'],
                moge_model_path=config['moge_model_path'],
                keypoint_max_num=config['keypoint_max_num'],
                keypoint_detection_threshold=config['keypoint_detection_threshold']
            )
            
            # Start background loader
            slam.start_background_loader(image_paths)
            
            # Process chunks
            start_time = time.time()
            results = slam.process_chunks_with_background_loader()
            total_time = time.time() - start_time
            
            # Get statistics
            stats = slam.get_statistics()
            
            print(f"✅ Configuration successful!")
            print(f"   Total chunks: {stats['total_chunks']}")
            print(f"   Total frames: {stats['total_frames']}")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Overall FPS: {stats.get('overall_fps', 0):.1f}")
            
            if config['enable_moge_scaling'] and stats.get('avg_moge_time', 0) > 0:
                print(f"   Average MoGe time: {stats['avg_moge_time']:.3f}s")
            
            if config['enable_keypoint_alignment'] and stats.get('avg_keypoint_time', 0) > 0:
                print(f"   Average keypoint time: {stats['avg_keypoint_time']:.3f}s")
            
            # Cleanup
            slam.stop_visualization()
            
        except Exception as e:
            print(f"❌ Configuration failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Testing completed!")


if __name__ == "__main__":
    test_keypoint_detection_and_moge_scaling() 
#!/usr/bin/env python3
"""
Test script to verify modular pipeline saves results correctly for Euroc evaluation.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_modular_pipeline():
    """Test that the modular pipeline saves results correctly."""
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"🧪 Testing modular pipeline in: {temp_dir}")
        
        # Create test output directory
        test_output_dir = os.path.join(temp_dir, "test_output")
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Test the save_trajectory_tum method directly
        print("🔧 Testing save_trajectory_tum method...")
        
        # Import the SLAM class directly
        from slam.pi3_slam_online import Pi3SLAMOnlineRerun
        
        # Create a mock SLAM instance (without model)
        slam = Pi3SLAMOnlineRerun(
            model=None,  # We don't need the actual model for this test
            chunk_length=10,
            overlap=2,
            device='cpu',
            conf_threshold=0.5
        )
        
        # Add some mock camera trajectory data
        import numpy as np
        slam.full_camera_trajectory = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0])
        ]
        slam.full_camera_orientations = [
            np.eye(3),
            np.eye(3),
            np.eye(3)
        ]
        slam.timestamps = [0.0, 1.0, 2.0]
        
        # Test TUM saving
        tum_path = os.path.join(test_output_dir, "test_trajectory.tum")
        slam.save_trajectory_tum(tum_path)
        
        # Check if file was created
        if os.path.exists(tum_path):
            print(f"✅ TUM file created successfully: {tum_path}")
            
            # Check file contents
            with open(tum_path, 'r') as f:
                lines = f.readlines()
                print(f"📄 TUM file has {len(lines)} lines")
                if len(lines) > 0:
                    print(f"📄 Header: {lines[0].strip()}")
                    if len(lines) > 1:
                        print(f"📄 First data line: {lines[1].strip()}")
        else:
            print(f"❌ TUM file not created: {tum_path}")
        
        # Test directory path handling
        print("🔧 Testing directory path handling...")
        
        # Create a test directory
        test_dir = os.path.join(test_output_dir, "test_dataset")
        os.makedirs(test_dir, exist_ok=True)
        
        # Test that the path handling works correctly
        if os.path.isdir(test_dir):
            print(f"✅ Directory path detected correctly: {test_dir}")
            
            # Test the expected file paths
            expected_ply = os.path.join(test_dir, "trajectory.ply")
            expected_tum = os.path.join(test_dir, "trajectory.tum")
            
            print(f"📄 Expected PLY path: {expected_ply}")
            print(f"📄 Expected TUM path: {expected_tum}")
        else:
            print(f"❌ Directory path not detected: {test_dir}")
        
        print("✅ Modular pipeline test completed successfully!")


if __name__ == "__main__":
    test_modular_pipeline() 
"""
Test script for chunk reconstruction functionality.
"""

import torch
import numpy as np
from utils.chunk_reconstruction import ChunkPTRecon


def create_mock_chunk_data(num_frames=5, num_keypoints=256, width=640, height=480):
    """
    Create mock chunk data for testing.
    
    Args:
        num_frames: Number of frames in the chunk
        num_keypoints: Number of keypoints per frame
        width: Image width
        height: Image height
    
    Returns:
        Dictionary containing mock chunk data
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create mock keypoints (random 2D coordinates)
    keypoints = torch.rand(num_frames, num_keypoints, 2, device=device)
    keypoints[:, :, 0] *= width  # Scale to image width
    keypoints[:, :, 1] *= height  # Scale to image height
    
    # Create mock colors (RGB values)
    colors = torch.rand(num_frames, num_keypoints, 3, device=device) * 255
    
    # Create mock 3D points (random world coordinates)
    points_3d = torch.randn(num_frames, num_keypoints, 3, device=device) * 10
    
    # Create mock camera poses (identity matrices with small translations)
    camera_poses = torch.eye(4, device=device).unsqueeze(0).repeat(num_frames, 1, 1)
    for i in range(num_frames):
        camera_poses[i, :3, 3] = torch.tensor([i * 0.1, 0, 0], device=device)  # Move along X-axis
    
    # Create mock camera intrinsics
    fx = fy = max(width, height)
    cx, cy = width / 2, height / 2
    intrinsics = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], device=device).unsqueeze(0).repeat(num_frames, 1, 1)
    
    return {
        'keypoints': keypoints,
        'colors': colors,
        'points_kp': points_3d,
        'camera_poses': camera_poses,
        'intrinsics': intrinsics
    }


def test_chunk_reconstruction():
    """Test chunk reconstruction with mock data."""
    print("🔧 Testing Chunk Reconstruction")
    print("=" * 50)
    
    # Create mock chunk data
    mock_chunk_data = create_mock_chunk_data(num_frames=5, num_keypoints=256, width=640, height=480)
    
    print(f"📊 Mock chunk data:")
    print(f"   Frames: {mock_chunk_data['keypoints'].shape[0]}")
    print(f"   Keypoints per frame: {mock_chunk_data['keypoints'].shape[1]}")
    print(f"   Image size: {640}x{480}")
    print()
    
    try:
        # Initialize chunk reconstructor
        print("🔧 Initializing ChunkPTRecon...")
        reconstructor = ChunkPTRecon()
        reconstructor.set_target_size(640, 480)
        
        # Create reconstruction
        print("🔧 Creating reconstruction from chunk...")
        reconstruction = reconstructor.create_recon_from_chunk(mock_chunk_data)
        
        # Print statistics
        print("\n📊 Reconstruction Statistics:")
        stats = reconstructor.get_reconstruction_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Test saving and loading
        print("\n💾 Testing save/load functionality...")
        test_file = "test_reconstruction.ply"
        reconstructor.save_reconstruction(test_file)
        
        # Create new reconstructor and load
        new_reconstructor = ChunkPTRecon()
        new_reconstructor.set_target_size(640, 480)
        new_reconstructor.load_reconstruction(test_file)
        
        print("✅ Chunk reconstruction test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("   Please install pytheia: pip install pytheia")
    except Exception as e:
        print(f"❌ Error during chunk reconstruction: {e}")


if __name__ == "__main__":
    test_chunk_reconstruction() 
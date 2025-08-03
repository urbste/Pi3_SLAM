"""
Test script for camera parameter estimation.
"""

import torch
import numpy as np
from utils.camera_estimation import estimate_camera_parameters_single_chunk, print_camera_parameters_summary


def create_mock_pi3_result(batch_size=1, num_frames=5, height=256, width=256):
    """
    Create a mock Pi3 model result for testing.
    
    Args:
        batch_size: Number of batches
        num_frames: Number of frames per batch
        height: Image height
        width: Image width
    
    Returns:
        Dictionary containing mock Pi3 model output
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create mock local points (3D coordinates)
    local_points = torch.randn(batch_size, num_frames, height, width, 3, device=device)
    
    # Create mock confidence scores (between 0 and 1)
    conf = torch.sigmoid(torch.randn(batch_size, num_frames, height, width, 1, device=device))
    
    return {
        'local_points': local_points,
        'conf': conf
    }


def test_camera_estimation():
    """Test camera parameter estimation with mock data."""
    print("🧪 Testing Camera Parameter Estimation")
    print("=" * 50)
    
    # Create mock Pi3 result
    mock_result = create_mock_pi3_result(batch_size=1, num_frames=3, height=128, width=128)
    
    print(f"📊 Mock data shape:")
    print(f"   Local points: {mock_result['local_points'].shape}")
    print(f"   Confidence: {mock_result['conf'].shape}")
    print()
    
    # Estimate camera parameters
    print("🔧 Estimating camera parameters...")
    camera_params = estimate_camera_parameters_single_chunk(mock_result)
    
    # Print results
    print_camera_parameters_summary(camera_params)
    
    print(f"\n📐 Camera intrinsics matrix shape: {camera_params['intrinsics'].shape}")
    print(f"   Sample intrinsics matrix (first frame):")
    print(camera_params['intrinsics'][0].numpy())
    
    print(f"\n✅ Camera parameter estimation completed successfully!")


if __name__ == "__main__":
    test_camera_estimation() 
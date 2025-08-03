"""
Test script for keypoint extraction functionality.
"""

import torch
import numpy as np
from utils.keypoint_extraction import ALIKEDExtractor


def create_mock_images(batch_size=1, num_frames=3, height=256, width=256):
    """
    Create mock images for testing.
    
    Args:
        batch_size: Number of batches
        num_frames: Number of frames per batch
        height: Image height
        width: Image width
    
    Returns:
        Tensor of shape (batch_size, num_frames, 3, height, width) containing mock images
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create mock RGB images (values between 0 and 1)
    images = torch.rand(batch_size, num_frames, 3, height, width, device=device)
    
    return images


def test_keypoint_extraction():
    """Test keypoint extraction with mock data."""
    print("🔍 Testing Keypoint Extraction")
    print("=" * 50)
    
    # Create mock images
    mock_images = create_mock_images(batch_size=1, num_frames=3, height=128, width=128)
    
    print(f"📊 Mock images shape: {mock_images.shape}")
    print()
    
    try:
        # Initialize keypoint extractor
        print("🔧 Initializing ALIKED extractor...")
        extractor = ALIKEDExtractor(
            max_num_keypoints=256,
            detection_threshold=0.005,
            device=mock_images.device
        )
        
        # Extract keypoints
        print("🔍 Extracting keypoints...")
        keypoint_results = extractor.extract_with_colors(mock_images)
        
        # Print results
        print(f"✅ Keypoint extraction completed!")
        print(f"   Keypoints shape: {keypoint_results['keypoints'].shape}")
        print(f"   Descriptors shape: {keypoint_results['descriptors'].shape}")
        print(f"   Scores shape: {keypoint_results['scores'].shape}")
        print(f"   Colors shape: {keypoint_results['colors'].shape}")
        
        # Print summary statistics
        num_frames = keypoint_results['keypoints'].shape[1]
        total_keypoints = keypoint_results['keypoints'].shape[-2]
        avg_keypoints = total_keypoints / num_frames if num_frames > 0 else 0
        
        print(f"\n📊 Keypoint Summary:")
        print(f"   Number of frames: {num_frames}")
        print(f"   Total keypoints: {total_keypoints}")
        print(f"   Average keypoints per frame: {avg_keypoints:.1f}")
        
        # Show sample keypoint coordinates
        if keypoint_results['keypoints'].shape[-2] > 0:
            sample_keypoints = keypoint_results['keypoints'][0, 0, :5]  # First 5 keypoints from first frame
            print(f"   Sample keypoints (first 5):")
            for i, kp in enumerate(sample_keypoints):
                print(f"     Keypoint {i}: ({kp[0]:.1f}, {kp[1]:.1f})")
        
        # Show sample colors
        if keypoint_results['colors'].shape[-2] > 0:
            sample_colors = keypoint_results['colors'][0, 0, :5]  # First 5 colors from first frame
            print(f"   Sample colors (first 5):")
            for i, color in enumerate(sample_colors):
                print(f"     Color {i}: RGB({color[0]}, {color[1]}, {color[2]})")
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("   Please install lightglue: pip install lightglue")
    except Exception as e:
        print(f"❌ Error during keypoint extraction: {e}")


if __name__ == "__main__":
    test_keypoint_extraction() 
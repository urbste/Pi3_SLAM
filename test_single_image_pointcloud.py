#!/usr/bin/env python3
"""
Test script for estimating point cloud from single image using Pi3 model.
"""

import torch
import numpy as np
import sys
import os
from PIL import Image
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pi3.models.pi3 import Pi3
from utils.image_utils import calculate_target_size, load_single_image


def estimate_pointcloud_from_image(model: Pi3, image_path: str, device: str = 'cuda', 
                                 conf_threshold: float = 0.5, max_points: int = 50000):
    """
    Estimate point cloud from a single image using Pi3 model.
    
    Args:
        model: Pi3 model instance
        image_path: Path to input image
        device: Device to run inference on
        conf_threshold: Confidence threshold for filtering points
        max_points: Maximum number of points to visualize
    
    Returns:
        Dictionary containing points, colors, camera pose, and confidence scores
    """
    print(f"🖼️  Loading image: {image_path}")
    
    # Load and preprocess image
    start_time = time.time()
    
    # Calculate target size for memory efficiency
    target_size = calculate_target_size(image_path, pixel_limit=255000)
    print(f"📏 Target size: {target_size}")
    
    # Load image
    image_tensor = load_single_image(image_path, target_size, device='cpu')
    print(f"📷 Image tensor shape: {image_tensor.shape}")
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)
    
    load_time = time.time() - start_time
    print(f"⏱️  Image loading time: {load_time:.3f}s")
    
    # Model inference
    print(f"🤖 Running Pi3 inference on {device}...")
    inference_start = time.time()
    
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            result = model(image_tensor[None].to(device))
    
    inference_time = time.time() - inference_start
    print(f"⏱️  Inference time: {inference_time:.3f}s")
    
    # Process results
    print("🔧 Processing results...")
    process_start = time.time()
    
    # Extract points and confidence
    points = result['points'][0]  # Remove batch dimension
    confidence = result['conf'][0]
    camera_pose = result['camera_poses'][0]
    
    print(f"🔍 Debug info:")
    print(f"  Points shape: {points.shape}")
    print(f"  Confidence shape: {confidence.shape}")
    print(f"  Confidence values range: [{confidence.min():.6f}, {confidence.max():.6f}]")
    print(f"  Sigmoid confidence range: [{torch.sigmoid(confidence).min():.6f}, {torch.sigmoid(confidence).max():.6f}]")
    
    # Filter by confidence
    masks = torch.sigmoid(confidence[..., 0]) > conf_threshold
    print(f"📊 Confidence threshold: {conf_threshold}")
    print(f"📊 Total points: {points.shape[0]}")
    print(f"📊 High confidence points: {masks.sum().item()}")
    
    # Apply mask
    filtered_points = points[masks]
    filtered_confidence = confidence[masks]
    
    # Extract colors from original image
    image_np = image_tensor[0].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
    colors = image_tensor[0].permute(1, 2, 0).view(-1, 3).cpu().numpy()
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)

    
    process_time = time.time() - process_start
    print(f"⏱️  Processing time: {process_time:.3f}s")
    
    # Subsample if too many points
    if len(filtered_points) > max_points:
        indices = np.random.choice(len(filtered_points), max_points, replace=False)
        filtered_points = filtered_points[indices]
        filtered_confidence = filtered_confidence[indices]
        colors = colors[indices]
        print(f"📊 Subsampled to {max_points} points for visualization")
    
    return {
        'points': filtered_points.cpu().numpy(),
        'colors': colors,
        'camera_pose': camera_pose.cpu().numpy(),
        'confidence': filtered_confidence.cpu().numpy(),
        'image': image_np,
        'timing': {
            'load': load_time,
            'inference': inference_time,
            'process': process_time
        }
    }


def visualize_pointcloud_open3d(points: np.ndarray, colors: np.ndarray, 
                               camera_pose: np.ndarray = None, image: np.ndarray = None):
    """
    Visualize point cloud using Open3D.
    
    Args:
        points: 3D points (N, 3)
        colors: Point colors (N, 3) in [0, 1] range
        camera_pose: Camera pose matrix (4, 4) for visualization
        image: Original image for reference
    """
    try:
        import open3d as o3d
        print("✅ Open3D imported successfully")
    except ImportError as e:
        print(f"❌ Open3D import failed: {e}")
        return False
    
    print(f"🎨 Creating Open3D visualization with {len(points)} points...")
    

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Set colors (convert from [0, 1] to [0, 255])
    colors_uint8 = (colors * 255).astype(np.uint8)
    pcd.colors = o3d.utility.Vector3dVector(colors_uint8 / 255.0)  # Open3D expects [0, 1]
    
    # Create coordinate frame for camera pose
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Pi3 Point Cloud Visualization", width=1200, height=800)
    
    # Add geometries
    vis.add_geometry(pcd)
    if camera_pose is not None:
        # Handle camera pose - remove extra dimension if present and convert to float64
        if camera_pose.ndim == 3:
            camera_pose = camera_pose[0]  # Remove batch dimension
        camera_pose = camera_pose.astype(np.float64)  # Convert to float64
        
        # Transform coordinate frame to camera pose
        coordinate_frame.transform(camera_pose)
        vis.add_geometry(coordinate_frame)
    
    # Set initial view
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    
    # Add camera controls
    print("🎮 Controls:")
    print("  - Mouse: Rotate view")
    print("  - Shift + Mouse: Pan view")
    print("  - Ctrl + Mouse: Zoom")
    print("  - Q: Quit visualization")
    
    # Run visualization
    print("🖥️  Starting visualization...")
    vis.run()
    vis.destroy_window()
    
    return True


def main():
    """Main function to test single image point cloud estimation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Estimate point cloud from single image using Pi3")
    parser.add_argument("--image_path", default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/undist/frame_00000.jpg", help="Path to input image")
    parser.add_argument("--device", default="cuda", help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--conf-threshold", type=float, default=0.0, help="Confidence threshold")
    parser.add_argument("--max-points", type=int, default=50000, help="Maximum points to visualize")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"❌ Image not found: {args.image_path}")
        return
    
    print("🚀 Pi3 Single Image Point Cloud Estimation")
    print("=" * 50)
    print(f"📁 Image: {args.image_path}")
    print(f"🖥️  Device: {args.device}")
    print(f"🎯 Confidence threshold: {args.conf_threshold}")
    print(f"📊 Max points: {args.max_points}")
    
    # Load Pi3 model
    print("\n🤖 Loading Pi3 model...")
    try:
        model = Pi3.from_pretrained("yyfz233/Pi3")
        model.to(args.device)
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Estimate point cloud
    print("\n🔄 Estimating point cloud...")
    try:
        result = estimate_pointcloud_from_image(
            model, args.image_path, args.device, args.conf_threshold, args.max_points
        )
        
        points = result['points']
        colors = result['colors']
        camera_pose = result['camera_pose']
        confidence = result['confidence']
        timing = result['timing']
        
        print(f"\n📊 Results:")
        print(f"  Points: {len(points)}")
        if len(confidence) > 0:
            print(f"  Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
        else:
            print(f"  Confidence range: No points above threshold")
        if len(points) > 0:
            print(f"  Point range: X[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
            print(f"  Point range: Y[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
            print(f"  Point range: Z[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        else:
            print(f"  Point range: No points to display")
        print(f"  Camera pose:\n{camera_pose}")
        print(f"  Total time: {sum(timing.values()):.3f}s")
        
    except Exception as e:
        print(f"❌ Point cloud estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Visualize if requested
    if not args.no_viz:
        print("\n🎨 Starting visualization...")
        try:
            success = visualize_pointcloud_open3d(points, colors, camera_pose, result.get('image'))
            if success:
                print("✅ Visualization completed")
            else:
                print("❌ Visualization failed")
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎉 Test completed!")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Test script for multi-view alignment using Pi3, LoFTR, and OpenCV PnP.
"""

import torch
import numpy as np
import sys
import os
import cv2
import time
from typing import List, Tuple, Dict
from PIL import Image

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pi3.models.pi3 import Pi3
from utils.image_utils import calculate_target_size, load_single_image
from utils.geometry_torch import recover_focal_shift



def load_and_preprocess_images(image_paths: List[str], target_size: Tuple[int, int], device: str = 'cpu') -> torch.Tensor:
    """
    Load and preprocess multiple images.
    
    Args:
        image_paths: List of image paths
        target_size: Target size (width, height)
        device: Device to load tensors on
    
    Returns:
        Batch of preprocessed image tensors (N, C, H, W)
    """
    images = []
    for path in image_paths:
        image_tensor = load_single_image(path, target_size, device)
        images.append(image_tensor)
    
    return torch.stack(images)


def create_reference_map(model: Pi3, keyframe_paths: List[str], device: str = 'cuda') -> Dict:
    """
    Create reference map from keyframes using Pi3.
    
    Args:
        model: Pi3 model instance
        keyframe_paths: List of keyframe image paths
        device: Device to run inference on
    
    Returns:
        Dictionary containing reference map data
    """
    print(f"🗺️  Creating reference map from {len(keyframe_paths)} keyframes...")
    
    # Calculate target size for memory efficiency
    target_size = calculate_target_size(keyframe_paths[0], pixel_limit=255000)
    print(f"📏 Target size: {target_size}")
    
    # Load keyframe images
    start_time = time.time()
    keyframe_images = load_and_preprocess_images(keyframe_paths, target_size, device='cpu')
    print(f"📷 Keyframe images shape: {keyframe_images.shape}")
    
    # Add batch dimension for Pi3 model (B, N, C, H, W)
    keyframe_images = keyframe_images.unsqueeze(0)  # Add batch dimension
    print(f"📷 Keyframe images shape after adding batch dim: {keyframe_images.shape}")
    
    load_time = time.time() - start_time
    print(f"⏱️  Image loading time: {load_time:.3f}s")
    
    # Run Pi3 inference
    print(f"🤖 Running Pi3 inference on {device}...")
    inference_start = time.time()
    
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            result = model(keyframe_images.to(device))
    
    inference_time = time.time() - inference_start
    print(f"⏱️  Inference time: {inference_time:.3f}s")
    
    # Process results
    print("🔧 Processing reference map...")
    process_start = time.time()
    
    # Extract points, confidence, and camera poses
    points = result['points'][0]  # Remove batch dimension: (N, H, W, 3)
    local_points = result['local_points'][0]  # Remove batch dimension: (N, H, W, 3)

    print(f"  Local points shape: {local_points.shape}")
    print
    
    confidence = result['conf'][0]  # Remove batch dimension: (N, H, W, 1)
    camera_poses = result['camera_poses'][0]  # Remove batch dimension: (N, 4, 4)
    
    gt_focal = 491
    # scale with original and target size
    gt_scaled = gt_focal * target_size[1] / 960

    focal = torch.tensor([gt_scaled]).expand(points.shape[0]).to(points.device)

    focal, shift = recover_focal_shift(local_points, confidence > 0.5, focal=focal)
    print(focal)
    print(shift)
    aspect_ratio = target_size[0] / target_size[1]
    import utils3d

    fx, fy = gt_scaled / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, gt_scaled / 2 * (1 + aspect_ratio ** 2) ** 0.5 

    intrinsics = utils3d.torch.intrinsics_from_focal_center(gt_scaled/target_size[1], gt_scaled/target_size[1], 0.5, 0.5)
    print(intrinsics)
    print(shift.shape)
    print(points.shape)
    local_points += shift[..., None, None, None].to(local_points.device)
    depth = local_points[..., 2].clone()
    points = utils3d.torch.depth_to_points(depth, intrinsics=intrinsics.to(depth.device))
    print(depth)
    print(points[..., 2])
    print(local_points[..., 2])
    # find scale between points and local_points
    scale = points[..., 2] / local_points[..., 2]
    print(scale)
    camera_poses = camera_poses * scale[..., None, None]

    print(f"🔍 Debug tensor shapes:")
    print(f"  Points shape: {points.shape}")
    print(f"  Confidence shape: {confidence.shape}")
    print(f"  Camera poses shape: {camera_poses.shape}")
    print(f"  Focal: {fx} {fy}")
    print(f"  intrinsics: {intrinsics}")
    
    # Reshape points and confidence to 2D for easier processing
    if len(points.shape) == 4:  # (N, H, W, 3)
        N, H, W, _ = points.shape
        points = points.reshape(N, H*W, 3)  # (N, H*W, 3)
        confidence = confidence.reshape(N, H*W, 1)  # (N, H*W, 1)
    else:
        print(f"⚠️  Unexpected points shape: {points.shape}")
        # Try to handle as is
        pass
    
    # Filter by confidence
    masks = torch.sigmoid(confidence[..., 0]) > 0.3
    print(f"📊 Total points per keyframe: {points.shape[1]}")
    print(f"📊 High confidence points per keyframe: {masks.sum(dim=1).tolist()}")
    
    # Combine all high-confidence points from all keyframes
    all_points = []
    all_colors = []
    all_camera_poses = []
    
    for i in range(len(keyframe_paths)):
        # Get filtered points for this keyframe
        keyframe_points = points[i][masks[i]]
        keyframe_confidence = confidence[i][masks[i]]
        
        # Get colors from original image (access the correct dimension)
        image_tensor = keyframe_images[0, i]  # (C, H, W) - batch=0, keyframe=i
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
        colors = image_tensor.permute(1, 2, 0).view(-1, 3).cpu().numpy()
        
        # Apply mask to colors (simplified - in practice you'd need proper point-to-pixel mapping)
        if len(keyframe_points) > 0:
            # Use random colors for now (proper mapping would require back-projection)
            keyframe_colors = np.random.rand(len(keyframe_points), 3)
            keyframe_colors = np.clip(keyframe_colors, 0, 1)
            
            all_points.append(keyframe_points.cpu().numpy())
            all_colors.append(keyframe_colors)
            all_camera_poses.append(camera_poses[i].cpu().numpy())
    
    process_time = time.time() - process_start
    print(f"⏱️  Processing time: {process_time:.3f}s")
    
    # Combine all data
    if all_points:
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        reference_camera_poses = np.stack(all_camera_poses)
    else:
        combined_points = np.array([])
        combined_colors = np.array([])
        reference_camera_poses = np.array([])
    
    print(f"📊 Reference map created:")
    print(f"  Total points: {len(combined_points)}")
    print(f"  Reference cameras: {len(reference_camera_poses)}")
    
    return {
        'points': combined_points,
        'colors': combined_colors,
        'camera_poses': reference_camera_poses,
        'keyframe_images': keyframe_images,
        'timing': {
            'load': load_time,
            'inference': inference_time,
            'process': process_time
        }
    }


def match_and_align_query_image(query_image_path: str, reference_map: Dict, device: str = 'cuda') -> Dict:
    """
    Match and align a query image to the reference map using LoFTR and OpenCV PnP.
    
    Args:
        query_image_path: Path to query image
        reference_map: Reference map dictionary
        device: Device to run inference on
    
    Returns:
        Dictionary containing alignment results
    """
    print(f"🔍 Aligning query image: {query_image_path}")
    
    try:
        # Import LoFTR
        import kornia.feature as KF
        print("✅ LoFTR imported successfully")
    except ImportError:
        print("❌ LoFTR not available. Please install kornia: pip install kornia")
        return {'success': False, 'error': 'LoFTR not available'}
    
    # Load and preprocess query image
    target_size = calculate_target_size(query_image_path, pixel_limit=255000)
    query_image = load_single_image(query_image_path, target_size, device='cpu')
    query_image = query_image.unsqueeze(0)  # Add batch dimension: (1, C, H, W)
    print(f"📷 Query image shape: {query_image.shape}")
    
    # Get reference image (first keyframe)
    reference_images = reference_map['keyframe_images']  # (1, N, C, H, W)
    reference_image = reference_images[0, 0]  # First keyframe: (C, H, W)
    reference_image = reference_image.unsqueeze(0)  # Add batch dimension: (1, C, H, W)
    print(f"📷 Reference image shape: {reference_image.shape}")
    
    # Convert to grayscale for LoFTR
    def rgb_to_grayscale(rgb_tensor):
        """Convert RGB tensor to grayscale."""
        # RGB to grayscale conversion weights
        weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        grayscale = torch.sum(rgb_tensor * weights, dim=1, keepdim=True)
        return grayscale
    
    query_gray = rgb_to_grayscale(query_image)
    reference_gray = rgb_to_grayscale(reference_image)
    
    print(f"📷 Query grayscale shape: {query_gray.shape}")
    print(f"📷 Reference grayscale shape: {reference_gray.shape}")
    
    # Initialize LoFTR
    print("🔍 Running LoFTR feature matching...")
    matcher = KF.LoFTR(pretrained='outdoor')
    matcher.to(device)
    
    # Run feature matching
    try:
        with torch.no_grad():
            # Move to device
            query_gray = query_gray.to(device)
            reference_gray = reference_gray.to(device)
            
            # Get matches
            input_dict = {
                'image0': query_gray,
                'image1': reference_gray
            }
            
            correspondences = matcher(input_dict)
            print(correspondences)
            
            # Extract keypoints and matches
            kpts0 = correspondences['keypoints0'].cpu().numpy()
            kpts1 = correspondences['keypoints1'].cpu().numpy()
            matches = correspondences['confidence'].cpu().numpy()
            
            # Filter matches by confidence
            confidence_threshold = 0.5
            good_matches = matches > confidence_threshold
            
            if good_matches.sum() < 4:
                print(f"❌ Not enough good matches: {good_matches.sum()}")
                return {'success': False, 'error': 'Not enough good matches'}
            
            # Get matched keypoints
            matched_kpts0 = kpts0[good_matches]
            matched_kpts1 = kpts1[good_matches]
            
            print(f"📊 Found {len(matched_kpts0)} good matches")
            
            # Get 3D points from reference map for PnP
            reference_points_3d = reference_map['points']
            reference_image_shape = reference_gray.shape[-2:]  # (H, W)
            
            # Convert 2D keypoints to 3D points using the reference map
            # This is a simplified approach - in practice you'd need proper back-projection
            matched_3d_points = []
            matched_2d_points = []
            
            for kpt in matched_kpts1:
                # Convert normalized coordinates to pixel coordinates
                x, y = kpt[0], kpt[1]
                
                # Convert to integer pixel coordinates
                px, py = int(x), int(y)
                
                # Ensure coordinates are within bounds
                if 0 <= px < reference_image_shape[1] and 0 <= py < reference_image_shape[0]:
                    # Get 3D point from reference map (simplified - use nearest neighbor)
                    # In practice, you'd need proper back-projection from the Pi3 reconstruction
                    idx = py * reference_image_shape[1] + px
                    if idx < len(reference_points_3d):
                        matched_3d_points.append(reference_points_3d[idx])
                        matched_2d_points.append(kpt)
            
            if len(matched_3d_points) < 4:
                print(f"❌ Not enough 3D-2D correspondences: {len(matched_3d_points)}")
                return {'success': False, 'error': 'Not enough 3D-2D correspondences'}
            
            # Convert to numpy arrays
            points_3d = np.array(matched_3d_points, dtype=np.float32)
            points_2d = np.array(matched_2d_points, dtype=np.float32)
            
            # Use OpenCV PnP to estimate camera pose
            camera_matrix = np.array([
                [1000, 0, reference_image_shape[1] / 2],
                [0, 1000, reference_image_shape[0] / 2],
                [0, 0, 1]
            ], dtype=np.float32)
            
            dist_coeffs = np.zeros(5, dtype=np.float32)
            
            # Solve PnP
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d, points_2d, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                print("❌ PnP failed")
                return {'success': False, 'error': 'PnP failed'}
            
            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            
            # Create transformation matrix
            transformation = np.eye(4, dtype=np.float32)
            transformation[:3, :3] = R
            transformation[:3, 3] = tvec.flatten()
            
            print(f"✅ Alignment successful with {len(inliers)} inliers")
            
            return {
                'success': True,
                'transformation': transformation,
                'inliers': len(inliers),
                'total_matches': len(matched_kpts0)
            }
            
    except Exception as e:
        print(f"❌ Alignment error: {e}")
        return {'success': False, 'error': str(e)}


def visualize_multiview_alignment(reference_map: Dict, query_results: List[Dict], query_image_paths: List[str]):
    """
    Visualize multi-view alignment using Open3D.
    
    Args:
        reference_map: Reference map data
        query_results: List of query alignment results
        query_image_paths: List of query image paths
    """
    try:
        import open3d as o3d
        print("✅ Open3D imported successfully")
    except ImportError as e:
        print(f"❌ Open3D import failed: {e}")
        return False
    
    print(f"🎨 Creating multi-view visualization...")
    
    # Create point cloud for reference map
    reference_points = reference_map['points']
    reference_colors = reference_map['colors']
    
    if len(reference_points) == 0:
        print("❌ No reference points to visualize")
        return False
    
    # Create reference point cloud (blue color)
    reference_pcd = o3d.geometry.PointCloud()
    reference_pcd.points = o3d.utility.Vector3dVector(reference_points)
    reference_colors_blue = np.full((len(reference_points), 3), [0.0, 0.0, 1.0])  # Blue
    reference_pcd.colors = o3d.utility.Vector3dVector(reference_colors_blue)
    
    # Create coordinate frames for reference cameras (blue)
    reference_frames = []
    for i, pose in enumerate(reference_map['camera_poses']):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        frame.transform(pose.astype(np.float64))
        # Color the frame blue
        frame.paint_uniform_color([0.0, 0.0, 1.0])
        reference_frames.append(frame)
    
    # Create coordinate frames for query cameras (red)
    query_frames = []
    for i, result in enumerate(query_results):
        if result is not None:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            frame.transform(result['transformation'].astype(np.float64))
            # Color the frame red
            frame.paint_uniform_color([1.0, 0.0, 0.0])
            query_frames.append(frame)
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Multi-View Alignment", width=1400, height=900)
    
    # Add geometries
    vis.add_geometry(reference_pcd)
    
    for frame in reference_frames:
        vis.add_geometry(frame)
    
    for frame in query_frames:
        vis.add_geometry(frame)
    
    # Set initial view
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    
    # Add legend
    print("🎨 Visualization legend:")
    print("  🔵 Blue: Reference cameras and points")
    print("  🔴 Red: Query cameras (aligned)")
    
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
    """Main function to test multi-view alignment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-view alignment using Pi3, LoFTR, and OpenCV PnP")
    parser.add_argument("--keyframes", nargs='+', required=True, help="Keyframe image paths")
    parser.add_argument("--queries", nargs='+', required=True, help="Query image paths to align")
    parser.add_argument("--device", default="cuda", help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    
    args = parser.parse_args()
    
    # Check if images exist
    all_images = args.keyframes + args.queries
    for image_path in all_images:
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
    
    print("🚀 Multi-View Alignment Pipeline")
    print("=" * 50)
    print(f"🗺️  Keyframes: {len(args.keyframes)} images")
    print(f"🔍 Queries: {len(args.queries)} images")
    print(f"🖥️  Device: {args.device}")
    
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
    
    # Create reference map
    print("\n🗺️  Creating reference map...")
    try:
        reference_map = create_reference_map(model, args.keyframes, args.device)
        
        if len(reference_map['points']) == 0:
            print("❌ No points in reference map")
            return
            
        print(f"✅ Reference map created with {len(reference_map['points'])} points")
        
    except Exception as e:
        print(f"❌ Reference map creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Align query images
    print("\n🔍 Aligning query images...")
    query_results = []
    
    for i, query_path in enumerate(args.queries):
        print(f"\n📷 Processing query {i+1}/{len(args.queries)}: {query_path}")
        try:
            result = match_and_align_query_image(query_path, reference_map, args.device)
            query_results.append(result)
            
            if result['success']:
                print(f"✅ Query {i+1} aligned successfully")
            else:
                print(f"❌ Query {i+1} alignment failed: {result['error']}")
                
        except Exception as e:
            print(f"❌ Query {i+1} alignment error: {e}")
            query_results.append({'success': False, 'error': str(e)})
    
    # Print summary
    print(f"\n📊 Alignment Summary:")
    print(f"  Reference cameras: {len(reference_map['camera_poses'])}")
    print(f"  Reference points: {len(reference_map['points'])}")
    successful_alignments = sum(1 for r in query_results if r['success'])
    print(f"  Successful alignments: {successful_alignments}/{len(args.queries)}")
    
    # Visualize if requested
    if not args.no_viz and successful_alignments > 0:
        print("\n🎨 Starting visualization...")
        try:
            success = visualize_multiview_alignment(reference_map, query_results, args.queries)
            if success:
                print("✅ Visualization completed")
            else:
                print("❌ Visualization failed")
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎉 Multi-view alignment test completed!")


if __name__ == "__main__":
    main() 
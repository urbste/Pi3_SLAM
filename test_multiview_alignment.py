#!/usr/bin/env python3
"""
Test script for loading a single chunk of 10 images and running PI3 reconstruction
to recover intrinsics from the point map.
"""

import torch
import numpy as np
import sys
import os
import glob
from PIL import Image
import time
from typing import List, Tuple

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pi3.models.pi3 import Pi3
from utils.image_utils import calculate_target_size, load_images_from_paths
from utils.geometry_torch import recover_focal_shift


def load_image_chunk_from_folder(folder_path: str, chunk_size: int = 10, start_idx: int = 0) -> List[str]:
    """
    Load a single chunk of images from a folder.
    
    Args:
        folder_path: Path to the folder containing images
        chunk_size: Number of images to load (default: 10)
        start_idx: Starting index for the chunk (default: 0)
    
    Returns:
        List of image paths for the chunk
    """
    print(f"📁 Loading image chunk from folder: {folder_path}")
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_paths = []
    
    for filename in sorted(os.listdir(folder_path)):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(folder_path, filename))
    
    print(f"📊 Found {len(image_paths)} total images in folder")
    
    # Check if we have enough images
    if start_idx >= len(image_paths):
        raise ValueError(f"Start index {start_idx} exceeds total images {len(image_paths)}")
    
    # Extract chunk
    end_idx = min(start_idx + chunk_size, len(image_paths))
    chunk_paths = image_paths[start_idx:end_idx]
    
    print(f"📷 Loading chunk {start_idx}-{end_idx-1} ({len(chunk_paths)} images)")
    
    return chunk_paths


def run_pi3_reconstruction(model: Pi3, image_paths: List[str], device: str = 'cuda') -> dict:
    """
    Run PI3 reconstruction on a chunk of images.
    
    Args:
        model: Pi3 model instance
        image_paths: List of image paths
        device: Device to run inference on
    
    Returns:
        Dictionary containing reconstruction results
    """
    print(f"🤖 Running PI3 reconstruction on {len(image_paths)} images...")
    
    # Calculate target size for memory efficiency
    target_size = 518
    
    # Load images
    start_time = time.time()
    #images_tensor, original_coords = load_and_preprocess_images_square(image_paths, target_size)
    images_tensor = load_images_from_paths(image_paths, calculate_target_size(image_paths[0]))
    print(f"📷 Images tensor shape: {images_tensor.shape}")
    
    # Add batch dimension: (N, C, H, W) -> (1, N, C, H, W)
    images_tensor = images_tensor.unsqueeze(0)
    
    load_time = time.time() - start_time
    print(f"⏱️  Image loading time: {load_time:.3f}s")
    
    # Model inference
    print(f"🤖 Running Pi3 inference on {device}...")
    inference_start = time.time()
    
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            result = model(images_tensor.to(device))
    
    inference_time = time.time() - inference_start
    print(f"⏱️  Inference time: {inference_time:.3f}s")
    
    return result, images_tensor


def recover_intrinsics_from_points(result: dict, images_tensor: torch.Tensor) -> dict:
    """
    Recover intrinsics from the point map using recover_focal_shift function.
    
    Args:
        result: PI3 reconstruction result dictionary
    
    Returns:
        Dictionary containing recovered intrinsics
    """
    print("🔧 Recovering intrinsics from point map...")
    
    # Extract points and confidence
    points = result["local_points"]  # Shape: (B, N, H, W, 3)
    masks = torch.sigmoid(result["conf"][..., 0]) > 0.1  # Shape: (B, N, H, W)

    # Get original dimensions
    original_height, original_width = points.shape[-3:-1]
    aspect_ratio = original_width / original_height

    
    # Use recover_focal_shift function from MoGe
    focal, shift = recover_focal_shift(points, masks, downsample_size=(64, 64))
        
    # Calculate fx, fy from focal
    fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio * original_width
    fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 * original_height
    
    cx = original_width//2 
    cy = original_height//2 
    
    import utils3d
    intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, cx, cy)

    from moge.model.v2 import MoGeModel # Let's try MoGe-2
    device = torch.device("cuda")
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()


    def get_scale_factor_for_pi3(moge_metric_depth, pi3_metric_depth, mask):
        moge_metric_depth = moge_metric_depth[mask]
        pi3_metric_depth = pi3_metric_depth[mask]
        scale_factor = moge_metric_depth / pi3_metric_depth
        scale_factor = scale_factor.median()
        return scale_factor



    from pi3.utils.geometry import homogenize_points
    # get local points map of first camera
    result["intrinsics"] = intrinsics
    result["shift"] = shift

    from utils.geometry_torch import project_xyz_to_uv

    # now project a point from camera one to camera two
    # take a point and unproject it using the intrinsics of camera one
    
    def interpolate_color_for_ref(images, keypoints):
        # get the color of the point using grid_sample
        H, W = images.shape[-2:]
        grid_x = (keypoints[:, :, 0] / (W - 1)) * 2 - 1  # Convert to [-1, 1]
        grid_y = (keypoints[:, :, 1] / (H - 1)) * 2 - 1  # Convert to [-1, 1]
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # Shape: (1, 1, N, 2)
        color = torch.nn.functional.grid_sample(
            images.to(keypoints.device), grid,
            mode="bilinear", align_corners=False, padding_mode="border")
        return (color.squeeze()*255).detach().cpu().numpy().astype(np.uint8)


    def interpolate_world_points_for_ref(result, keypoints, idx_ref):
        # unproject the point from camera 0 to camera 0 world
        p_cam_ref = torch.linalg.inv(result["intrinsics"][:,idx_ref]) @ homogenize_points(keypoints.float()).transpose(-2,-1)
        
        # Get the depth map for the reference camera
        depth_map = result["local_points"][0, idx_ref, :, :, 2]  # Shape: (H, W) - extract Z component
        
        # Normalize grid coordinates to [-1, 1] range for grid_sample
        H, W = depth_map.shape
        grid_x = (keypoints[:, :, 0] / (W - 1)) * 2 - 1  # Convert to [-1, 1]
        grid_y = (keypoints[:, :, 1] / (H - 1)) * 2 - 1  # Convert to [-1, 1]
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # Shape: (1, 1, N, 2)
        
        # Interpolate depth values
        depth_map_unsqueezed = depth_map.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
        z_in_cam_ref = torch.nn.functional.grid_sample(
            depth_map_unsqueezed, grid, mode="bilinear", align_corners=False, padding_mode="border"
        ).squeeze()  # Shape: (N,)
        
        # Add the shift from the estimation
        z_in_cam_ref = z_in_cam_ref + result["shift"][0, idx_ref]
        
        # Create 3D points in camera reference frame
        p_cam_ref_3d = p_cam_ref * z_in_cam_ref.unsqueeze(0)  # Shape: (N, 3)

        # transform the point to the world frame
        p_cam_ref_world = result["camera_poses"][0, idx_ref] @ homogenize_points(p_cam_ref_3d.transpose(-2,-1)).transpose(-2,-1)

        return p_cam_ref_world

    def project_point_to_other_camera(result, pts_2d_cam_ref, idx_ref, idx_proj=[]):
        # unproject the point from camera 0 to camera 0 world
        p_cam_ref_world = interpolate_world_points_for_ref(result, pts_2d_cam_ref, idx_ref)

        # finally project the point to the other cameras
        uv_proj = []
        for idx in idx_proj: 
            p_cam_proj = result["camera_poses_cw"][0, idx] @ p_cam_ref_world
            uv_proj1 = project_xyz_to_uv(p_cam_proj.transpose(-2,-1), torch.tensor([0.0], device=points.device), result["intrinsics"][0, idx])[0]
            uv_proj.append(uv_proj1)
        return uv_proj
        
    # Test point projection
    # detect aliked features
    ref_id = 0

    # first image metric depth
    moge_model.eval()
    with torch.no_grad():
        moge_metric_depth = moge_model.infer(images_tensor[[0], ref_id].to(device))["depth"]

    scale_factor = get_scale_factor_for_pi3(moge_metric_depth[0], result["local_points"][0, 0, :, :, 2], masks[0, 0, :, :])
    print(f"📐 Scale factor: {scale_factor}")

    # scale the reconstruction
    result["local_points"] = result["local_points"] * scale_factor
    result["points"] = result["points"] * scale_factor
    result["camera_poses"][:,:,:3,3] = result["camera_poses"][:,:,:3,3] * scale_factor
    result["camera_poses_cw"] = torch.linalg.inv(result["camera_poses"])
    result["shift"] = result["shift"] * scale_factor


    from lightglue import ALIKED, SIFT, SuperPoint
    aliked_extractor = ALIKED(max_num_keypoints=512, detection_threshold=0.005).to(points.device).eval()
    keypoints = aliked_extractor({"image": images_tensor[[0], ref_id].to(points.device)})["keypoints"]

    qry_ids = list(range(1,10))
    start_time = time.time()
    with torch.no_grad():   
        uv_proj = project_point_to_other_camera(result, keypoints, ref_id, qry_ids)
    end_time = time.time()
    print(f"🔧 Projected points in {end_time - start_time:.3f}s")


    import cv2

    img0 = images_tensor[0, ref_id].permute(1, 2, 0).cpu().numpy() * 255
    img0 = cv2.cvtColor(img0.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    for i in range(keypoints.shape[1]):
        cv2.circle(img0, (int(keypoints[0, i, 0]), int(keypoints[0, i, 1])), 3, (0, 0, 255), -1)

    # Create a list to store images for GIF
    gif_images = []
    
    for i in range(len(uv_proj)):
        uv_points = uv_proj[i][0].cpu().numpy()
        img_qry = images_tensor[0, qry_ids[i]].permute(1, 2, 0).cpu().numpy() * 255
        img_qry = cv2.cvtColor(img_qry.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Calculate camera travel distance from reference pose
        ref_pose = result["camera_poses"][0, ref_id, :3, 3]  # Reference camera position
        qry_pose = result["camera_poses"][0, qry_ids[i], :3, 3]  # Query camera position
        travel_distance = torch.norm(qry_pose - ref_pose).item()  # Distance in meters
        
        # Draw projected points on query image with color coding
        valid_points = 0
        for j in range(uv_points.shape[0]):
            point = uv_points[j,:]
            # Check if point is within image bounds
            if 0 <= point[0] < img_qry.shape[1] and 0 <= point[1] < img_qry.shape[0]:
                cv2.circle(img_qry, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)  # Green for valid points
                cv2.circle(img_qry, (int(point[0]), int(point[1])), 7, (0, 0, 255), 2)   # Red border
                valid_points += 1
            else:
                # Draw outside image bounds (smaller, different color)
                cv2.circle(img_qry, (max(0, min(int(point[0]), img_qry.shape[1]-1)), 
                                   max(0, min(int(point[1]), img_qry.shape[0]-1))), 3, (128, 128, 128), -1)

        # Add text labels with travel distance
        cv2.putText(img0, f"Reference Camera {ref_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_qry, f"Query Camera {qry_ids[i]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add travel distance in meters
        cv2.putText(img_qry, f"Distance: {travel_distance:.2f}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Add point count
        cv2.putText(img_qry, f"Valid: {valid_points}/{len(uv_points)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        img_plot = np.concatenate([img0, img_qry], axis=1)
        
        # Convert BGR to RGB for PIL
        img_plot_rgb = cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB)
        gif_images.append(Image.fromarray(img_plot_rgb))
        
        # Also save individual frames
        cv2.imwrite(f"frame_{i:03d}.png", img_plot)
    
    # Create GIF
    if len(gif_images) > 0:
        gif_path = "point_projection_animation.gif"
        
        # Create GIF with optimized settings
        gif_images[0].save(
            gif_path,
            save_all=True,
            append_images=gif_images[1:],
            duration=800,  # 800ms per frame for better viewing
            loop=0,  # Loop indefinitely
            optimize=True,  # Optimize for smaller file size
            quality=85  # Good quality while keeping file size reasonable
        )
        
        print(f"🎬 GIF saved as: {gif_path}")
        print(f"📊 Created {len(gif_images)} frames")
        print(f"📁 Individual frames saved as frame_XXX.png")
        
        # Also create a smaller preview GIF
        if len(gif_images) > 4:
            # Take every other frame for a faster preview
            preview_images = gif_images[::2]
            preview_path = "point_projection_preview.gif"
            preview_images[0].save(
                preview_path,
                save_all=True,
                append_images=preview_images[1:],
                duration=400,  # Faster preview
                loop=0,
                optimize=True,
                quality=80
            )
            print(f"🎬 Preview GIF saved as: {preview_path}")

    import pytheia as pt
    recon = pt.sfm.Reconstruction()
    from pi3.utils.camera import Camera
    

    # Sample 150 points from each image and project them to the others to generate correspondences
    import random

    keypoints = aliked_extractor({"image": images_tensor[0,:].to(points.device)})["keypoints"]

    # Add cameras to reconstruction
    vid_recs = []
    for vid in range(images_tensor.shape[1]):
        cam_intr_id = vid
        timestamp_ns = 0 + vid
        vid_rec = recon.AddView(str(timestamp_ns), cam_intr_id, timestamp_ns)
        v = recon.MutableView(vid_rec)

        cam = Camera()
        cam.create_from_intrinsics(intrinsics[0,vid].cpu().numpy(), original_width, original_height, 1.0)
        c = v.MutableCamera()
        v.SetCameraIntrinsicsPrior(cam.prior)
        c.SetPosition(result["camera_poses"][0,vid,:3,3].cpu().numpy())
        c.SetOrientationFromRotationMatrix(result["camera_poses"][0,i,:3,:3].cpu().numpy().T)
        v.SetIsEstimated(True)
        vid_recs.append(vid_rec)

    pt.sfm.SetCameraIntrinsicsFromPriors(recon)

    for vid in range(images_tensor.shape[1]):
        with torch.no_grad():
            p_cam_ref_world = interpolate_world_points_for_ref(result, keypoints[[vid]], vid).squeeze(0).T.cpu().numpy()
            color = interpolate_color_for_ref(images_tensor[0,[vid]], keypoints[[vid]])
        # extract keypoints from image
        t_ids = []
        keypoint_in_frame = keypoints[vid].detach().cpu().numpy()
        for i, point in enumerate(keypoint_in_frame):
            t_id = recon.AddTrack()
            track = recon.MutableTrack(t_id)
            track.SetPoint(p_cam_ref_world[i,:])
            # track.SetReferenceViewId(vid_rec)
            track.SetColor(color[:,i])
            track.SetIsEstimated(True)

            t_ids.append(t_id)

            recon.AddObservation(vid_recs[vid], t_id, pt.sfm.Feature(point))

        next_vids = [j for j in range(vid+1, vid+6)]
        # make sure we do not have vids > images_tensor.shape[1]
        next_vids = [j for j in next_vids if j < images_tensor.shape[1]]
        with torch.no_grad():
            uv_proj = project_point_to_other_camera(result, keypoints[[vid]], vid, next_vids)

        # and now add observations for these tracks
        for id in range(len(uv_proj)):
            uvs = uv_proj[id][0].detach().cpu().numpy()
            next_vid = next_vids[id]
            for idx, t_id in enumerate(t_ids):
                projected_pt = uvs[idx]
                if projected_pt[0] < 0 or projected_pt[0] > original_width or projected_pt[1] < 0 or projected_pt[1] > original_height:
                    continue
                recon.AddObservation(vid_recs[next_vid], t_id, pt.sfm.Feature(projected_pt))

    # Add points to reconstruction

    # opts = pt.sfm.BundleAdjustmentOptions()
    # opts.robust_loss_width = 1.345
    # opts.verbose = True
    # opts.loss_function_type = pt.sfm.LossFunctionType.HUBER
    # opts.use_position_priors = False
    # # opts.use_gravity_priors = True
    # opts.use_homogeneous_point_parametrization = True
    # opts.use_inverse_depth_parametrization = False
    # estimated_views = sorted(pt.sfm.GetEstimatedViewsFromReconstruction(recon))
    # const_views = [estimated_views[0]]
    # var_views = [vid for vid in estimated_views if vid not in const_views]
    # #fix the first and last view
    # #otherwise the reconstruction might drift away after setting it to the origin
    # ba_sum = pt.sfm.BundleAdjustPartialViewsConstant(opts, var_views, const_views, recon)
    pt.io.WritePlyFile("reconstruction_pcl.ply", recon, np.random.randint(0,255, (3)).tolist(),1)
    pt.io.WriteReconstruction("reconstruction.sfm", recon)

    return {
        'focal': focal,
        'shift': shift,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'intrinsics': intrinsics,
        'aspect_ratio': aspect_ratio,
        'original_height': original_height,
        'original_width': original_width
    }


def main():
    """Main function to test multiview alignment with intrinsics recovery."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load image chunk and run PI3 reconstruction with intrinsics recovery")
    parser.add_argument("--folder_path", help="Path to folder containing images", 
                        default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/undist_reduced/")
    parser.add_argument("--chunk_size", type=int, default=21, help="Number of images in chunk (default: 10)")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index for chunk (default: 0)")
    parser.add_argument("--device", default="cuda", help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--conf-threshold", type=float, default=0.1, help="Confidence threshold for point filtering")
    
    args = parser.parse_args()
    
    # Check if folder exists
    if not os.path.exists(args.folder_path):
        print(f"❌ Folder not found: {args.folder_path}")
        return
    
    print("🚀 Pi3 Multiview Alignment with Intrinsics Recovery")
    print("=" * 60)
    print(f"📁 Folder: {args.folder_path}")
    print(f"📊 Chunk size: {args.chunk_size}")
    print(f"📊 Start index: {args.start_idx}")
    print(f"🖥️  Device: {args.device}")
    print(f"🎯 Confidence threshold: {args.conf_threshold}")
    
    # Load image chunk
    print("\n📷 Loading image chunk...")
    try:
        image_paths = load_image_chunk_from_folder(
            args.folder_path, args.chunk_size, args.start_idx
        )
        
        if len(image_paths) == 0:
            print("❌ No images found in folder")
            return
            
        print(f"✅ Loaded {len(image_paths)} images")
        
    except Exception as e:
        print(f"❌ Failed to load images: {e}")
        return
    
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
    
    # Run PI3 reconstruction
    print("\n🔄 Running PI3 reconstruction...")
    try:
        result, images_tensor = run_pi3_reconstruction(model, image_paths, args.device)
        print("✅ Reconstruction completed")
        
    except Exception as e:
        print(f"❌ Reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Recover intrinsics
    print("\n🔧 Recovering intrinsics...")
    try:
        intrinsics_result = recover_intrinsics_from_points(result, images_tensor)
        print("✅ Intrinsics recovery completed")
        
    except Exception as e:
        print(f"❌ Intrinsics recovery failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()

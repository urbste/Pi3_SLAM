#!/usr/bin/env python3
"""
Script to create two overlapping chunks of 20 cameras each with 10 cameras overlap,
and save both chunks as PyTheia reconstructions with consistent camera naming.
"""

import torch
import numpy as np
import sys
import os
import glob
from PIL import Image
import time
from typing import List, Tuple, Dict
import pytheia as pt

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pi3.models.pi3 import Pi3
from utils.image_utils import calculate_target_size, load_images_from_paths
from utils.geometry_torch import recover_focal_shift
from pi3.utils.camera import Camera

# Global model instances - loaded once and reused
pi3_model = None
moge_model = None
aliked_extractor = None


def load_image_chunk_from_folder(folder_path: str, chunk_size: int = 20, start_idx: int = 0) -> List[str]:
    """
    Load a single chunk of images from a folder.
    
    Args:
        folder_path: Path to the folder containing images
        chunk_size: Number of images to load (default: 20)
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


def run_pi3_reconstruction(image_paths: List[str], device: str = 'cuda') -> Tuple[dict, torch.Tensor]:
    """
    Run PI3 reconstruction on a chunk of images.
    
    Args:
        image_paths: List of image paths
        device: Device to run inference on
    
    Returns:
        Tuple of (reconstruction result, images tensor)
    """
    print(f"🤖 Running PI3 reconstruction on {len(image_paths)} images...")
    
    # Calculate target size for memory efficiency
    target_size = 518
    
    # Load images
    start_time = time.time()
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
            result = pi3_model(images_tensor.to(device))
    
    inference_time = time.time() - inference_start
    print(f"⏱️  Inference time: {inference_time:.3f}s")
    
    return result, images_tensor


def recover_intrinsics_and_scale(result: dict, images_tensor: torch.Tensor, device: str = 'cuda') -> dict:
    """
    Recover intrinsics from the point map and scale the reconstruction using MoGe.
    
    Args:
        result: PI3 reconstruction result dictionary
        images_tensor: Images tensor
        device: Device to run inference on
    
    Returns:
        Dictionary containing recovered intrinsics and scaled reconstruction
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

    def get_scale_factor_for_pi3(moge_metric_depth, pi3_metric_depth, mask):
        moge_metric_depth = moge_metric_depth[mask]
        pi3_metric_depth = pi3_metric_depth[mask]
        scale_factor = moge_metric_depth / pi3_metric_depth
        scale_factor = scale_factor.median()
        return scale_factor

    # Get scale factor from first image using global MoGe model
    moge_model.eval()
    with torch.no_grad():
        moge_metric_depth = moge_model.infer(images_tensor[[0], 0].to(device))["depth"]

    scale_factor = get_scale_factor_for_pi3(moge_metric_depth[0], result["local_points"][0, 0, :, :, 2], masks[0, 0, :, :])
    print(f"📐 Scale factor: {scale_factor}")

    # Scale the reconstruction
    result["local_points"] = result["local_points"] * scale_factor
    result["points"] = result["points"] * scale_factor
    result["camera_poses"][:,:,:3,3] = result["camera_poses"][:,:,:3,3] * scale_factor
    result["camera_poses_cw"] = torch.linalg.inv(result["camera_poses"])
    result["shift"] = shift * scale_factor
    result["intrinsics"] = intrinsics
    result["original_width"] = original_width
    result["original_height"] = original_height

    return result


def create_pytheia_reconstruction(result: dict, images_tensor: torch.Tensor, 
                                 chunk_start_idx: int, chunk_name: str) -> pt.sfm.Reconstruction:
    """
    Create a PyTheia reconstruction from PI3 results.
    
    Args:
        result: PI3 reconstruction result dictionary
        images_tensor: Images tensor
        chunk_start_idx: Starting index of this chunk in the global sequence
        chunk_name: Name for this chunk (for output files)
    
    Returns:
        PyTheia reconstruction object
    """
    print(f"🏗️  Creating PyTheia reconstruction for {chunk_name}...")
    
    recon = pt.sfm.Reconstruction()
    
    # Helper functions from the original code
    def interpolate_color_for_ref(images, keypoints):
        H, W = images.shape[-2:]
        grid_x = (keypoints[:, :, 0] / (W - 1)) * 2 - 1
        grid_y = (keypoints[:, :, 1] / (H - 1)) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        color = torch.nn.functional.grid_sample(
            images.to(keypoints.device), grid,
            mode="bilinear", align_corners=False, padding_mode="border")
        return (color.squeeze()*255).detach().cpu().numpy().astype(np.uint8)

    def interpolate_world_points_for_ref(result, keypoints, idx_ref):
        from pi3.utils.geometry import homogenize_points
        p_cam_ref = torch.linalg.inv(result["intrinsics"][:,idx_ref]) @ homogenize_points(keypoints.float()).transpose(-2,-1)
        
        depth_map = result["local_points"][0, idx_ref, :, :, 2]
        
        H, W = depth_map.shape
        grid_x = (keypoints[:, :, 0] / (W - 1)) * 2 - 1
        grid_y = (keypoints[:, :, 1] / (H - 1)) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        depth_map_unsqueezed = depth_map.unsqueeze(0).unsqueeze(0)
        z_in_cam_ref = torch.nn.functional.grid_sample(
            depth_map_unsqueezed, grid, mode="bilinear", align_corners=False, padding_mode="border"
        ).squeeze()
        
        z_in_cam_ref = z_in_cam_ref + result["shift"][0, idx_ref]
        p_cam_ref_3d = p_cam_ref * z_in_cam_ref.unsqueeze(0)
        p_cam_ref_world = result["camera_poses"][0, idx_ref] @ homogenize_points(p_cam_ref_3d.transpose(-2,-1)).transpose(-2,-1)
        return p_cam_ref_world

    def project_point_to_other_camera(result, pts_2d_cam_ref, idx_ref, idx_proj=[]):
        from utils.geometry_torch import project_xyz_to_uv
        p_cam_ref_world = interpolate_world_points_for_ref(result, pts_2d_cam_ref, idx_ref)
        uv_proj = []
        for idx in idx_proj: 
            p_cam_proj = result["camera_poses_cw"][0, idx] @ p_cam_ref_world
            uv_proj1 = project_xyz_to_uv(p_cam_proj.transpose(-2,-1), torch.tensor([0.0], device=result["local_points"].device), result["intrinsics"][0, idx])[0]
            uv_proj.append(uv_proj1)
        return uv_proj

    # Extract keypoints using global ALIKED model
    keypoints = aliked_extractor({"image": images_tensor[0,:].to(result["local_points"].device)})["keypoints"]

    # Add cameras to reconstruction
    vid_recs = []
    for vid in range(images_tensor.shape[1]):
        # Use global camera index for consistent naming across chunks
        global_cam_id = chunk_start_idx + vid
        cam_intr_id = vid
        timestamp_ns = global_cam_id
        vid_rec = recon.AddView(str(timestamp_ns), cam_intr_id, timestamp_ns)
        v = recon.MutableView(vid_rec)

        cam = Camera()
        cam.create_from_intrinsics(result["intrinsics"][0,vid].cpu().numpy(), 
                                 result["original_width"], result["original_height"], 1.0)
        c = v.MutableCamera()
        v.SetCameraIntrinsicsPrior(cam.prior)
        c.SetPosition(result["camera_poses"][0,vid,:3,3].cpu().numpy())
        c.SetOrientationFromRotationMatrix(result["camera_poses"][0,vid,:3,:3].cpu().numpy().T)
        v.SetIsEstimated(True)
        vid_recs.append(vid_rec)

    pt.sfm.SetCameraIntrinsicsFromPriors(recon)

    # Add tracks and observations
    for vid in range(images_tensor.shape[1]):
        with torch.no_grad():
            p_cam_ref_world = interpolate_world_points_for_ref(result, keypoints[[vid]], vid).squeeze(0).T.cpu().numpy()
            color = interpolate_color_for_ref(images_tensor[0,[vid]], keypoints[[vid]])
        
        t_ids = []
        keypoint_in_frame = keypoints[vid].detach().cpu().numpy()
        for i, point in enumerate(keypoint_in_frame):
            t_id = recon.AddTrack()
            track = recon.MutableTrack(t_id)
            track.SetPoint(p_cam_ref_world[i,:])
            track.SetColor(color[:,i])
            track.SetIsEstimated(True)
            t_ids.append(t_id)
            recon.AddObservation(vid_recs[vid], t_id, pt.sfm.Feature(point))

        # Project to next 5 cameras
        next_vids = [j for j in range(vid+1, vid+6)]
        next_vids = [j for j in next_vids if j < images_tensor.shape[1]]
        with torch.no_grad():
            uv_proj = project_point_to_other_camera(result, keypoints[[vid]], vid, next_vids)

        # Add observations for projected points
        for id in range(len(uv_proj)):
            uvs = uv_proj[id][0].detach().cpu().numpy()
            next_vid = next_vids[id]
            for idx, t_id in enumerate(t_ids):
                projected_pt = uvs[idx]
                if (projected_pt[0] < 0 or projected_pt[0] > result["original_width"] or 
                    projected_pt[1] < 0 or projected_pt[1] > result["original_height"]):
                    continue
                recon.AddObservation(vid_recs[next_vid], t_id, pt.sfm.Feature(projected_pt))

    # Save reconstruction
    ply_filename = f"reconstruction_{chunk_name}.ply"
    sfm_filename = f"reconstruction_{chunk_name}.sfm"
    
    pt.io.WritePlyFile(ply_filename, recon, np.random.randint(0,255, (3)).tolist(), 1)
    pt.io.WriteReconstruction(recon, sfm_filename)
    
    print(f"💾 Saved PyTheia reconstruction: {sfm_filename}")
    print(f"💾 Saved point cloud: {ply_filename}")
    
    return recon


def main():
    """Main function to process two overlapping chunks and create PyTheia reconstructions."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create two overlapping chunks and save as PyTheia reconstructions")
    parser.add_argument("--folder_path", help="Path to folder containing images", 
                        default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/undist_reduced/")
    parser.add_argument("--chunk_size", type=int, default=20, help="Number of images per chunk (default: 20)")
    parser.add_argument("--overlap", type=int, default=10, help="Number of overlapping images (default: 10)")
    parser.add_argument("--device", default="cuda", help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--conf-threshold", type=float, default=0.1, help="Confidence threshold for point filtering")
    
    args = parser.parse_args()
    
    # Check if folder exists
    if not os.path.exists(args.folder_path):
        print(f"❌ Folder not found: {args.folder_path}")
        return
    
    print("🚀 Pi3 Two-Chunk Processing with PyTheia Export")
    print("=" * 60)
    print(f"📁 Folder: {args.folder_path}")
    print(f"📊 Chunk size: {args.chunk_size}")
    print(f"📊 Overlap: {args.overlap}")
    print(f"🖥️  Device: {args.device}")
    print(f"🎯 Confidence threshold: {args.conf_threshold}")
    
    # Calculate chunk indices
    chunk1_start = 0
    chunk1_end = args.chunk_size
    chunk2_start = args.chunk_size - args.overlap
    chunk2_end = chunk2_start + args.chunk_size
    
    print(f"\n📋 Chunk configuration:")
    print(f"   Chunk 1: images {chunk1_start}-{chunk1_end-1} (20 images)")
    print(f"   Chunk 2: images {chunk2_start}-{chunk2_end-1} (20 images)")
    print(f"   Overlap: images {chunk2_start}-{chunk1_end-1} (10 images)")
    
    # Load global models
    print("\n🤖 Loading global models...")
    try:
        global pi3_model, moge_model, aliked_extractor
        
        # Load Pi3 model
        print("   Loading Pi3 model...")
        pi3_model = Pi3.from_pretrained("yyfz233/Pi3")
        pi3_model.to(args.device)
        pi3_model.eval()
        
        # Load MoGe model
        print("   Loading MoGe model...")
        from moge.model.v2 import MoGeModel
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(args.device).eval()
        
        # Load ALIKED extractor
        print("   Loading ALIKED extractor...")
        from lightglue import ALIKED
        aliked_extractor = ALIKED(max_num_keypoints=512, detection_threshold=0.005).to(args.device).eval()
        
        print("✅ All models loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return
    
    # Process Chunk 1
    print(f"\n🔄 Processing Chunk 1 (images {chunk1_start}-{chunk1_end-1})...")
    try:
        # Load images for chunk 1
        image_paths_chunk1 = load_image_chunk_from_folder(
            args.folder_path, args.chunk_size, chunk1_start
        )
        
        if len(image_paths_chunk1) == 0:
            print("❌ No images found for chunk 1")
            return
            
        # Run PI3 reconstruction
        result1, images_tensor1 = run_pi3_reconstruction(image_paths_chunk1, args.device)
        
        # Recover intrinsics and scale
        result1 = recover_intrinsics_and_scale(result1, images_tensor1, args.device)
        
        # Create PyTheia reconstruction
        recon1 = create_pytheia_reconstruction(result1, images_tensor1, chunk1_start, "chunk1")
        
        print("✅ Chunk 1 processing completed")
        
    except Exception as e:
        print(f"❌ Chunk 1 processing failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process Chunk 2
    print(f"\n🔄 Processing Chunk 2 (images {chunk2_start}-{chunk2_end-1})...")
    try:
        # Load images for chunk 2
        image_paths_chunk2 = load_image_chunk_from_folder(
            args.folder_path, args.chunk_size, chunk2_start
        )
        
        if len(image_paths_chunk2) == 0:
            print("❌ No images found for chunk 2")
            return
            
        # Run PI3 reconstruction
        result2, images_tensor2 = run_pi3_reconstruction(image_paths_chunk2, args.device)
        
        # Recover intrinsics and scale
        intrinsics_result2 = recover_intrinsics_and_scale(result2, images_tensor2, args.device)
        
        # Create PyTheia reconstruction
        recon2 = create_pytheia_reconstruction(result2, images_tensor2, chunk2_start, "chunk2")
        
        print("✅ Chunk 2 processing completed")
        
    except Exception as e:
        print(f"❌ Chunk 2 processing failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
#
    # # find the same features in the two reconstructions
    # overlap = 10

    # # get same features in first overlapping camera
    # id0 = chunk1_end - overlap
    # id1 = chunk2_start -overlap
    # features_cam0 = []
    # tids1 = sorted(recon1.View(id0).TrackIds())
    # for td in tids1:
    #     features_cam0.append(recon1.View(id0).GetFeature(td).point.tolist())
    
    # features_cam1 = []
    # tids2 = sorted(recon2.View(id1).TrackIds())
    # for td in tids2:
    #     features_cam1.append(recon2.View(id1).GetFeature(td).point.tolist())
    
    # # find the same features by calculating the intersection as they should be identical
    # features_cam0 = set(tuple(f) for f in features_cam0)
    # features_cam1 = set(tuple(f) for f in features_cam1)
    # same_features = features_cam0.intersection(features_cam1)
    # print(f"🔍 Found {len(same_features)} same features")



    print(f"\n🎉 Processing completed successfully!")
    print(f"📁 Output files:")
    print(f"   - reconstruction_chunk1.sfm (PyTheia reconstruction for chunk 1)")
    print(f"   - reconstruction_chunk1.ply (Point cloud for chunk 1)")
    print(f"   - reconstruction_chunk2.sfm (PyTheia reconstruction for chunk 2)")
    print(f"   - reconstruction_chunk2.ply (Point cloud for chunk 2)")
    print(f"\n📊 Camera naming:")
    print(f"   - Chunk 1 cameras: {chunk1_start}-{chunk1_end-1}")
    print(f"   - Chunk 2 cameras: {chunk2_start}-{chunk2_end-1}")
    print(f"   - Overlapping cameras ({chunk2_start}-{chunk1_end-1}) have consistent names across both reconstructions")


if __name__ == "__main__":
    main()

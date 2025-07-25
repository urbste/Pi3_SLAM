import torch
import argparse
import os
import glob
import time
from pi3.models.pi3 import Pi3
from pi3_slam_online_rerun_class import Pi3SLAMOnlineRerun


def main():
    """Example usage of Pi3SLAM Online with Rerun visualization."""
    parser = argparse.ArgumentParser(description="Run Pi3SLAM Online with Rerun visualization.")
    
    parser.add_argument("--data_path", type=str, default='/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/undist',
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to the dataset directory (alternative to data_path for EuroC datasets)")
    parser.add_argument("--save_path", type=str, default='examples/result_slam_rerun.ply',
                        help="Path to save the output .ply file.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for saving results (for EuroC evaluation)")
    parser.add_argument("--save_tum", action="store_true",
                        help="Save trajectory in TUM format for evaluation")
    parser.add_argument("--calib_file", type=str, default=None,
                        help="Path to camera calibration file (alternative to cam_dist_path)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
    parser.add_argument("--chunk_length", type=int, default=100,
                        help="Number of frames per chunk")
    parser.add_argument("--overlap", type=int, default=10,
                        help="Number of overlapping frames between chunks")
    parser.add_argument("--max_points", type=int, default=1000000,
                        help="Maximum number of points to save")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                        help="Confidence threshold for filtering points (default: 0.5)")
    parser.add_argument("--max_points_vis", type=int, default=10000,
                        help="Maximum number of points to show in visualization (default: 10000)")
    parser.add_argument("--rerun_port", type=int, default=9090,
                        help="Port for Rerun server (default: 9090)")
    parser.add_argument("--no_visualization", action="store_true",
                        help="Disable real-time visualization")
    parser.add_argument("--cam_dist_path", type=str, default=None,
                        help="Path to distorted camera calibration JSON file (undistorted camera will be created automatically)")
    parser.add_argument("--cam_scale", type=float, default=1.0,
                        help="Scaling factor for camera parameters")
    parser.add_argument("--enable_disk_cache", action="store_true",
                        help="Enable disk caching to reduce memory usage")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory for disk cache (default: temporary directory)")
    parser.add_argument("--max_chunks_in_memory", type=int, default=10,
                        help="Maximum number of chunks to keep in memory (default: 10)")
    parser.add_argument("--vis_subsample_ratio", type=float, default=0.1,
                        help="Ratio of points to show in visualization (default: 0.1 = 10%%)")
    
    args = parser.parse_args()
    
    # Handle dataset path vs data path
    if args.dataset_path is not None:
        data_path = args.dataset_path
        # For EuroC datasets, look for cam0 images
        if os.path.exists(os.path.join(args.dataset_path, "mav0", "cam0", "data")):
            data_path = os.path.join(args.dataset_path, "mav0", "cam0", "data")
        elif os.path.exists(os.path.join(args.dataset_path, "cam0", "data")):
            data_path = os.path.join(args.dataset_path, "cam0", "data")
    else:
        data_path = args.data_path
    
    # Handle output directory
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, "trajectory.ply")
        tum_path = os.path.join(args.output_dir, "trajectory.tum")
    else:
        save_path = args.save_path
        tum_path = args.save_path.replace('.ply', '.tum')
    
    # Handle calibration file
    calib_path = args.calib_file if args.calib_file is not None else args.cam_dist_path
    
    print("🚀 Pi3SLAM Online with Rerun - Real-time Processing and Visualization")
    print(f"📁 Data path: {data_path}")
    print(f"💾 Save path: {save_path}")
    if args.save_tum:
        print(f"📊 TUM trajectory: {tum_path}")
    print(f"🔧 Chunk length: {args.chunk_length}, Overlap: {args.overlap}")
    print(f"🎯 Confidence threshold: {args.conf_threshold}")
    print(f"👁️  Visualization: {'Disabled' if args.no_visualization else 'Enabled (Rerun)'}")
    print(f"📊 Max points in visualization: {args.max_points_vis}")
    print(f"🌐 Rerun port: {args.rerun_port}")
    print(f"📷 Camera calibration: {calib_path}")
    print(f"📷 Camera scale: {args.cam_scale}")
    print(f"💾 Disk caching: {'Enabled' if args.enable_disk_cache else 'Disabled'}")
    if args.enable_disk_cache:
        print(f"💾 Cache directory: {args.cache_dir or 'Temporary'}")
    print(f"🧠 Max chunks in memory: {args.max_chunks_in_memory}")
    print(f"👁️  Visualization subsample ratio: {args.vis_subsample_ratio:.1%}")
    
    # Create undistortion maps if calibration file is provided
    undistortion_maps = None
    if calib_path is not None:
        try:
            from pi3.utils.undistortion import create_undistortion_maps_from_file
            print("🔧 Creating undistortion maps...")
            undistortion_maps = create_undistortion_maps_from_file(
                calib_path, args.cam_scale
            )
            print("✅ Undistortion maps created successfully")
        except Exception as e:
            print(f"❌ Failed to create undistortion maps: {e}")
            print("Continuing without undistortion...")
    
    # Load model
    print(f"\n🤖 Loading model...")
    device = torch.device(args.device)
    
    if args.ckpt is not None:
        model = Pi3().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(weight)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    
    print("✅ Model loaded successfully")
    
    # Create Pi3SLAM Online Rerun instance
    slam = Pi3SLAMOnlineRerun(
        model, 
        chunk_length=args.chunk_length, 
        overlap=args.overlap, 
        device=device, 
        conf_threshold=args.conf_threshold,
        undistortion_maps=undistortion_maps
    )
    
    # Configure memory management
    slam.max_chunks_in_memory = args.max_chunks_in_memory
    slam.visualization_subsample_ratio = args.vis_subsample_ratio
    
    # Enable disk caching if requested
    if args.enable_disk_cache:
        slam.enable_disk_caching(args.cache_dir)
    
    # Start visualization if enabled
    if not args.no_visualization:
        print(f"\n🎥 Starting Rerun visualization on port {args.rerun_port}...")
        slam.start_visualization()
        
        # Check if visualization started successfully
        if slam.vis_running:
            print("✅ Rerun visualization process started successfully")
            print(f"🌐 Open http://localhost:{args.rerun_port} in your browser to view the visualization")
            time.sleep(1)  # Give visualization time to start
        else:
            print("⚠️  Rerun visualization failed to start, continuing without visualization")
            args.no_visualization = True
    
    # Get list of image files
    print(f"\n📂 Loading image sequence from: {data_path}")
    
    if data_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # For video, we'll need to extract frames
        from pi3.utils.basic import get_video_frame_count
        
        total_frames = get_video_frame_count(data_path, use_torchcodec=True)
        
        # Create frame indices
        frame_indices = list(range(0, total_frames, 1))
        image_paths = [(data_path, idx) for idx in frame_indices]  # (video_path, frame_idx)
    else:
        # For image directory, get file paths
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(data_path, ext)))
            image_paths.extend(glob.glob(os.path.join(data_path, ext.upper())))
        
        image_paths.sort()
    
    if len(image_paths) == 0:
        raise ValueError("No images found in the sequence")
    
    print(f"📸 Found {len(image_paths)} images")
    print(f"🔄 Processing in chunks of {args.chunk_length} with {args.overlap} overlap")
    
    # Calculate total number of chunks
    total_chunks = 0
    for chunk_idx in range(0, len(image_paths), args.chunk_length - args.overlap):
        end_idx = min(chunk_idx + args.chunk_length, len(image_paths))
        if end_idx - chunk_idx >= 2:
            total_chunks += 1
    
    print(f"📊 Total chunks to process: {total_chunks}")
    
    # Start background image loader
    print(f"\n🔄 Starting background image loader...")
    slam.start_background_loader(image_paths)
    
    # Process chunks using background loader
    print(f"\n🔄 Starting chunk-based processing...")
    print("=" * 60)
    
    try:
        # Use the new chunk-based processing method
        results = slam.process_chunks_with_background_loader()
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        raise
    finally:
        # Print detailed timing summary
        slam.print_timing_summary()
        
        # Save final result
        print(f"\n💾 Saving final result...")
        slam.save_final_result(save_path, max_points=args.max_points)
        
        # Save TUM format trajectory if requested
        if args.save_tum:
            print(f"\n📊 Saving TUM format trajectory...")
            slam.save_trajectory_tum(tum_path)
        
        # Stop background loader and visualization
        print(f"\n🔄 Stopping background image loader...")
        slam.stop_background_loader()
        
        if not args.no_visualization:
            print(f"\n🎥 Stopping Rerun visualization...")
            slam.stop_visualization()
        
        # Clean up disk cache
        if args.enable_disk_cache:
            print(f"\n🧹 Cleaning up disk cache...")
            slam.cleanup_disk_cache()
        
        print(f"\n✅ Processing completed successfully!")
        print(f"📁 Final result saved to: {save_path}")
        if args.save_tum:
            print(f"📊 TUM trajectory saved to: {tum_path}")


if __name__ == '__main__':
    main() 
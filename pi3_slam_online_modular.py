"""
Pi3SLAM Online with Interactive 3D Visualization - Enhanced Modular Version.
Real-time SLAM processing with Viser-based 3D visualization.
"""

import argparse
import os
from typing import List

from pi3.models.pi3 import Pi3
from slam import Pi3SLAMOnline
from utils import create_undistortion_maps


def load_image_paths(image_dir: str, video_path: str = None, start_frame: int = 0, end_frame: int = None, 
                    skip_start: int = 0, skip_end: int = 0) -> List:
    """
    Load image paths from directory or video file with frame skipping.
    
    Args:
        image_dir: Directory containing images (if not using video)
        video_path: Path to video file (if using video)
        start_frame: Starting frame index for video
        end_frame: Ending frame index for video (None for all frames)
        skip_start: Number of frames to skip from the beginning
        skip_end: Number of frames to skip from the end
    
    Returns:
        List of image paths or (video_path, frame_idx) tuples
    """
    if video_path and os.path.exists(video_path):
        # Video mode
        print(f"🎬 Loading video frames from: {video_path}")
        
        # Get video metadata to determine frame count
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if end_frame is None:
                end_frame = total_frames
            
            # Apply frame skipping
            effective_start = start_frame + skip_start
            effective_end = end_frame - skip_end
            
            # Ensure valid range
            if effective_start >= effective_end:
                print(f"❌ Error: Invalid frame range after skipping. Start: {effective_start}, End: {effective_end}")
                return []
            
            if effective_start >= total_frames:
                print(f"❌ Error: Start frame {effective_start} exceeds total frames {total_frames}")
                return []
            
            if effective_end > total_frames:
                print(f"⚠️  Warning: End frame {effective_end} exceeds total frames {total_frames}, using {total_frames}")
                effective_end = total_frames
            
            # Create list of (video_path, frame_idx) tuples
            image_paths = [(video_path, i) for i in range(effective_start, effective_end)]
            print(f"   📊 Loaded {len(image_paths)} frames ({effective_start}-{effective_end-1})")
            print(f"   ⏭️  Skipped {skip_start} frames from start, {skip_end} frames from end")
            
        except Exception as e:
            print(f"❌ Error loading video: {e}")
            return []
    else:
        # Image directory mode
        print(f"📷 Loading images from directory: {image_dir}")
        
        if not os.path.exists(image_dir):
            print(f"❌ Image directory not found: {image_dir}")
            return []
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_paths = []
        
        for filename in sorted(os.listdir(image_dir)):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(image_dir, filename))
        
        # Apply frame skipping
        total_images = len(image_paths)
        effective_start = skip_start
        effective_end = total_images - skip_end
        
        # Ensure valid range
        if effective_start >= effective_end:
            print(f"❌ Error: Invalid frame range after skipping. Start: {effective_start}, End: {effective_end}")
            return []
        
        if effective_start >= total_images:
            print(f"❌ Error: Start frame {effective_start} exceeds total images {total_images}")
            return []
        
        if effective_end > total_images:
            print(f"⚠️  Warning: End frame {effective_end} exceeds total images {total_images}, using {total_images}")
            effective_end = total_images
        
        # Apply frame range
        image_paths = image_paths[effective_start:effective_end]
        print(f"   📊 Loaded {len(image_paths)} images ({effective_start}-{effective_end-1})")
        print(f"   ⏭️  Skipped {skip_start} frames from start, {skip_end} frames from end")
    
    return image_paths


def main():
    """Main function for Pi3SLAM Online with interactive 3D visualization."""
    parser = argparse.ArgumentParser(
        description='Pi3SLAM Online with Interactive 3D Visualization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    input_group = parser.add_argument_group('📥 Input Options')
    input_group.add_argument('--image_dir', type=str, help='Directory containing images', 
                        default='/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/undist/')
    input_group.add_argument('--video_path', type=str, help='Path to video file')
    input_group.add_argument('--start_frame', type=int, default=0, help='Starting frame for video')
    input_group.add_argument('--end_frame', type=int, default=None, help='Ending frame for video')
    input_group.add_argument('--skip_start', type=int, default=0, help='Number of frames to skip from the beginning')
    input_group.add_argument('--skip_end', type=int, default=0, help='Number of frames to skip from the end')
    # Model options
    model_group = parser.add_argument_group('🤖 Model Options')
    model_group.add_argument('--model_path', type=str, default="yyfz233/Pi3", help='Path to Pi3 model checkpoint')
    model_group.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda/cpu)')
    
    # Processing options
    proc_group = parser.add_argument_group('⚙️ Processing Options')
    proc_group.add_argument('--chunk_length', type=int, default=30, help='Number of frames per chunk')
    proc_group.add_argument('--overlap', type=int, default=5, help='Number of overlapping frames between chunks')
    proc_group.add_argument('--conf_threshold', type=float,default=0.5, help='Confidence threshold for filtering points')
    proc_group.add_argument('--cam_scale', type=float, default=1.0, help='Scale factor for camera poses')
    
    # Camera parameter estimation options
    camera_group = parser.add_argument_group('📷 Camera Options')
    camera_group.add_argument('--estimate_camera_params', action='store_true', default=True, 
                             help='Enable camera parameter estimation for each chunk')
    camera_group.add_argument('--cam_dist_path', type=str, help='Path to camera calibration file for undistortion')
    
    # Keypoint extraction options
    keypoint_group = parser.add_argument_group('🎯 Keypoint Extraction')
    keypoint_group.add_argument('--keypoint_type', default="grid", help='Type of keypoint extractor to use')
    keypoint_group.add_argument('--max_num_keypoints', type=int, default=200, help='Maximum number of keypoints to extract per frame')
    keypoint_group.add_argument('--keypoint_detection_threshold', type=float, default=0.005, help='Detection threshold for keypoints')
    
    # Reconstruction options
    recon_group = parser.add_argument_group('🔧 Reconstruction Options')
    recon_group.add_argument('--save_chunk_reconstructions', action='store_true', default=False, 
                            help='Save each chunk reconstruction to disk')
    recon_group.add_argument('--save_transformed_reconstructions', action='store_true', default=False, 
                            help='Save transformed reconstructions as PLY files')
    recon_group.add_argument('--save_debug_reconstructions', action='store_true', default=False, 
                            help='Save debug files for all reconstruction stages')
    recon_group.add_argument('--save_debug_projections', action='store_true', default=False, 
                            help='Save debug projections as GIF files')
    recon_group.add_argument('--max_observations_per_track', type=int, default=6, 
                            help='Maximum number of observations per track')
    recon_group.add_argument('--do_metric_depth', action='store_true', default=True, 
                            help='Use MoGe model to estimate metric depth')
    recon_group.add_argument('--use_inverse_depth', action='store_true', default=False,
                            help='Use inverse-depth parametrization in reconstruction (matches offline option)')
    
    # Visualization options
    viz_group = parser.add_argument_group('🎥 Visualization Options')
    viz_group.add_argument('--viz_port', type=int, default=8080, help='Port for 3D visualization server')
    viz_group.add_argument('--no_visualization', action='store_true', help='Disable 3D visualization')
    viz_group.add_argument('--keep_viz_open', action='store_true', help='Keep visualization open after processing finishes')
    
    # Output options
    output_group = parser.add_argument_group('💾 Output Options')
    output_group.add_argument('--output_path', type=str, 
                             default='/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/undist_result', 
                             help='Output path for trajectory')
    output_group.add_argument('--max_points', type=int, default=1000000, help='Maximum points to save')
    output_group.add_argument('--save_tum', action='store_true',  help='Save trajectory in TUM format for evaluation')
    output_group.add_argument('--tum_integer_timestamp', action='store_true',
                             help='Use integer timestamps in TUM output (for 7-scenes)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("🎯 Pi3SLAM Online - Enhanced Modular Version")
    print("=" * 60)
    
    # Validate input
    if not args.image_dir and not args.video_path:
        print("❌ Error: Must specify either --image_dir or --video_path")
        return
    
    if args.image_dir and args.video_path:
        print("❌ Error: Cannot specify both --image_dir and --video_path")
        return
    
    # Validate frame skipping parameters
    if args.skip_start < 0:
        print("❌ Error: --skip_start must be non-negative")
        return
    
    if args.skip_end < 0:
        print("❌ Error: --skip_end must be non-negative")
        return
    
    # Load image paths with frame skipping
    print("\n📥 Loading input data...")
    image_paths = load_image_paths(
        args.image_dir, 
        args.video_path, 
        args.start_frame, 
        args.end_frame,
        args.skip_start,
        args.skip_end
    )
    
    if not image_paths:
        print("❌ No images found after applying frame skipping")
        return
    
    print(f"✅ Total images/frames to process: {len(image_paths):,}")
    
    # Create undistortion maps if calibration file is provided
    undistortion_maps = None
    if args.cam_dist_path:
        print(f"\n🔧 Creating undistortion maps from: {args.cam_dist_path}")
        undistortion_maps = create_undistortion_maps(args.cam_dist_path)
        if undistortion_maps:
            print("✅ Undistortion maps created successfully")
        else:
            print("⚠️  Failed to create undistortion maps, continuing without undistortion")
    else:
        print("ℹ️  No camera calibration file provided, processing without undistortion")
    
    # Load model
    print(f"\n🤖 Loading Pi3 model from: {args.model_path}")
    try:
        model = Pi3.from_pretrained(args.model_path)
        model.to(args.device)
        model.eval()
        print(f"✅ Model loaded successfully on {args.device}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Initialize Pi3SLAM Online
    print("\n🚀 Initializing Pi3SLAM Online...")
    slam = Pi3SLAMOnline(
        model=model,
        chunk_length=args.chunk_length,
        overlap=args.overlap,
        device=args.device,
        conf_threshold=args.conf_threshold,
        undistortion_maps=undistortion_maps,
        cam_scale=args.cam_scale,
        visualization_port=args.viz_port,  # Using viz_port for compatibility
        estimate_camera_params=args.estimate_camera_params,
        keypoint_type=args.keypoint_type,
        max_num_keypoints=args.max_num_keypoints,
        keypoint_detection_threshold=args.keypoint_detection_threshold,
        save_chunk_reconstructions=args.save_chunk_reconstructions,
        max_observations_per_track=args.max_observations_per_track,
        do_metric_depth=args.do_metric_depth,
        save_debug_projections=args.save_debug_projections,
        model_path=args.model_path,
        use_inverse_depth=args.use_inverse_depth,
    )
    

    if os.path.isdir(args.output_path) or args.output_path.endswith('/'):
        output_dir = args.output_path
    else:
        output_dir = os.path.dirname(args.output_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    slam.set_output_directory(output_dir)
    print(f"💾 Chunk reconstructions will be saved to: {os.path.join(output_dir, 'reconstructions')}")

    # Set reconstruction alignment parameters
    slam.save_transformed_reconstructions = args.save_transformed_reconstructions
    slam.save_debug_reconstructions = args.save_debug_reconstructions
    
    print(f"\n🔧 Configuration Summary:")
    print(f"   📊 Chunk length: {args.chunk_length}, Overlap: {args.overlap}")
    print(f"   🎯 Keypoint type: {args.keypoint_type}, Max keypoints: {args.max_num_keypoints}")
    print(f"   💾 Save reconstructions: {args.save_chunk_reconstructions}")
    print(f"   💾 Save transformed PLY: {slam.save_transformed_reconstructions}")
    print(f"   🔍 Save debug files: {slam.save_debug_reconstructions}")
    
    # Start background loader
    print("\n🔄 Starting background image loader...")
    slam.start_background_loader(image_paths)
    
    # Start visualization if enabled
    if not args.no_visualization:
        print(f"🎥 Starting interactive 3D visualization on port {args.viz_port}...")
        print(f"🌐 Open browser to: http://localhost:{args.viz_port}")
        slam.start_visualization()
    else:
        print("🚫 Visualization disabled")
    
    try:
        # Process all chunks
        print("\n🔄 Starting SLAM processing...")
        print("=" * 60)
        # Use producer/consumer processing with background inference worker
        results = slam.process_chunks_with_inference_worker(auto_close_visualization=not args.keep_viz_open)
        
        # Print final statistics
        print("\n" + "="*60)
        print("📊 FINAL STATISTICS")
        print("="*60)
        stats = slam.get_statistics()
        print(f"✅ Total chunks processed: {stats['total_chunks']}")
        print(f"✅ Total frames processed: {stats['total_frames']}")
        
        # Save results
        print(f"\n💾 Saving results to: {args.output_path}")
        
        os.makedirs(args.output_path, exist_ok=True)
        
        # Handle output path - if it's a directory, create trajectory.ply inside it
        if os.path.isdir(args.output_path):
            # It's a directory, create trajectory.ply inside it
            output_dir = args.output_path
            ply_path = os.path.join(output_dir, 'trajectory.ply')
            tum_path = os.path.join(output_dir, 'trajectory.tum')
        else:
            # It's a file path
            ply_path = args.output_path
            tum_path = args.output_path.replace('.ply', '.tum')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(ply_path), exist_ok=True)
        
        slam.save_final_result(ply_path, max_points=args.max_points)
        
        # Save TUM format trajectory if requested
        if args.save_tum:
            print(f"📊 Saving TUM format trajectory...")
            slam.save_trajectory_tum(tum_path, integer_timestamp=args.tum_integer_timestamp)
        
        print("\n🎉 Processing completed successfully!")
        
        if not args.no_visualization and args.keep_viz_open:
            print(f"\n🌐 3D visualization running at: http://localhost:{args.viz_port}")
            print("Press Ctrl+C to close, or rerun without --keep_viz_open to auto-close on finish.")
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping visualization...")
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        slam.stop_visualization()
        print("✅ Cleanup completed")


if __name__ == "__main__":
    main()
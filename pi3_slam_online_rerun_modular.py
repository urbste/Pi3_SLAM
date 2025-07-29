"""
Pi3SLAM Online with Rerun visualization - Modular version.
This is a simplified main file that uses the modular components.
"""

import argparse
import os
from typing import List

from pi3.models.pi3 import Pi3
from slam.pi3_slam_online import Pi3SLAMOnlineRerun
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
            print(f"   Loaded {len(image_paths)} frames ({effective_start}-{effective_end-1})")
            print(f"   Skipped {skip_start} frames from start, {skip_end} frames from end")
            
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
        print(f"   Loaded {len(image_paths)} images ({effective_start}-{effective_end-1})")
        print(f"   Skipped {skip_start} frames from start, {skip_end} frames from end")
    
    return image_paths


def main():
    """Main function for Pi3SLAM Online with Rerun visualization."""
    parser = argparse.ArgumentParser(description='Pi3SLAM Online with Rerun visualization')
    
    # Input options
    parser.add_argument('--image_dir', type=str, help='Directory containing images')
    parser.add_argument('--video_path', type=str, help='Path to video file')
    parser.add_argument('--start_frame', type=int, default=0, help='Starting frame for video')
    parser.add_argument('--end_frame', type=int, default=None, help='Ending frame for video')
    parser.add_argument('--skip_start', type=int, default=0, help='Number of frames to skip from the beginning')
    parser.add_argument('--skip_end', type=int, default=0, help='Number of frames to skip from the end')
    
    # Model options
    parser.add_argument('--model_path', type=str, default="yyfz233/Pi3", help='Path to Pi3 model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda/cpu)')
    
    # Processing options
    parser.add_argument('--chunk_length', type=int, default=50, help='Number of frames per chunk')
    parser.add_argument('--overlap', type=int, default=10, help='Number of overlapping frames between chunks')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='Confidence threshold for filtering points')
    parser.add_argument('--cam_scale', type=float, default=1.0, help='Scale factor for camera poses')
    
    # Alignment options
    parser.add_argument('--ransac_distance', type=float, default=0.01, help='RANSAC max correspondence distance')
    parser.add_argument('--ransac_iterations', type=int, default=100, help='RANSAC max iterations')
    parser.add_argument('--icp_threshold', type=float, default=0.01, help='ICP distance threshold')
    parser.add_argument('--icp_iterations', type=int, default=100, help='ICP max iterations')
    parser.add_argument('--no_sim3_optimization', action='store_true', help='Disable SIM3 optimization after RANSAC+ICP')
    
    # Undistortion options
    parser.add_argument('--cam_dist_path', type=str, help='Path to camera calibration file for undistortion')
    
    # Memory and caching options
    parser.add_argument('--max_chunks_in_memory', type=int, default=5, help='Maximum chunks to keep in memory')
    parser.add_argument('--enable_disk_cache', action='store_true', help='Enable disk caching for processed chunks')
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory for disk cache')
    
    # Visualization options
    parser.add_argument('--rerun_port', type=int, default=9090, help='Port for Rerun visualization')
    parser.add_argument('--no_visualization', action='store_true', help='Disable Rerun visualization')
    
    # Output options
    parser.add_argument('--output_path', type=str, default='output/trajectory.ply', help='Output path for trajectory')
    parser.add_argument('--max_points', type=int, default=1000000, help='Maximum points to save')
    parser.add_argument('--save_tum', action='store_true', help='Save trajectory in TUM format for evaluation')
    parser.add_argument('--tum_integer_timestamp', action='store_true', help='Use integer timestamps in TUM output (for 7-scenes)')
    
    args = parser.parse_args()
    
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
    
    print(f"📊 Total images/frames to process: {len(image_paths)}")
    
    # Create undistortion maps if calibration file is provided
    undistortion_maps = None
    if args.cam_dist_path:
        print(f"🔧 Creating undistortion maps from calibration file: {args.cam_dist_path}")
        undistortion_maps = create_undistortion_maps(args.cam_dist_path)
        if undistortion_maps:
            print("✅ Undistortion maps created successfully")
        else:
            print("⚠️  Failed to create undistortion maps, continuing without undistortion")
    else:
        print("ℹ️  No camera calibration file provided, processing without undistortion")
    
    # Load model
    print(f"🤖 Loading Pi3 model from: {args.model_path}")
    try:
        model = Pi3.from_pretrained(args.model_path)
        model.to(args.device)
        model.eval()
        print(f"✅ Model loaded successfully on {args.device}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Initialize Pi3SLAM Online
    print("🚀 Initializing Pi3SLAM Online...")
    slam = Pi3SLAMOnlineRerun(
        model=model,
        chunk_length=args.chunk_length,
        overlap=args.overlap,
        device=args.device,
        conf_threshold=args.conf_threshold,
        undistortion_maps=undistortion_maps,
        cam_scale=args.cam_scale,
        max_chunks_in_memory=args.max_chunks_in_memory,
        enable_disk_cache=args.enable_disk_cache,
        cache_dir=args.cache_dir,
        rerun_port=args.rerun_port,
        enable_sim3_optimization=not args.no_sim3_optimization
    )
    
    # Configure robust alignment parameters
    print("🔧 Configuring robust alignment parameters...")
    slam.configure_alignment(
        ransac_max_correspondence_distance=args.ransac_distance,
        ransac_max_iterations=args.ransac_iterations,
        icp_threshold=args.icp_threshold,
        icp_max_iterations=args.icp_iterations
    )
    
    # Start background loader
    print("🔄 Starting background image loader...")
    slam.start_background_loader(image_paths)
    
    # Start visualization if enabled
    if not args.no_visualization:
        print("🎥 Starting Rerun visualization...")
        slam.start_visualization()
    
    try:
        # Process all chunks
        print("🔄 Starting chunk processing...")
        results = slam.process_chunks_with_background_loader()
        
        # Print final statistics
        print("\n" + "="*60)
        print("📊 FINAL STATISTICS")
        print("="*60)
        stats = slam.get_statistics()
        print(f"Total chunks processed: {stats['total_chunks']}")
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Total processing time: {stats.get('total_processing_time', 0):.2f}s")
        print(f"Overall FPS: {stats.get('overall_fps', 0):.1f}")
        
        # Print alignment configuration
        print(f"\n🔧 Alignment Configuration:")
        print(f"   RANSAC distance threshold: {args.ransac_distance}")
        print(f"   RANSAC max iterations: {args.ransac_iterations}")
        print(f"   ICP distance threshold: {args.icp_threshold}")
        print(f"   ICP max iterations: {args.icp_iterations}")
        
        # Save results
        print(f"\n💾 Saving results to: {args.output_path}")
        
        # Handle output path - if it's a directory, create trajectory.ply inside it
        if os.path.isdir(args.output_path) or args.output_path.endswith('/'):
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
            print(f"\n📊 Saving TUM format trajectory...")
            slam.save_trajectory_tum(tum_path, integer_timestamp=args.tum_integer_timestamp)
        
        print("✅ Processing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        raise
    finally:
        # Cleanup
        print("🧹 Cleaning up...")
        slam.stop_visualization()
        
        if args.enable_disk_cache:
            slam.cleanup_disk_cache()


if __name__ == "__main__":
    main() 
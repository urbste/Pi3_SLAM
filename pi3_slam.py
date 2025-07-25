import torch
import argparse
from pi3.models.pi3 import Pi3
from pi3_slam_class import Pi3SLAM


def main():
    """Example usage of Pi3SLAM."""
    parser = argparse.ArgumentParser(description="Run Pi3SLAM inference on longer trajectories.")
    
    parser.add_argument("--data_path", type=str, default='/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/undist_reduced/',
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--save_path", type=str, default='examples/result_slam.ply',
                        help="Path to save the output .ply file.")
    parser.add_argument("--save_tum", action="store_true",
                        help="Save trajectory in TUM format for evaluation")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
    parser.add_argument("--chunk_length", type=int, default=10,
                        help="Number of frames per chunk")
    parser.add_argument("--overlap", type=int, default=2,
                        help="Number of overlapping frames between chunks")
    parser.add_argument("--max_points", type=int, default=1000000,
                        help="Maximum number of points to save")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                        help="Confidence threshold for filtering points (default: 0.5)")
    parser.add_argument("--subsample_factor", type=int, default=20,
                        help="Factor to subsample points (default: 4)")
    parser.add_argument("--chunk_colors", action="store_true",
                        help="Use chunk-based colors instead of image colors")
    parser.add_argument("--cam_dist_path", type=str, default=None,
                        help="Path to distorted camera calibration JSON file (undistorted camera will be created automatically)")
    parser.add_argument("--interval", type=int, default=None,
                        help="Sampling interval for frames (default: 1 for images, 10 for video)")
    
    args = parser.parse_args()
    
    # Set default interval based on file type
    if args.interval is None:
        if args.data_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            args.interval = 10
        else:
            args.interval = 1
    
    print("🚀 Pi3SLAM - Long Trajectory Reconstruction")
    print(f"📁 Data path: {args.data_path}")
    print(f"💾 Save path: {args.save_path}")
    if args.save_tum:
        tum_path = args.save_path.replace('.ply', '.tum')
        print(f"📊 TUM trajectory: {tum_path}")
    print(f"🔧 Chunk length: {args.chunk_length}, Overlap: {args.overlap}")
    print(f"📊 Sampling interval: {args.interval}")
    print(f"🎯 Confidence threshold: {args.conf_threshold}")
    print(f"📊 Max points: {args.max_points}")
    print(f"📊 Subsample factor: {args.subsample_factor}")
    print(f"🎨 Chunk colors: {args.chunk_colors}")
    print(f"📷 Camera calibration: {args.cam_dist_path}")
    print(f"💻 Device: {args.device}")
    
    # Load model
    print("\n🤖 Loading Pi3 model...")
    model = Pi3.from_pretrained("yyfz233/Pi3").to(args.device).eval()
    print("✅ Model loaded successfully!")
    
    # Handle camera calibration
    undistortion_maps = None
    if args.cam_dist_path:
        print(f"\n🔧 Loading camera calibration from: {args.cam_dist_path}")
        try:
            from pi3.utils.undistortion import create_undistortion_maps_from_file
            undistortion_maps = create_undistortion_maps_from_file(args.cam_dist_path, scale=1.0)
            print("✅ Camera calibration loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading camera calibration: {e}")
            print("Continuing without camera calibration...")
    
    # Create Pi3SLAM instance
    print(f"\n🔧 Creating Pi3SLAM instance...")
    slam = Pi3SLAM(
        model=model,
        chunk_length=args.chunk_length,
        overlap=args.overlap,
        device=args.device,
        conf_threshold=args.conf_threshold,
        undistortion_maps=undistortion_maps
    )
    print("✅ Pi3SLAM instance created!")
    
    # Process sequence
    print(f"\n🔄 Processing sequence...")
    result = slam.process_sequence(args.data_path)
    print("✅ Sequence processing completed!")
    
    # Save result with colors
    print(f"\n💾 Saving results...")
    slam.save_result(result, args.save_path, max_points=args.max_points, 
                    use_chunk_colors=args.chunk_colors,
                    conf_threshold=args.conf_threshold, subsample_factor=args.subsample_factor)
    
    # Save TUM format trajectory if requested
    if args.save_tum:
        print(f"\n📊 Saving TUM format trajectory...")
        tum_path = args.save_path.replace('.ply', '.tum')
        slam.save_trajectory_tum(tum_path, result)
    
    print("✅ Done.")


if __name__ == '__main__':
    main() 
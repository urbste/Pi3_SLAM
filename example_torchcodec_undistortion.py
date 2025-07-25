#!/usr/bin/env python3
"""
Example script demonstrating how to use torchcodec for video processing with undistortion.

This script shows how to:
1. Load video frames efficiently using torchcodec
2. Apply undistortion to video frames
3. Process video chunks with undistortion
"""

import torch
import os
import sys
import time
from pi3.utils.undistortion import create_undistortion_maps_from_file, VideoUndistortionLoader
from pi3.utils.basic import load_images_as_tensor_with_undistortion_maps


def example_torchcodec_video_loading():
    """Example of using torchcodec for video loading with undistortion."""
    print("=" * 60)
    print("Example 1: TorchCodec Video Loading with Undistortion")
    print("=" * 60)
    
    # Camera calibration file path (replace with your actual path)
    cam_dist_path = "/home/steffen/Data/GPStrava/GoproCalib/GoPro9/1080_50_wide_stab/dataset1/cam/cam_calib_GH018983_di_2.json"
    
    # Video file path (replace with your actual path)
    video_path = "/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/run1.MP4"
    
    # Check if files exist
    if not os.path.exists(cam_dist_path):
        print(f"❌ Camera calibration file not found: {cam_dist_path}")
        print("Please update the path to your actual calibration file.")
        return
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("Please update the path to your actual video file.")
        return
    
    print(f"📷 Camera calibration: {cam_dist_path}")
    print(f"🎥 Video file: {video_path}")
    
    try:
        # Create undistortion maps
        print("🔧 Creating undistortion maps...")
        undistortion_maps = create_undistortion_maps_from_file(cam_dist_path, scale=1.0)
        print("✅ Undistortion maps created successfully")
        
        # Load video with undistortion using torchcodec
        print("\n🎬 Loading video with torchcodec...")
        start_time = time.time()
        
        frames_tensor = load_images_as_tensor_with_undistortion_maps(
            path=video_path,
            interval=10,  # Load every 10th frame
            PIXEL_LIMIT=255000,
            undistortion_maps=undistortion_maps,
            use_torchcodec=True,
            device="cpu"  # or "cuda" for GPU processing
        )
        
        end_time = time.time()
        load_time = end_time - start_time
        
        print(f"✅ Video loading completed in {load_time:.2f} seconds")
        print(f"   Loaded {frames_tensor.shape[0]} frames")
        print(f"   Frame shape: {frames_tensor.shape[1:]}")
        print(f"   Loading speed: {frames_tensor.shape[0] / load_time:.1f} frames/second")
        
    except Exception as e:
        print(f"❌ Error loading video with torchcodec: {e}")
        import traceback
        traceback.print_exc()


def example_video_loader_api():
    """Example of using the VideoUndistortionLoader API directly."""
    print("\n" + "=" * 60)
    print("Example 2: Direct VideoUndistortionLoader API")
    print("=" * 60)
    
    # Camera calibration file path (replace with your actual path)
    cam_dist_path = "/home/steffen/Data/GPStrava/GoproCalib/GoPro9/1080_50_wide_stab/dataset1/cam/cam_calib_GH018983_di_2.json"
    
    # Video file path (replace with your actual path)
    video_path = "/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/run1.MP4"
    
    # Check if files exist
    if not os.path.exists(cam_dist_path) or not os.path.exists(video_path):
        print("❌ Required files not found. Skipping example.")
        return
    
    try:
        # Create undistortion maps
        print("🔧 Creating undistortion maps...")
        undistortion_maps = create_undistortion_maps_from_file(cam_dist_path, scale=1.0)
        
        # Create video loader
        print("🎬 Creating video loader...")
        video_loader = VideoUndistortionLoader(undistortion_maps, device="cpu")
        
        # Get video metadata
        print("📊 Getting video metadata...")
        metadata = video_loader.get_video_metadata(video_path)
        print(f"   Total frames: {metadata.num_frames}")
        print(f"   Duration: {metadata.duration_seconds:.2f} seconds")
        print(f"   FPS: {metadata.average_fps:.2f}")
        print(f"   Codec: {metadata.codec}")
        
        # Example 1: Load single frame
        print("\n🎥 Example 1: Loading single frame...")
        frame_idx = 100
        single_frame = video_loader.load_and_undistort_frame(video_path, frame_idx)
        print(f"   Frame {frame_idx} shape: {single_frame.shape}")
        print(f"   Value range: [{single_frame.min():.3f}, {single_frame.max():.3f}]")
        
        # Example 2: Load multiple specific frames
        print("\n🎥 Example 2: Loading multiple specific frames...")
        frame_indices = [0, 50, 100, 150, 200]
        multiple_frames = video_loader.load_and_undistort_frames(video_path, frame_indices)
        print(f"   Multiple frames shape: {multiple_frames.shape}")
        
        # Example 3: Load frame slice
        print("\n🎥 Example 3: Loading frame slice...")
        start_idx, end_idx, step = 0, 100, 10  # Frames 0, 10, 20, ..., 90
        frame_slice = video_loader.load_and_undistort_video_slice(
            video_path, start_idx, end_idx, step
        )
        print(f"   Frame slice shape: {frame_slice.shape}")
        print(f"   Expected frames: {len(range(start_idx, end_idx, step))}")
        
        # Clean up
        video_loader.close_decoder(video_path)
        print("✅ Video loader cleaned up successfully")
        
    except Exception as e:
        print(f"❌ Error in video loader example: {e}")
        import traceback
        traceback.print_exc()


def example_performance_comparison():
    """Example comparing torchcodec vs OpenCV performance."""
    print("\n" + "=" * 60)
    print("Example 3: Performance Comparison (TorchCodec vs OpenCV)")
    print("=" * 60)
    
    # Camera calibration file path (replace with your actual path)
    cam_dist_path = "/home/steffen/Data/GPStrava/GoproCalib/GoPro9/1080_50_wide_stab/dataset1/cam/cam_calib_GH018983_di_2.json"
    
    # Video file path (replace with your actual path)
    video_path = "/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/run1.MP4"
    
    # Check if files exist
    if not os.path.exists(cam_dist_path) or not os.path.exists(video_path):
        print("❌ Required files not found. Skipping performance comparison.")
        return
    
    try:
        # Create undistortion maps
        print("🔧 Creating undistortion maps...")
        undistortion_maps = create_undistortion_maps_from_file(cam_dist_path, scale=1.0)
        
        # Test parameters
        interval = 20  # Load every 20th frame
        num_frames_to_test = 50  # Limit for performance test
        
        print(f"🧪 Testing with interval={interval}, max frames={num_frames_to_test}")
        
        # Test 1: TorchCodec
        print("\n🚀 Test 1: TorchCodec")
        start_time = time.time()
        
        frames_torchcodec = load_images_as_tensor_with_undistortion_maps(
            path=video_path,
            interval=interval,
            PIXEL_LIMIT=255000,
            undistortion_maps=undistortion_maps,
            use_torchcodec=True,
            device="cpu"
        )
        
        # Limit frames for fair comparison
        if frames_torchcodec.shape[0] > num_frames_to_test:
            frames_torchcodec = frames_torchcodec[:num_frames_to_test]
        
        torchcodec_time = time.time() - start_time
        print(f"   Time: {torchcodec_time:.2f} seconds")
        print(f"   Frames: {frames_torchcodec.shape[0]}")
        print(f"   Speed: {frames_torchcodec.shape[0] / torchcodec_time:.1f} frames/second")
        
        # Test 2: OpenCV (fallback)
        print("\n🐌 Test 2: OpenCV (fallback)")
        start_time = time.time()
        
        frames_opencv = load_images_as_tensor_with_undistortion_maps(
            path=video_path,
            interval=interval,
            PIXEL_LIMIT=255000,
            undistortion_maps=undistortion_maps,
            use_torchcodec=False,
            device="cpu"
        )
        
        # Limit frames for fair comparison
        if frames_opencv.shape[0] > num_frames_to_test:
            frames_opencv = frames_opencv[:num_frames_to_test]
        
        opencv_time = time.time() - start_time
        print(f"   Time: {opencv_time:.2f} seconds")
        print(f"   Frames: {frames_opencv.shape[0]}")
        print(f"   Speed: {frames_opencv.shape[0] / opencv_time:.1f} frames/second")
        
        # Performance comparison
        if torchcodec_time > 0 and opencv_time > 0:
            speedup = opencv_time / torchcodec_time
            print(f"\n📊 Performance Summary:")
            print(f"   TorchCodec speedup: {speedup:.1f}x faster")
            print(f"   Time saved: {opencv_time - torchcodec_time:.2f} seconds")
        
        # Verify results are similar
        if frames_torchcodec.shape == frames_opencv.shape:
            diff = torch.abs(frames_torchcodec - frames_opencv).max()
            print(f"   Max difference between methods: {diff:.6f}")
            if diff < 0.01:
                print("   ✅ Results are consistent between methods")
            else:
                print("   ⚠️  Results differ between methods")
        
    except Exception as e:
        print(f"❌ Error in performance comparison: {e}")
        import traceback
        traceback.print_exc()


def example_gpu_processing():
    """Example of GPU processing with torchcodec."""
    print("\n" + "=" * 60)
    print("Example 4: GPU Processing with TorchCodec")
    print("=" * 60)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Skipping GPU example.")
        return
    
    # Camera calibration file path (replace with your actual path)
    cam_dist_path = "/home/steffen/Data/GPStrava/GoproCalib/GoPro9/1080_50_wide_stab/dataset1/cam/cam_calib_GH018983_di_2.json"
    
    # Video file path (replace with your actual path)
    video_path = "/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/run1.MP4"
    
    # Check if files exist
    if not os.path.exists(cam_dist_path) or not os.path.exists(video_path):
        print("❌ Required files not found. Skipping GPU example.")
        return
    
    try:
        # Create undistortion maps
        print("🔧 Creating undistortion maps...")
        undistortion_maps = create_undistortion_maps_from_file(cam_dist_path, scale=1.0)
        
        # Test GPU processing
        print("🚀 Testing GPU processing...")
        start_time = time.time()
        
        frames_gpu = load_images_as_tensor_with_undistortion_maps(
            path=video_path,
            interval=10,
            PIXEL_LIMIT=255000,
            undistortion_maps=undistortion_maps,
            use_torchcodec=True,
            device="cuda"
        )
        
        gpu_time = time.time() - start_time
        print(f"   GPU Time: {gpu_time:.2f} seconds")
        print(f"   Frames: {frames_gpu.shape[0]}")
        print(f"   Speed: {frames_gpu.shape[0] / gpu_time:.1f} frames/second")
        print(f"   Device: {frames_gpu.device}")
        
        # Compare with CPU
        print("\n🔄 Comparing with CPU processing...")
        start_time = time.time()
        
        frames_cpu = load_images_as_tensor_with_undistortion_maps(
            path=video_path,
            interval=10,
            PIXEL_LIMIT=255000,
            undistortion_maps=undistortion_maps,
            use_torchcodec=True,
            device="cpu"
        )
        
        cpu_time = time.time() - start_time
        print(f"   CPU Time: {cpu_time:.2f} seconds")
        print(f"   Speed: {frames_cpu.shape[0] / cpu_time:.1f} frames/second")
        
        if gpu_time > 0 and cpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"\n📊 GPU vs CPU Performance:")
            print(f"   GPU speedup: {speedup:.1f}x faster")
            print(f"   Time saved: {cpu_time - gpu_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error in GPU processing example: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all torchcodec examples."""
    print("🎬 TorchCodec Video Processing Examples")
    print("This script demonstrates efficient video processing with undistortion using torchcodec.")
    print("Please update the file paths to match your setup.\n")
    
    # Run examples
    example_torchcodec_video_loading()
    example_video_loader_api()
    example_performance_comparison()
    example_gpu_processing()
    
    print("\n" + "=" * 60)
    print("📚 Summary")
    print("=" * 60)
    print("TorchCodec integration provides:")
    print("✅ Efficient video loading with GPU support")
    print("✅ Automatic frame indexing and slicing")
    print("✅ Memory-efficient processing")
    print("✅ Seamless integration with undistortion")
    print("✅ Fallback to OpenCV when torchcodec is not available")
    print("\nInstall torchcodec with: pip install torchcodec")


if __name__ == "__main__":
    main() 
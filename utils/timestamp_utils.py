"""
Timestamp extraction utilities for Pi3SLAM.
"""

import os
import re
from typing import List
from pi3.utils.basic import TORCHCODEC_AVAILABLE


def extract_timestamps_from_paths(image_paths: List) -> List[float]:
    """
    Extract timestamps from image paths or video frames.
    
    Args:
        image_paths: List of image paths or (video_path, frame_idx) tuples
    
    Returns:
        List of timestamps in nanoseconds
    """
    timestamps = []
    
    for path_info in image_paths:
        if isinstance(path_info, tuple):
            # Video frame - calculate timestamp from frame index and video metadata
            video_path, frame_idx = path_info
            timestamp = get_video_frame_timestamp(video_path, frame_idx)
            timestamps.append(timestamp)
        else:
            # Image file - extract timestamp from filename
            timestamp = extract_timestamp_from_filename(path_info)
            timestamps.append(timestamp)
    
    return timestamps


def get_video_frame_timestamp(video_path: str, frame_idx: int) -> float:
    """
    Get timestamp for a video frame based on frame index and video metadata.
    
    Args:
        video_path: Path to video file
        frame_idx: Frame index
    
    Returns:
        Timestamp in nanoseconds
    """
    # Cache video metadata to avoid repeated loading
    if not hasattr(get_video_frame_timestamp, '_video_metadata_cache'):
        get_video_frame_timestamp._video_metadata_cache = {}
    
    if video_path not in get_video_frame_timestamp._video_metadata_cache:
        try:
            # Try to get metadata using torchcodec
            if TORCHCODEC_AVAILABLE:
                from torchcodec.decoders import VideoDecoder
                decoder = VideoDecoder(video_path, device="cpu")
                metadata = decoder.metadata
                get_video_frame_timestamp._video_metadata_cache[video_path] = {
                    'fps': metadata.average_fps,
                    'duration': metadata.duration_seconds,
                    'total_frames': metadata.num_frames
                }
                del decoder
            else:
                # Fallback to OpenCV
                import cv2
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                cap.release()
                
                get_video_frame_timestamp._video_metadata_cache[video_path] = {
                    'fps': fps,
                    'duration': duration,
                    'total_frames': total_frames
                }
        except Exception as e:
            print(f"Warning: Could not get video metadata for {video_path}: {e}")
            # Use default values
            get_video_frame_timestamp._video_metadata_cache[video_path] = {
                'fps': 30.0,
                'duration': 0.0,
                'total_frames': 0
            }
    
    metadata = get_video_frame_timestamp._video_metadata_cache[video_path]
    fps = metadata['fps']
    
    if fps > 0:
        # Calculate timestamp in seconds, then convert to nanoseconds
        timestamp_seconds = frame_idx / fps
        timestamp_nanoseconds = timestamp_seconds * 1e9
        return timestamp_nanoseconds
    else:
        # Fallback: use frame index as timestamp (in nanoseconds)
        return float(frame_idx) * 1e9


def extract_timestamp_from_filename(image_path: str) -> float:
    """
    Extract timestamp from image filename.
    Assumes filename contains a nanosecond timestamp.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Timestamp in nanoseconds
    """
    filename = os.path.basename(image_path)
    
    # Try to extract timestamp from filename
    # Common patterns: 1403636580838555648.jpg, 1403636580838555648.000000000.jpg, etc.
    timestamp_patterns = [
        r'(\d{16,19})',  # 16-19 digit timestamp
        r'(\d{10,13})',  # 10-13 digit timestamp (seconds or milliseconds)
    ]
    
    for pattern in timestamp_patterns:
        match = re.search(pattern, filename)
        if match:
            timestamp_str = match.group(1)
            timestamp = float(timestamp_str)
            
            # If it's a short timestamp (10-13 digits), assume it's in seconds and convert to nanoseconds
            if len(timestamp_str) <= 13:
                timestamp *= 1e9
            
            return timestamp
    
    # If no timestamp found, use file modification time as fallback
    try:
        mtime = os.path.getmtime(image_path)
        return mtime * 1e9  # Convert to nanoseconds
    except:
        # Last resort: use a default timestamp
        return 0.0 
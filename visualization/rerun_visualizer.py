"""
Rerun visualization utilities for Pi3SLAM.
"""

import numpy as np
import time
import multiprocessing as mp
import rerun as rr


def visualization_process(queue: mp.Queue, rerun_port: int, max_points: int = 50000, update_interval: float = 0.1):
    """
    Separate process for handling Rerun visualization.
    
    Args:
        queue: Multiprocessing queue for receiving visualization data
        rerun_port: Port for Rerun server
        max_points: Maximum number of points to visualize
        update_interval: Minimum time between updates
    """
    try:
        # Initialize Rerun in this process
        rr.init("Pi3SLAM Online - Visualization Process")
        rr.spawn(port=rerun_port, connect=True)
        
        # Set up the coordinate system
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        
        # Log initial camera trajectory
        rr.log("world/camera_trajectory", rr.Points3D(
            positions=np.zeros((1, 3)), 
            colors=np.array([[1, 0, 0]])
        ), static=True)
        
        print(f"🎥 Visualization process started on port {rerun_port}")
        
        last_update_time = 0
        
        while True:
            try:
                # Get data from queue with timeout
                data = queue.get(timeout=1.0)
                
                if data is None:  # Shutdown signal
                    break
                
                current_time = time.time()
                if current_time - last_update_time < update_interval:
                    continue
                
                last_update_time = current_time
                
                # Extract visualization data
                all_points = data.get('points', np.array([]))
                all_colors = data.get('colors', np.array([]))
                camera_positions = data.get('camera_positions', np.array([]))
                camera_orientations = data.get('camera_orientations', np.array([]))
                chunk_end_poses = data.get('chunk_end_poses', np.array([]))
                current_frame = data.get('current_frame', None)
                chunk_start_frame = data.get('chunk_start_frame', None)
                chunk_end_frame = data.get('chunk_end_frame', None)
                frame_info = data.get('frame_info', '')
                stats_text = data.get('stats_text', '')
                
                # Subsample points for visualization
                if len(all_points) > max_points:
                    indices = np.random.choice(len(all_points), max_points, replace=False)
                    all_points = all_points[indices]
                    all_colors = all_colors[indices]
                
                # Log point cloud
                if len(all_points) > 0:
                    rr.log("world/point_cloud", rr.Points3D(
                        positions=all_points,
                        colors=all_colors
                    ), static=True)
                
                # Log camera trajectory and poses
                if len(camera_positions) > 0:
                    # Camera positions as points
                    rr.log("world/camera_poses", rr.Points3D(
                        positions=camera_positions,
                        colors=np.full((len(camera_positions), 3), [1.0, 0.0, 0.0])
                    ), static=True)
                    
                    # Camera trajectory as line
                    if len(camera_positions) > 1:
                        rr.log("world/camera_trajectory", rr.LineStrips3D(
                            strips=[camera_positions],
                            colors=np.array([1.0, 0.0, 0.0])
                        ), static=True)
                
                # Log chunk end poses as linestrips
                if len(chunk_end_poses) > 1:
                    rr.log("world/chunk_end_trajectory", rr.LineStrips3D(
                        strips=[chunk_end_poses],
                        colors=np.array([0.0, 1.0, 0.0])  # Green color for chunk end trajectory
                    ), static=True)
                    
                    # Also log chunk end poses as points
                    rr.log("world/chunk_end_poses", rr.Points3D(
                        positions=chunk_end_poses,
                        colors=np.full((len(chunk_end_poses), 3), [0.0, 1.0, 0.0])
                    ), static=True)
                    
                # Camera coordinate frames (only recent ones)
                num_cameras = len(camera_positions)
                recent_cameras = min(5, num_cameras)
                
                for i in range(num_cameras):
                    if i < num_cameras - recent_cameras:
                        # Clear old coordinate frames
                        rr.log(f"world/camera_{i}/x_axis", rr.LineStrips3D(strips=[]), static=True)
                        rr.log(f"world/camera_{i}/y_axis", rr.LineStrips3D(strips=[]), static=True)
                        rr.log(f"world/camera_{i}/z_axis", rr.LineStrips3D(strips=[]), static=True)
                        rr.log(f"world/camera_{i}/frustum", rr.LineStrips3D(strips=[]), static=True)
                    else:
                        # Show recent coordinate frames
                        camera_pos = camera_positions[i]
                        camera_rot = camera_orientations[i]
                        
                        # Create coordinate frame vectors
                        scale = 0.15
                        origin = camera_pos
                        x_axis = origin + scale * camera_rot[:, 0]
                        y_axis = origin + scale * camera_rot[:, 1]
                        z_axis = origin + scale * camera_rot[:, 2]
                        
                        # Log coordinate frame axes
                        rr.log(f"world/camera_{i}/x_axis", rr.LineStrips3D(
                            strips=[[origin, x_axis]],
                            colors=np.array([1.0, 0.0, 0.0])
                        ), static=True)
                        
                        rr.log(f"world/camera_{i}/y_axis", rr.LineStrips3D(
                            strips=[[origin, y_axis]],
                            colors=np.array([0.0, 1.0, 0.0])
                        ), static=True)
                        
                        rr.log(f"world/camera_{i}/z_axis", rr.LineStrips3D(
                            strips=[[origin, z_axis]],
                            colors=np.array([0.0, 0.0, 1.0])
                        ), static=True)
                        
                        # Add camera frustum for the most recent camera
                        if i == num_cameras - 1:
                            frustum_scale = 0.2
                            near_plane = origin + frustum_scale * camera_rot[:, 2]
                            
                            corner1 = near_plane + frustum_scale * 0.5 * (camera_rot[:, 0] + camera_rot[:, 1])
                            corner2 = near_plane + frustum_scale * 0.5 * (camera_rot[:, 0] - camera_rot[:, 1])
                            corner3 = near_plane + frustum_scale * 0.5 * (-camera_rot[:, 0] - camera_rot[:, 1])
                            corner4 = near_plane + frustum_scale * 0.5 * (-camera_rot[:, 0] + camera_rot[:, 1])
                            
                            rr.log(f"world/camera_{i}/frustum", rr.LineStrips3D(
                                strips=[[origin, corner1], [origin, corner2], [origin, corner3], [origin, corner4],
                                       [corner1, corner2], [corner2, corner3], [corner3, corner4], [corner4, corner1]],
                                colors=np.array([1.0, 1.0, 0.0])
                            ), static=True)
                
                # Always show frustum for the most recent camera (even if it's the only one)
                if num_cameras > 0:
                    latest_camera_pos = camera_positions[-1]
                    latest_camera_rot = camera_orientations[-1]
                    
                    frustum_scale = 0.2
                    origin = latest_camera_pos
                    near_plane = origin + frustum_scale * latest_camera_rot[:, 2]
                    
                    corner1 = near_plane + frustum_scale * 0.5 * (latest_camera_rot[:, 0] + latest_camera_rot[:, 1])
                    corner2 = near_plane + frustum_scale * 0.5 * (latest_camera_rot[:, 0] - latest_camera_rot[:, 1])
                    corner3 = near_plane + frustum_scale * 0.5 * (-latest_camera_rot[:, 0] - latest_camera_rot[:, 1])
                    corner4 = near_plane + frustum_scale * 0.5 * (-latest_camera_rot[:, 0] + latest_camera_rot[:, 1])
                    
                    rr.log("world/latest_camera_frustum", rr.LineStrips3D(
                        strips=[[origin, corner1], [origin, corner2], [origin, corner3], [origin, corner4],
                               [corner1, corner2], [corner2, corner3], [corner3, corner4], [corner4, corner1]],
                        colors=np.array([1.0, 1.0, 0.0])
                    ), static=False)
                
                # Log current video frame
                if current_frame is not None:
                    try:
                        print(f"📷 Logging current frame: shape {current_frame.shape}, dtype {current_frame.dtype}")
                        
                        # Ensure the frame is in the correct format for Rerun
                        if len(current_frame.shape) == 3 and current_frame.shape[2] == 3:  # HWC format
                            rr.log("world/current_frame", rr.Image(current_frame), static=False)
                        else:
                            print(f"⚠️  Skipping current frame: invalid shape {current_frame.shape}")
                    except Exception as e:
                        print(f"⚠️  Error logging current frame: {e}")
                
                # Log current chunk start and end frames
                if chunk_start_frame is not None:
                    try:
                        print(f"📷 Logging chunk start frame: shape {chunk_start_frame.shape}, dtype {chunk_start_frame.dtype}")
                        
                        # Ensure the frame is in the correct format for Rerun
                        if len(chunk_start_frame.shape) == 3 and chunk_start_frame.shape[2] == 3:  # HWC format
                            rr.log("world/chunk_start_frame", rr.Image(chunk_start_frame), static=False)
                        else:
                            print(f"⚠️  Skipping chunk start frame: invalid shape {chunk_start_frame.shape}")
                    except Exception as e:
                        print(f"⚠️  Error logging chunk start frame: {e}")
                
                if chunk_end_frame is not None:
                    try:
                        print(f"📷 Logging chunk end frame: shape {chunk_end_frame.shape}, dtype {chunk_end_frame.dtype}")
                        
                        # Ensure the frame is in the correct format for Rerun
                        if len(chunk_end_frame.shape) == 3 and chunk_end_frame.shape[2] == 3:  # HWC format
                            rr.log("world/chunk_end_frame", rr.Image(chunk_end_frame), static=False)
                        else:
                            print(f"⚠️  Skipping chunk end frame: invalid shape {chunk_end_frame.shape}")
                    except Exception as e:
                        print(f"⚠️  Error logging chunk end frame: {e}")
                
                # Log frame info and statistics
                if frame_info:
                    rr.log("world/frame_info", rr.TextDocument(frame_info), static=False)
                
                if stats_text:
                    rr.log("world/stats", rr.TextDocument(stats_text), static=True)
                
            except mp.queues.Empty:
                continue  # No data available, continue waiting
            except Exception as e:
                print(f"⚠️  Visualization process error: {e}")
                continue
        
        # Cleanup
        rr.disconnect()
        print("🎥 Visualization process stopped")
        
    except Exception as e:
        print(f"❌ Failed to start visualization process: {e}") 
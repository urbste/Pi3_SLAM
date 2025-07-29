"""
Pi3SLAM Online model with modular components.
"""

import torch
import numpy as np
import time
import multiprocessing as mp
import os
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

from pi3.models.pi3 import Pi3
from pi3.utils.basic import write_ply
from pi3.utils.geometry import depth_edge

# Import our modular components
from utils.geometry_utils import apply_transformation_to_points, apply_transformation_to_poses, verify_coordinate_system
from utils.image_utils import calculate_target_size, load_images_from_paths, extract_colors_from_chunk
from utils.timestamp_utils import extract_timestamps_from_paths
from alignment.sim3_transformation import (
    SIM3Transformation, estimate_sim3_transformation, estimate_sim3_transformation_robust,
    estimate_sim3_transformation_robust_irls
)
from alignment.correspondence import find_corresponding_points
from datasets.image_datasets import ChunkImageDataset
from visualization.rerun_visualizer import visualization_process


class Pi3SLAMOnlineRerun:
    """
    Pi3SLAM with online processing and Rerun visualization.
    Processes long sequences in chunks with real-time visualization.
    """
    
    def __init__(self, model: Pi3, chunk_length: int = 100, overlap: int = 10, 
                 device: str = 'cuda', conf_threshold: float = 0.5, 
                 undistortion_maps=None, cam_scale: float = 1.0,
                 max_chunks_in_memory: int = 5, enable_disk_cache: bool = False,
                 cache_dir: str = None, rerun_port: int = 9090, enable_sim3_optimization: bool = True):
        """
        Initialize Pi3SLAM Online with Rerun visualization.
        """
        self.model = model
        self.chunk_length = chunk_length
        self.overlap = overlap
        self.device = device
        self.conf_threshold = conf_threshold
        self.undistortion_maps = undistortion_maps
        self.cam_scale = cam_scale
        self.max_chunks_in_memory = max_chunks_in_memory
        self.enable_disk_cache = enable_disk_cache
        self.cache_dir = cache_dir
        self.pixel_limit = 255000
        self.memory_monitoring = True
        self.rerun_port = rerun_port
        
        # Initialize data structures
        self.chunk_results = []
        self.camera_ids = []
        self.aligned_points = []
        self.aligned_camera_poses = []
        self.aligned_camera_ids = []
        self.aligned_colors = []
        
        # Camera trajectory tracking
        self.full_camera_trajectory = []
        self.full_camera_orientations = []
        self.chunk_end_poses = []
        
        # Timestamp tracking
        self.timestamps = []
        
        # Statistics
        self.stats = {
            'total_chunks': 0,
            'total_frames': 0,
            'processing_times': [],
            'alignment_times': [],
            'load_times': [],
            'inference_times': [],
            'postprocess_times': [],
            'cpu_transfer_times': []
        }
        
        # Configuration for correspondence search
        self.correspondence_subsample_factor = 6
        self.use_robust_alignment = False
        
        # Configuration for robust alignment
        self.ransac_max_correspondence_distance = 0.01  # Maximum distance for valid correspondences
        self.ransac_max_iterations = 500  # Maximum RANSAC iterations
        self.icp_threshold = 0.05  # ICP distance threshold
        self.icp_max_iterations = 10  # Maximum ICP iterations
        self.enable_sim3_optimization = enable_sim3_optimization  # Enable/disable SIM3 optimization
        
        # Visualization
        self.vis_running = False
        self.vis_process = None
        self.vis_queue = None
        self.current_chunk_result = None
        self.max_points_visualization = 50000
        self.update_interval = 0.1
        self.visualization_subsample_ratio = 0.1
        
        # Background loading
        self.loader_running = False
        self.loader_process = None
        self.loader_queue = None
        self.image_cache = {}
        self.target_size = None
        
        # Disk cache
        self.disk_cache = {}
        if self.enable_disk_cache:
            self._setup_disk_cache()
        
        print(f"🚀 Pi3SLAM Online initialized with chunk_length={chunk_length}, overlap={overlap}")
        print(f"   Device: {device}, Confidence threshold: {conf_threshold}")
        print(f"   Camera scale: {cam_scale}")
        if undistortion_maps:
            print(f"   Undistortion: Enabled")
        if enable_disk_cache:
            print(f"   Disk cache: Enabled ({cache_dir or 'Temporary'})")
        print(f"   Max chunks in memory: {max_chunks_in_memory}")
        print(f"   Robust alignment: RANSAC+ICP (distance: {self.ransac_max_correspondence_distance}, RANSAC iter: {self.ransac_max_iterations}, ICP iter: {self.icp_max_iterations})")
        print(f"   SIM3 optimization: {'ENABLED' if self.enable_sim3_optimization else 'DISABLED'}")
    
    def _setup_disk_cache(self):
        """Set up disk cache if enabled."""
        if self.cache_dir is None:
            import tempfile
            self.cache_dir = tempfile.mkdtemp(prefix="pi3slam_cache_")
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"💾 Disk caching enabled: {self.cache_dir}")
    
    def start_background_loader(self, all_image_paths: List):
        """Start background image loading for the entire sequence using chunk-based loading."""
        if self.loader_running:
            return
        
        try:
            # Store the full image paths list
            self.image_paths = all_image_paths
            
            # Calculate target size from first image if not already set
            if self.target_size is None:
                self.target_size = calculate_target_size(all_image_paths[0], self.pixel_limit)
            
            # Create chunk-based dataset
            dataset = ChunkImageDataset(all_image_paths, self.chunk_length, self.overlap, 
                                      self.target_size, device='cpu', 
                                      undistortion_maps=self.undistortion_maps)
            
            # Create DataLoader with multiple workers for parallel loading
            self.background_loader = DataLoader(
                dataset,
                batch_size=1,  # Each batch is one chunk
                shuffle=False,  # Keep original order
                num_workers=2,  # Use 2 worker processes for chunk loading
                pin_memory=True,  # Pin memory for faster GPU transfer
                persistent_workers=True,  # Keep workers alive between epochs
                prefetch_factor=1  # Prefetch 1 batch per worker
            )
            
            self.loader_running = True
            print(f"🔄 Background chunk loader started with {len(all_image_paths)} images in {len(dataset)} chunks")
            
        except Exception as e:
            print(f"❌ Failed to start background loader: {e}")
            print("🖥️  Continuing with synchronous loading...")
            self.loader_running = False
    
    def start_visualization(self):
        """Start the Rerun visualization in a separate process."""
        if self.vis_running:
            return
        
        try:
            # Create multiprocessing queue
            self.vis_queue = mp.Queue()
            
            # Start visualization process
            self.vis_process = mp.Process(
                target=visualization_process,
                args=(self.vis_queue, self.rerun_port, self.max_points_visualization, self.update_interval)
            )
            self.vis_process.start()
            self.vis_running = True
            
            print(f"🎥 Rerun visualization process started on port {self.rerun_port}")
            print(f"🌐 Open http://localhost:{self.rerun_port} in your browser to view the visualization")
            
        except Exception as e:
            print(f"❌ Failed to start visualization process: {e}")
            print("🖥️  Continuing without visualization...")
            self.vis_running = False
    
    def stop_visualization(self):
        """Stop the Rerun visualization process."""
        if self.vis_running and self.vis_process is not None:
            try:
                # Send shutdown signal
                if self.vis_queue is not None:
                    self.vis_queue.put(None)
                
                # Wait for process to finish
                self.vis_process.join(timeout=5.0)
                
                # Force terminate if still running
                if self.vis_process.is_alive():
                    self.vis_process.terminate()
                    self.vis_process.join(timeout=2.0)
                
                self.vis_running = False
                print("🎥 Rerun visualization process stopped")
                
            except Exception as e:
                print(f"⚠️  Error stopping visualization process: {e}")
    
    def process_chunks_with_background_loader(self) -> List[Dict]:
        """
        Process all chunks using the background loader for optimal performance.
        """
        if not self.loader_running or self.background_loader is None:
            raise ValueError("Background loader not started")
        
        results = []
        chunk_count = 0
        total_chunks = len(self.background_loader.dataset)
        
        print(f"\n🔄 Starting chunk-based processing with {total_chunks} chunks...")
        print("=" * 60)
        
        try:
            for chunk_data in self.background_loader:
                chunk_count += 1
                start_idx = chunk_data['start_idx'].item()
                end_idx = chunk_data['end_idx'].item()
                chunk_images = chunk_data['chunk']
                chunk_paths = chunk_data['chunk_paths'][0]
                
                print(f"\n📦 Processing chunk {chunk_count}/{total_chunks}: frames {start_idx + 1}-{end_idx}")
                
                # Process chunk directly with loaded images
                result = self._process_chunk_with_images(chunk_images, chunk_paths, start_idx, end_idx)
                results.append(result)
                
                # Small delay to allow visualization to update
                if self.vis_running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print(f"\n⚠️  Processing interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            raise
        
        return results
    
    def _process_chunk_with_images(self, chunk_images: torch.Tensor, chunk_paths: List, 
                                  start_idx: int, end_idx: int) -> Dict:
        """Process a chunk using pre-loaded images."""
        start_time = time.time()
        
        # Extract timestamps for this chunk
        actual_paths = []
        for path_item in chunk_paths:
            if isinstance(path_item, list):
                actual_paths.extend(path_item)
            else:
                actual_paths.append(path_item)
        
        chunk_timestamps = extract_timestamps_from_paths(actual_paths)
        self.timestamps.extend(chunk_timestamps)
        
        # Model inference
        inference_start = time.time()
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                result = self.model(chunk_images.to(self.device))
        inference_time = time.time() - inference_start
        
        # Process masks
        postprocess_start = time.time()
        masks = torch.sigmoid(result['conf'][..., 0]) > 0.1
        non_edge = ~depth_edge(result['local_points'][..., 2], rtol=0.03)
        masks = torch.logical_and(masks, non_edge)[0]
        
        # Move all results to CPU to save GPU memory
        cpu_transfer_start = time.time()
        cpu_result = {
            'points': result['points'][0].cpu(),
            'local_points': result['local_points'][0].cpu(),
            'camera_poses': result['camera_poses'][0].cpu(),
            'conf': result['conf'][0].cpu(),
            'masks': masks.cpu(),
            'image_paths': chunk_paths,
            'images': chunk_images.squeeze(0).cpu()
        }
        cpu_transfer_time = time.time() - cpu_transfer_start
        postprocess_time = time.time() - postprocess_start
        
        # Store timing info
        self.stats.setdefault('load_times', []).append(0.0)
        self.stats.setdefault('inference_times', []).append(inference_time)
        self.stats.setdefault('postprocess_times', []).append(postprocess_time)
        self.stats.setdefault('cpu_transfer_times', []).append(cpu_transfer_time)
        
        # Generate camera IDs for this chunk
        chunk_idx = len(self.chunk_results)
        start_frame = chunk_idx * (self.chunk_length - self.overlap)
        chunk_camera_ids = list(range(start_frame + 1, start_frame + len(chunk_paths) + 1))
        
        # Store chunk result
        self.chunk_results.append(cpu_result)
        self.camera_ids.append(chunk_camera_ids)
        
        # Track current chunk for frame display
        self.current_chunk_result = cpu_result
        
        # Align with previous chunks
        alignment_start = time.time()
        aligned_chunk = self._align_chunk_online(cpu_result, chunk_camera_ids)
        alignment_time = time.time() - alignment_start
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats['total_chunks'] += 1
        self.stats['total_frames'] += len(chunk_paths)
        self.stats['processing_times'].append(processing_time)
        self.stats['alignment_times'].append(alignment_time)
        
        # Update visualization
        if self.vis_running:
            self._update_visualization()
        
        # Print progress
        fps = len(chunk_paths) / processing_time if processing_time > 0 else 0
        avg_fps = self.stats['total_frames'] / sum(self.stats['processing_times']) if self.stats['processing_times'] else 0
        
        print(f"📊 Chunk {self.stats['total_chunks']}: {len(chunk_paths)} frames in {processing_time:.2f}s ({fps:.1f} FPS)")
        print(f"   Inference: {inference_time:.2f}s, Postprocess: {postprocess_time:.2f}s")
        
        # Print alignment timing if available
        if len(self.chunk_results) > 1:
            # Extract timing information from alignment result
            timing = aligned_chunk.get('timing', {})
            correspondence_time = timing.get('correspondence', 0.0)
            sim3_time = timing.get('sim3', 0.0)
            optimize_time = timing.get('optimize', 0.0)
            transform_time = timing.get('transform', 0.0)
            color_time = timing.get('color', 0.0)
            
            print(f"   Alignment: {alignment_time:.2f}s (Corresp: {correspondence_time:.2f}s, SIM3: {sim3_time:.2f}s, Optimize: {optimize_time:.2f}s, Transform: {transform_time:.2f}s)")
            print(f"   Colors: {color_time:.2f}s")
        else:
            print(f"   Alignment: {alignment_time:.2f}s (first chunk)")
        
        print(f"   Total points: {len(self.aligned_points[-1]) if self.aligned_points else 0}, Average FPS: {avg_fps:.1f}")
        
        return {
            'chunk_result': cpu_result,
            'aligned_chunk': aligned_chunk,
            'processing_time': processing_time,
            'alignment_time': alignment_time,
            'camera_ids': chunk_camera_ids,
            'timestamps': chunk_timestamps
        }
    
    def _align_chunk_online(self, chunk_result: Dict, chunk_camera_ids: List[int]) -> Dict:
        """Align a new chunk with previously processed chunks."""
        if len(self.chunk_results) == 1:
            # First chunk, no alignment needed
            aligned_points = chunk_result['points']
            aligned_poses = chunk_result['camera_poses']
            color_start = time.time()
            aligned_colors = extract_colors_from_chunk(chunk_result)
            color_time = time.time() - color_start
            print(f"color_time: {color_time:.3f}s (using cached images)")
            
            # Store aligned data
            self.aligned_points.append(aligned_points.reshape(-1, 3).cpu().numpy())
            self.aligned_camera_poses.append(aligned_poses.cpu().numpy())
            self.aligned_camera_ids.extend(chunk_camera_ids)
            self.aligned_colors.append(aligned_colors.cpu().numpy())
            
            # Store camera trajectory data (always kept in memory)
            poses_np = aligned_poses.cpu().numpy()
            for pose in poses_np:
                camera_pos = pose[:3, 3]  # Translation part
                camera_rot = pose[:3, :3]  # Rotation part
                self.full_camera_trajectory.append(camera_pos)
                self.full_camera_orientations.append(camera_rot)
            
            # Store end pose of this chunk for linestrip visualization
            if len(poses_np) > 0:
                end_pose = poses_np[-1]  # Last pose of the chunk
                self.chunk_end_poses.append(end_pose[:3, 3])  # Store position only
            
            # Store end frame from this chunk for visualization
            if 'images' in chunk_result:
                # Get the last frame of the chunk
                images_tensor = chunk_result['images']
                print(f"📷 Images tensor shape: {images_tensor.shape}")
                
                # Handle different tensor shapes
                if len(images_tensor.shape) == 5:  # (B, N, C, H, W) - batch dimension still present
                    end_frame = images_tensor[0, -1]  # Remove batch, get last frame: (C, H, W)
                    print(f"📷 End frame shape (5D->3D): {end_frame.shape}")
                elif len(images_tensor.shape) == 4:  # (N, C, H, W) - batch already removed
                    end_frame = images_tensor[-1]  # Last frame: (C, H, W)
                    print(f"📷 End frame shape (4D->3D): {end_frame.shape}")
                else:
                    end_frame = images_tensor  # Already single frame
                    print(f"📷 End frame shape (already 3D): {end_frame.shape}")
                
                # Convert to RGB format for display (CHW -> HWC)
                if end_frame.shape[0] == 3:  # CHW format
                    frame_rgb = end_frame.permute(1, 2, 0).numpy()
                    print(f"📷 Frame RGB shape (CHW->HWC): {frame_rgb.shape}")
                else:
                    frame_rgb = end_frame.numpy()
                    print(f"📷 Frame RGB shape (direct): {frame_rgb.shape}")
                
                # Normalize to [0, 255] range
                if frame_rgb.max() <= 1.0:
                    frame_rgb = (frame_rgb * 255).astype(np.uint8)
                
                print(f"📷 Final frame shape: {frame_rgb.shape}, dtype: {frame_rgb.dtype}")
                self.current_chunk_end_frame = frame_rgb
            
            # Manage memory after storing new chunk
            self._manage_memory()
            
            # Set as current reference chunk (like offline version)
            self.current_reference_chunk = {
                'points': aligned_points,
                'camera_poses': aligned_poses,
                'conf': chunk_result['conf'],
                'masks': chunk_result['masks']
            }
            self.current_reference_camera_ids = chunk_camera_ids
            
            return {
                'points': aligned_points,
                'camera_poses': aligned_poses,
                'colors': aligned_colors,
                'transformation': np.eye(4),
                'timing': {
                    'correspondence': 0.0,
                    'sim3': 0.0,
                    'optimize': 0.0,
                    'transform': 0.0,
                    'color': color_time
                }
            }
        
        # Align with current reference chunk (like offline version)
        current_chunk = self.current_reference_chunk
        current_camera_ids = self.current_reference_camera_ids
        
        # Find overlapping cameras
        overlap_cameras_current = current_camera_ids[-self.overlap:] if len(current_camera_ids) >= self.overlap else current_camera_ids
        overlap_cameras_next = chunk_camera_ids[:self.overlap] if len(chunk_camera_ids) >= self.overlap else chunk_camera_ids
        
        common_camera_ids = set(overlap_cameras_current) & set(overlap_cameras_next)
        
        if len(common_camera_ids) >= 2:
            # Find corresponding points
            correspondence_start = time.time()
            corresponding_points1, corresponding_points2 = find_corresponding_points(
                current_chunk['points'], chunk_result['points'],
                current_camera_ids, chunk_camera_ids,
                conf1=current_chunk.get('conf', None),
                conf2=chunk_result.get('conf', None),
                subsample_factor=self.correspondence_subsample_factor,  # Use configurable subsampling
                conf_threshold=self.conf_threshold,
                use_robust_filtering=self.use_robust_alignment
            )
            correspondence_time = time.time() - correspondence_start
            
            if len(corresponding_points1) >= 10:
                # Estimate SIM3 transformation using robust Open3D RANSAC + ICP
                sim3_start = time.time()
                
                try:
                    import open3d as o3d
                    import open3d.pipelines.registration as treg
                    
                    # Create Open3D point clouds
                    source_pcd = o3d.geometry.PointCloud()
                    target_pcd = o3d.geometry.PointCloud()
                    
                    source_pcd.points = o3d.utility.Vector3dVector(corresponding_points2)  # Next chunk points
                    target_pcd.points = o3d.utility.Vector3dVector(corresponding_points1)  # Current chunk points
                    
                    # Create correspondence set
                    corres = o3d.utility.Vector2iVector()
                    for i in range(len(corresponding_points1)):
                        corres.append([i, i])
                    
                    # Set parameters for correspondence-based RANSAC
                    max_correspondence_distance = self.ransac_max_correspondence_distance
                    ransac_n = min(6, len(corresponding_points1))  # Number of correspondences for RANSAC (at least 3 for rigid, 6 for similarity)
                    
                    try:
                        # Use correspondence-based RANSAC with explicit point correspondences
                        ransac_result = treg.registration_ransac_based_on_correspondence(
                            source_pcd, target_pcd, corres,
                            max_correspondence_distance,
                            treg.TransformationEstimationPointToPoint(with_scaling=True),
                            ransac_n,
                            criteria=treg.RANSACConvergenceCriteria(max_iteration=self.ransac_max_iterations)
                        )
                        print(f"RANSAC result: {ransac_result}")
                        initial_transformation = ransac_result.transformation
                        print(f"Correspondence-based RANSAC fitness: {ransac_result.fitness:.4f}")
                        print(f"Correspondence-based RANSAC inlier RMSE: {ransac_result.inlier_rmse:.6f}")
                        print(f"Initial transformation matrix:\n{initial_transformation}")
                        
                        # Refine with ICP using the initial transformation
                        icp_result = treg.registration_icp(
                            source_pcd, target_pcd,
                            self.icp_threshold, initial_transformation,
                            treg.TransformationEstimationPointToPoint(with_scaling=True),
                            treg.ICPConvergenceCriteria(max_iteration=self.icp_max_iterations)
                        )
                        
                        transformation_matrix = icp_result.transformation
                        print(f"ICP refinement fitness: {icp_result.fitness:.4f}")
                        print(f"ICP refinement RMSE: {icp_result.inlier_rmse:.6f}")
                        print(f"Final transformation matrix:\n{transformation_matrix}")
                        
                        # Extract scale, rotation, and translation from final transformation
                        R = transformation_matrix[:3, :3]
                        t = transformation_matrix[:3, 3]
                        
                        # Extract scale from rotation matrix
                        U, S, Vt = np.linalg.svd(R)
                        scale = np.mean(S)
                        
                        # Normalize rotation matrix
                        R_normalized = R / scale
                        
                        # Ensure proper rotation matrix
                        U, _, Vt = np.linalg.svd(R_normalized)
                        R_normalized = U @ Vt
                        if np.linalg.det(R_normalized) < 0:
                            Vt[-1, :] *= -1
                            R_normalized = U @ Vt
                        
                        sim3_transform = SIM3Transformation(scale, R_normalized, t)
                        print(f"  🔧 Robust RANSAC+ICP SIM3 estimation successful (RANSAC fitness: {ransac_result.fitness:.3f}, ICP fitness: {icp_result.fitness:.3f})")
                        
                    except Exception as ransac_error:
                        print(f"  🔧 RANSAC+ICP failed: {ransac_error}, falling back to standard estimation")
                        sim3_transform = estimate_sim3_transformation(corresponding_points2, corresponding_points1)
                        
                except ImportError:
                    # Fallback to standard estimation if Open3D not available
                    sim3_transform = estimate_sim3_transformation(corresponding_points2, corresponding_points1)
                    print(f"  🔧 Open3D not available, using standard SIM3 estimation with {len(corresponding_points1)} points")
                except Exception as e:
                    # Fallback to standard estimation on error
                    sim3_transform = estimate_sim3_transformation(corresponding_points2, corresponding_points1)
                    print(f"  🔧 Open3D RANSAC+ICP failed: {e}, using standard SIM3 estimation")
                
                sim3_time = time.time() - sim3_start
                
                # Optimize SIM3 transformation using overlapping camera poses with LM (optional)
                if self.enable_sim3_optimization:
                    optimize_start = time.time()
                    optimized_transform = self._optimize_sim3_transformation(
                        sim3_transform, current_chunk, chunk_result, 
                        current_camera_ids, chunk_camera_ids, common_camera_ids
                    )
                    optimize_time = time.time() - optimize_start
                    transformation = optimized_transform.get_matrix()
                    print(f"  Optimized SIM3 transformation with scale={optimized_transform.s:.3f}")
                else:
                    # Skip optimization, use RANSAC+ICP result directly
                    transformation = sim3_transform.get_matrix()
                    optimize_time = 0.0
                    print(f"  Using RANSAC+ICP transformation directly (optimization disabled)")
                
                # Apply transformation
                transform_start = time.time()
                transformed_points = apply_transformation_to_points(chunk_result['points'], transformation)
                transformed_poses = apply_transformation_to_poses(chunk_result['camera_poses'], transformation)
                transform_time = time.time() - transform_start
                
                # Extract colors for transformed points
                color_start = time.time()
                aligned_colors = extract_colors_from_chunk(chunk_result)
                color_time = time.time() - color_start
                
                # Store aligned data (excluding overlap)
                overlap_size = self.overlap
                self.aligned_points.append(transformed_points[overlap_size:].reshape(-1, 3).cpu().numpy())
                self.aligned_camera_poses.append(transformed_poses[overlap_size:].cpu().numpy())
                self.aligned_camera_ids.extend(chunk_camera_ids[overlap_size:])
                self.aligned_colors.append(aligned_colors[overlap_size:].reshape(-1, 3).cpu().numpy())
                
                # Store camera trajectory data (always kept in memory) - excluding overlap
                poses_np = transformed_poses[overlap_size:].cpu().numpy()
                for pose in poses_np:
                    camera_pos = pose[:3, 3]  # Translation part
                    camera_rot = pose[:3, :3]  # Rotation part
                    self.full_camera_trajectory.append(camera_pos)
                    self.full_camera_orientations.append(camera_rot)
                
                # Store start and end frames from this chunk for visualization
                if 'images' in chunk_result:
                    # Get the first and last frames of the chunk
                    images_tensor = chunk_result['images']
                    
                    # Handle different tensor shapes
                    if len(images_tensor.shape) == 5:  # (B, N, C, H, W) - batch dimension still present
                        start_frame = images_tensor[0, 0]  # Remove batch, get first frame: (C, H, W)
                        end_frame = images_tensor[0, -1]  # Remove batch, get last frame: (C, H, W)
                    elif len(images_tensor.shape) == 4:  # (N, C, H, W) - batch already removed
                        start_frame = images_tensor[0]  # First frame: (C, H, W)
                        end_frame = images_tensor[-1]  # Last frame: (C, H, W)
                    else:
                        start_frame = images_tensor  # Already single frame
                        end_frame = images_tensor  # Already single frame
                    
                    # Convert to RGB format for display (CHW -> HWC)
                    if start_frame.shape[0] == 3:  # CHW format
                        start_rgb = start_frame.permute(1, 2, 0).numpy()
                    else:
                        start_rgb = start_frame.numpy()
                    
                    if end_frame.shape[0] == 3:  # CHW format
                        end_rgb = end_frame.permute(1, 2, 0).numpy()
                    else:
                        end_rgb = end_frame.numpy()
                    
                    # Normalize to [0, 255] range
                    if start_rgb.max() <= 1.0:
                        start_rgb = (start_rgb * 255).astype(np.uint8)
                    if end_rgb.max() <= 1.0:
                        end_rgb = (end_rgb * 255).astype(np.uint8)
                    
                    self.current_chunk_start_frame = start_rgb
                    self.current_chunk_end_frame = end_rgb
            
                # Manage memory after storing new chunk
                self._manage_memory()
                
                # Update current reference chunk for next iteration (like offline version)
                self.current_reference_chunk = {
                    'points': transformed_points,
                    'camera_poses': transformed_poses,
                    'conf': chunk_result['conf'],
                    'masks': chunk_result['masks']
                }
                self.current_reference_camera_ids = chunk_camera_ids
                
                return {
                    'points': transformed_points,
                    'camera_poses': transformed_poses,
                    'colors': aligned_colors,
                    'transformation': transformation,
                    'timing': {
                        'correspondence': correspondence_time,
                        'sim3': sim3_time,
                        'optimize': optimize_time,
                        'transform': transform_time,
                        'color': color_time
                    }
                }
            else:
                print(f"⚠️  Warning: Insufficient corresponding points ({len(corresponding_points1)}), using identity transformation")
        else:
            print(f"⚠️  Warning: Insufficient overlapping cameras ({len(common_camera_ids)}), using identity transformation")
    
        # Fallback: no transformation
        aligned_points = chunk_result['points']
        aligned_poses = chunk_result['camera_poses']
        color_start = time.time()
        aligned_colors = extract_colors_from_chunk(chunk_result)
        color_time = time.time() - color_start
        print(f"color_time: {color_time:.3f}s (using cached images)")
        
        # Store aligned data
        self.aligned_points.append(aligned_points.reshape(-1, 3).cpu().numpy())
        self.aligned_camera_poses.append(aligned_poses.cpu().numpy())
        self.aligned_camera_ids.extend(chunk_camera_ids)
        self.aligned_colors.append(aligned_colors.reshape(-1, 3).cpu().numpy())
        
        # Store camera trajectory data (always kept in memory)
        poses_np = aligned_poses.cpu().numpy()
        for pose in poses_np:
            camera_pos = pose[:3, 3]  # Translation part
            camera_rot = pose[:3, :3]  # Rotation part
            self.full_camera_trajectory.append(camera_pos)
            self.full_camera_orientations.append(camera_rot)
        
        # Store end pose of this chunk for linestrip visualization
        if len(poses_np) > 0:
            end_pose = poses_np[-1]  # Last pose of the chunk
            self.chunk_end_poses.append(end_pose[:3, 3])  # Store position only
        
        # Store start and end frames from this chunk for visualization
        if 'images' in chunk_result:
            # Get the first and last frames of the chunk
            images_tensor = chunk_result['images']
            
            # Handle different tensor shapes
            if len(images_tensor.shape) == 5:  # (B, N, C, H, W) - batch dimension still present
                start_frame = images_tensor[0, 0]  # Remove batch, get first frame: (C, H, W)
                end_frame = images_tensor[0, -1]  # Remove batch, get last frame: (C, H, W)
            elif len(images_tensor.shape) == 4:  # (N, C, H, W) - batch already removed
                start_frame = images_tensor[0]  # First frame: (C, H, W)
                end_frame = images_tensor[-1]  # Last frame: (C, H, W)
            else:
                start_frame = images_tensor  # Already single frame
                end_frame = images_tensor  # Already single frame
            
            # Convert to RGB format for display (CHW -> HWC)
            if start_frame.shape[0] == 3:  # CHW format
                start_rgb = start_frame.permute(1, 2, 0).numpy()
            else:
                start_rgb = start_frame.numpy()
            
            if end_frame.shape[0] == 3:  # CHW format
                end_rgb = end_frame.permute(1, 2, 0).numpy()
            else:
                end_rgb = end_frame.numpy()
            
            # Normalize to [0, 255] range
            if start_rgb.max() <= 1.0:
                start_rgb = (start_rgb * 255).astype(np.uint8)
            if end_rgb.max() <= 1.0:
                end_rgb = (end_rgb * 255).astype(np.uint8)
            
            self.current_chunk_start_frame = start_rgb
            self.current_chunk_end_frame = end_rgb
        
        # Manage memory after storing new chunk
        self._manage_memory()
        
        # Update current reference chunk for next iteration (like offline version)
        self.current_reference_chunk = {
            'points': aligned_points,
            'camera_poses': aligned_poses,
            'conf': chunk_result['conf'],
            'masks': chunk_result['masks']
        }
        self.current_reference_camera_ids = chunk_camera_ids
        
        return {
            'points': aligned_points,
            'camera_poses': aligned_poses,
            'colors': aligned_colors,
            'transformation': np.eye(4),
            'timing': {
                'correspondence': 0.0,
                'sim3': 0.0,
                'optimize': 0.0,
                'transform': 0.0,
                'color': color_time
            }
        }
    
    def _optimize_sim3_transformation(self, initial_transform: SIM3Transformation, 
                                    current_chunk: Dict, next_chunk: Dict,
                                    current_camera_ids: List[int], next_camera_ids: List[int],
                                    common_camera_ids: set) -> SIM3Transformation:
        """
        Optimize SIM3 transformation using LM least squares on overlapping camera poses.
        
        Args:
            initial_transform: Initial SIM3 transformation from Open3D RANSAC
            current_chunk: Current chunk data
            next_chunk: Next chunk data
            current_camera_ids: Camera IDs in current chunk
            next_camera_ids: Camera IDs in next chunk
            common_camera_ids: Set of overlapping camera IDs
        
        Returns:
            Optimized SIM3Transformation
        """
        try:
            from scipy.optimize import least_squares
        except ImportError:
            print("  Warning: scipy not available, using initial transformation")
            return initial_transform
        
        # Get overlapping camera poses
        current_poses = []
        next_poses = []
        
        for camera_id in sorted(common_camera_ids):
            # Find camera index in current chunk
            current_idx = current_camera_ids.index(camera_id)
            current_poses.append(current_chunk['camera_poses'][current_idx].numpy())
            
            # Find camera index in next chunk
            next_idx = next_camera_ids.index(camera_id)
            next_poses.append(next_chunk['camera_poses'][next_idx].numpy())
        
        current_poses = np.stack(current_poses)  # (N, 4, 4)
        next_poses = np.stack(next_poses)        # (N, 4, 4)
        
        # Extract camera positions (translation part)
        current_positions = current_poses[:, :3, 3]  # (N, 3)
        next_positions = next_poses[:, :3, 3]        # (N, 3)
        
        # Extract camera orientations (rotation part)
        current_rotations = current_poses[:, :3, :3]  # (N, 3, 3)
        next_rotations = next_poses[:, :3, :3]        # (N, 3, 3)
        
        def residual_function(params):
            """Residual function for least squares optimization."""
            # Unpack parameters: [scale, rx, ry, rz, tx, ty, tz]
            scale = params[0]
            rodrigues = params[1:4]
            translation = params[4:7]
            
            # Convert Rodrigues parameters to rotation matrix
            from utils.geometry_utils import rodrigues_to_rotation_matrix
            R = rodrigues_to_rotation_matrix(rodrigues)
            
            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = scale * R
            T[:3, 3] = translation
            
            residuals = []
            
            # Weighting factors for balancing position and rotation
            pos_weight = 1.0
            rot_weight = 0.1  # Rotation residuals are typically smaller in magnitude
            
            # Apply transformation to next chunk poses
            for i, pose in enumerate(next_poses):
                transformed_pose = T @ pose
                
                # Position residuals (3 components per camera) - weighted
                pos_residual = pos_weight * (current_positions[i] - transformed_pose[:3, 3])
                residuals.extend(pos_residual)
                
                # Rotation residuals using Rodrigues representation - weighted
                from utils.geometry_utils import rotation_matrix_to_rodrigues
                relative_rot = current_rotations[i] @ transformed_pose[:3, :3].T
                relative_rodrigues = rotation_matrix_to_rodrigues(relative_rot)
                rot_residual = rot_weight * relative_rodrigues
                residuals.extend(rot_residual)
            
            return np.array(residuals)
        
        # Initial parameters from the initial transformation
        initial_scale = initial_transform.s
        initial_rotation = initial_transform.R
        initial_translation = initial_transform.t
        
        # Convert rotation matrix to Rodrigues parameters
        from utils.geometry_utils import rotation_matrix_to_rodrigues
        initial_rodrigues = rotation_matrix_to_rodrigues(initial_rotation)
        
        initial_params = np.concatenate([
            [initial_scale],
            initial_rodrigues,
            initial_translation
        ])
        
        # Optimize using LM least squares
        print(f"  Optimizing SIM3 transformation with {len(common_camera_ids)} overlapping cameras using LM least squares...")
        
        try:
            result = least_squares(
                residual_function,
                initial_params,
                method='lm',
                max_nfev=100,  # Maximum function evaluations
                ftol=1e-6,     # Function tolerance
                xtol=1e-6,     # Parameter tolerance
                verbose=0
            )
            
            # Convert least_squares result to minimize-like format
            class ResultWrapper:
                def __init__(self, ls_result):
                    self.success = ls_result.success
                    self.x = ls_result.x
                    self.fun = np.sum(ls_result.fun ** 2)  # Sum of squared residuals
                    self.message = ls_result.message
            
            result = ResultWrapper(result)
            
        except Exception as e:
            print(f"  Warning: LM optimization failed: {e}, using initial transformation")
            return initial_transform
        
        if result.success:
            # Extract optimized parameters
            optimized_scale = result.x[0]
            optimized_rodrigues = result.x[1:4]
            optimized_translation = result.x[4:7]
            
            # Convert back to rotation matrix
            from utils.geometry_utils import rodrigues_to_rotation_matrix
            optimized_rotation = rodrigues_to_rotation_matrix(optimized_rodrigues)
            
            optimized_transform = SIM3Transformation(
                optimized_scale, optimized_rotation, optimized_translation
            )
            
            # Calculate final residuals for diagnostics
            final_residuals = residual_function(result.x)
            pos_residuals = final_residuals[::6]  # Every 6th residual (position components)
            rot_residuals = final_residuals[3::6]  # Every 6th residual starting from 3rd (rotation components)
            
            print(f"  LM optimization successful. Final cost: {result.fun:.6f}")
            print(f"  Position RMS: {np.sqrt(np.mean(pos_residuals**2)):.6f}")
            print(f"  Rotation RMS: {np.sqrt(np.mean(rot_residuals**2)):.6f}")
            return optimized_transform
        else:
            print(f"  LM optimization failed: {result.message}, using initial transformation")
            return initial_transform
    
    def _manage_memory(self):
        """Manage memory by moving old chunks to disk and keeping only recent ones in memory."""
        if len(self.aligned_points) <= self.max_chunks_in_memory:
            return
        
        # Move oldest chunks to disk
        chunks_to_move = len(self.aligned_points) - self.max_chunks_in_memory
        
        for i in range(chunks_to_move):
            chunk_idx = i
            if chunk_idx < len(self.aligned_points):
                # Save to disk
                self._save_chunk_to_disk(
                    chunk_idx,
                    self.aligned_points[chunk_idx],
                    self.aligned_colors[chunk_idx],
                    self.aligned_camera_poses[chunk_idx]
                )
                
                # Clear from memory (but keep camera trajectory data)
                self.aligned_points[chunk_idx] = None
                self.aligned_colors[chunk_idx] = None
                self.aligned_camera_poses[chunk_idx] = None
        
        # Compact lists by removing None entries
        self.aligned_points = [p for p in self.aligned_points if p is not None]
        self.aligned_colors = [c for c in self.aligned_colors if c is not None]
        self.aligned_camera_poses = [p for p in self.aligned_camera_poses if p is not None]
        
        print(f"💾 Moved {chunks_to_move} chunks to disk, keeping {len(self.aligned_points)} in memory")
        print(f"📷 Camera trajectory: {len(self.full_camera_trajectory)} poses, {len(self.chunk_end_poses)} chunk end poses")
        
        # Log memory usage if monitoring is enabled
        if self.memory_monitoring:
            self._log_memory_usage()
    
    def _save_chunk_to_disk(self, chunk_idx: int, points: np.ndarray, colors: np.ndarray, poses: np.ndarray):
        """Save chunk data to disk to free memory."""
        if self.cache_dir is None:
            return
        
        try:
            chunk_file = os.path.join(self.cache_dir, f"chunk_{chunk_idx:06d}.npz")
            np.savez_compressed(
                chunk_file,
                points=points,
                colors=colors,
                poses=poses
            )
        except Exception as e:
            print(f"⚠️  Failed to save chunk {chunk_idx} to disk: {e}")
    
    def _log_memory_usage(self):
        """Log current memory usage for monitoring."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
            
            # Count total points in memory
            total_points_in_memory = 0
            for points in self.aligned_points:
                if points is not None:
                    total_points_in_memory += len(points)
            
            print(f"🧠 Memory usage: {memory_mb:.1f} MB, Points in memory: {total_points_in_memory:,}, Camera poses: {len(self.full_camera_trajectory):,}, Chunk end poses: {len(self.chunk_end_poses)}")
            
        except ImportError:
            # psutil not available, skip memory logging
            pass
        except Exception as e:
            print(f"⚠️  Failed to log memory usage: {e}")
    
    def _update_visualization(self):
        """Update visualization with current data."""
        if not self.vis_running or self.vis_queue is None:
            return
        
        # Throttle visualization updates
        current_time = time.time()
        if hasattr(self, '_last_viz_update') and current_time - self._last_viz_update < self.update_interval:
            return
        self._last_viz_update = current_time
        
        try:
            # Get visualization points
            all_points, all_colors = self._get_visualization_points()
            
            # Get camera trajectory data
            camera_positions = np.array(self.full_camera_trajectory) if self.full_camera_trajectory else np.array([])
            camera_orientations = np.array(self.full_camera_orientations) if self.full_camera_orientations else np.array([])
            
            # Get current frame and chunk frames
            current_frame = None
            chunk_start_frame = None
            chunk_end_frame = None
            
            if self.current_chunk_result is not None and 'images' in self.current_chunk_result:
                chunk_images = self.current_chunk_result['images']
                if len(chunk_images) > 0:
                    # Current frame is the last frame in the chunk
                    current_frame = chunk_images[-1].numpy()
                    
                    # Chunk start frame is the first frame in the chunk
                    chunk_start_frame = chunk_images[0].numpy()
                    
                    # Chunk end frame is the last frame in the chunk (same as current frame)
                    chunk_end_frame = chunk_images[-1].numpy()
                    
                    # Convert from CHW to HWC format for visualization
                    if len(current_frame.shape) == 3 and current_frame.shape[0] == 3:
                        current_frame = np.transpose(current_frame, (1, 2, 0))
                        chunk_start_frame = np.transpose(chunk_start_frame, (1, 2, 0))
                        chunk_end_frame = np.transpose(chunk_end_frame, (1, 2, 0))
                    
                    # Normalize to 0-255 range if needed
                    if current_frame.max() <= 1.0:
                        current_frame = (current_frame * 255).astype(np.uint8)
                        chunk_start_frame = (chunk_start_frame * 255).astype(np.uint8)
                        chunk_end_frame = (chunk_end_frame * 255).astype(np.uint8)
            
            # Create frame info and statistics
            frame_info = f"Frame: {self.stats['total_frames']}, Chunk: {self.stats['total_chunks']}"
            stats_text = f"Chunks: {self.stats['total_chunks']}, Frames: {self.stats['total_frames']}, Points: {len(all_points)}"
            
            # Send data to visualization process
            viz_data = {
                'points': all_points,
                'colors': all_colors,
                'camera_positions': camera_positions,
                'camera_orientations': camera_orientations,
                'chunk_end_poses': np.array(self.chunk_end_poses) if self.chunk_end_poses else np.array([]),
                'current_frame': current_frame,
                'chunk_start_frame': chunk_start_frame,
                'chunk_end_frame': chunk_end_frame,
                'frame_info': frame_info,
                'stats_text': stats_text
            }
            
            # Non-blocking put to avoid blocking the main process
            try:
                self.vis_queue.put_nowait(viz_data)
            except mp.queues.Full:
                pass
            
        except Exception as e:
            print(f"⚠️  Error sending visualization data: {e}")
    
    def _get_visualization_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get subsampled points for visualization."""
        if not self.aligned_points:
            return np.array([]), np.array([])
        
        # Combine all available points
        all_points = []
        all_colors = []
        
        for i, points in enumerate(self.aligned_points):
            if points is not None:
                all_points.append(points)
                all_colors.append(self.aligned_colors[i])
        
        if not all_points:
            return np.array([]), np.array([])
        
        # Combine all points
        if len(all_points) == 1:
            combined_points = all_points[0]
            combined_colors = all_colors[0]
        else:
            combined_points = np.concatenate(all_points, axis=0)
            combined_colors = np.concatenate(all_colors, axis=0)
        
        # Subsample for visualization
        if len(combined_points) > 0:
            subsample_size = max(1000, int(len(combined_points) * self.visualization_subsample_ratio))
            if len(combined_points) > subsample_size:
                indices = np.random.choice(len(combined_points), subsample_size, replace=False)
                combined_points = combined_points[indices]
                combined_colors = combined_colors[indices]
        
        return combined_points, combined_colors
    
    def save_final_result(self, save_path: str, max_points: int = 1000000):
        """Save the final aligned trajectory result."""
        if not self.aligned_points:
            print("No points to save")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Combine all aligned points and colors
        all_points = np.vstack(self.aligned_points)
        all_colors = np.vstack(self.aligned_colors)
        
        # Subsample if needed
        if len(all_points) > max_points:
            indices = np.random.choice(len(all_points), max_points, replace=False)
            all_points = all_points[indices]
            all_colors = all_colors[indices]
        
        # Save point cloud
        print(f"💾 Saving final trajectory to: {save_path}")
        write_ply(torch.from_numpy(all_points), torch.from_numpy(all_colors), save_path, max_points=max_points)
        print(f"✅ Saved trajectory with {len(all_points)} points from {self.stats['total_chunks']} chunks")
        
        # Save camera poses
        if self.full_camera_trajectory:
            camera_positions = np.array(self.full_camera_trajectory)
            camera_colors = np.full((len(camera_positions), 3), [1.0, 0.0, 0.0])
            
            camera_filename = save_path.replace('.ply', '_camera_poses.ply')
            write_ply(torch.from_numpy(camera_positions), torch.from_numpy(camera_colors), camera_filename)
            print(f"📷 Saved {len(camera_positions)} camera poses to: {camera_filename}")
    
    def save_trajectory_tum(self, save_path: str, timestamps: List[float] = None):
        """
        Save the final aligned trajectory in TUM format.
        
        Args:
            save_path: Path to save the TUM format trajectory file
            timestamps: Optional list of timestamps for each pose (if None, will use stored timestamps or frame indices)
        """
        if not self.full_camera_trajectory:
            print("No camera trajectory available to save")
            return
        
        try:
            from scipy.spatial.transform import Rotation
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Use stored timestamps if available, otherwise use provided timestamps or frame indices
            if timestamps is None:
                if self.timestamps and len(self.timestamps) >= len(self.full_camera_trajectory):
                    timestamps = self.timestamps[:len(self.full_camera_trajectory)]
                else:
                    timestamps = list(range(len(self.full_camera_trajectory)))
            
            print(f"💾 Saving trajectory in TUM format to: {save_path}")
            print(f"   Using {len(timestamps)} timestamps for {len(self.full_camera_trajectory)} poses")
            
            with open(save_path, 'w') as f:
                # TUM format header
                f.write("# timestamp tx ty tz qx qy qz qw\n")
                
                for i, (camera_pos, camera_rot) in enumerate(zip(self.full_camera_trajectory, self.full_camera_orientations)):
                    # Get timestamp (use stored timestamp if available, otherwise use frame index)
                    t = timestamps[i] if i < len(timestamps) else float(i)
                    
                    # Extract translation (x, y, z)
                    x, y, z = camera_pos
                    
                    # Convert rotation matrix to quaternion
                    # Note: scipy uses quaternion format [x, y, z, w]
                    quat = Rotation.from_matrix(camera_rot).as_quat()
                    qx, qy, qz, qw = quat
                    
                    # Write TUM format line: timestamp tx ty tz qx qy qz qw
                    # Use full precision for timestamp to match ground truth format
                    f.write(f"{t:.9f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
            
            print(f"✅ Saved trajectory with {len(self.full_camera_trajectory)} poses to: {save_path}")
            
        except ImportError:
            print("❌ Error: scipy.spatial.transform.Rotation not available")
        except Exception as e:
            print(f"❌ Error saving trajectory: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.stop_visualization()
        except:
            pass
    
    def cleanup_disk_cache(self):
        """Clean up disk cache files."""
        if self.cache_dir and os.path.exists(self.cache_dir):
            try:
                import shutil
                shutil.rmtree(self.cache_dir)
                print(f"🧹 Cleaned up disk cache: {self.cache_dir}")
            except Exception as e:
                print(f"⚠️  Failed to clean up disk cache: {e}")
    
    def configure_alignment(self, ransac_max_correspondence_distance: float = 0.01, 
                          ransac_max_iterations: int = 10, icp_threshold: float = 0.01, 
                          icp_max_iterations: int = 50):
        """
        Configure alignment parameters for robust RANSAC+ICP alignment.
        
        Args:
            ransac_max_correspondence_distance: Maximum distance for valid correspondences in RANSAC
            ransac_max_iterations: Maximum RANSAC iterations
            icp_threshold: ICP distance threshold for refinement
            icp_max_iterations: Maximum ICP iterations for refinement
        """
        self.ransac_max_correspondence_distance = ransac_max_correspondence_distance
        self.ransac_max_iterations = ransac_max_iterations
        self.icp_threshold = icp_threshold
        self.icp_max_iterations = icp_max_iterations
        
        print(f"🔧 Alignment configuration updated:")
        print(f"   RANSAC max correspondence distance: {ransac_max_correspondence_distance}")
        print(f"   RANSAC max iterations: {ransac_max_iterations}")
        print(f"   ICP threshold: {icp_threshold}")
        print(f"   ICP max iterations: {icp_max_iterations}")
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        total_processing_time = sum(self.stats['processing_times']) if self.stats['processing_times'] else 0
        overall_fps = self.stats['total_frames'] / total_processing_time if total_processing_time > 0 else 0
        
        return {
            'total_chunks': self.stats['total_chunks'],
            'total_frames': self.stats['total_frames'],
            'total_processing_time': total_processing_time,
            'overall_fps': overall_fps,
            'processing_times': self.stats['processing_times'],
            'alignment_times': self.stats['alignment_times'],
            'load_times': self.stats['load_times'],
            'inference_times': self.stats['inference_times'],
            'postprocess_times': self.stats['postprocess_times'],
            'cpu_transfer_times': self.stats['cpu_transfer_times']
        } 
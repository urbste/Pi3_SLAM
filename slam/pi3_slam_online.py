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
from utils.camera_estimation import estimate_camera_parameters_single_chunk, print_camera_parameters_summary
from utils.keypoint_extraction import ALIKEDExtractor
from utils.chunk_reconstruction import ChunkPTRecon
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
                 cache_dir: str = None, rerun_port: int = 9090, enable_sim3_optimization: bool = True,
                 estimate_camera_params: bool = False, extract_keypoints: bool = False,
                 max_num_keypoints: int = 512, keypoint_detection_threshold: float = 0.005,
                 create_chunk_reconstruction: bool = False, save_chunk_reconstructions: bool = False):
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
        self.estimate_camera_params = estimate_camera_params
        self.extract_keypoints = extract_keypoints
        self.create_chunk_reconstruction = create_chunk_reconstruction
        self.save_chunk_reconstructions = save_chunk_reconstructions
        
        # Initialize keypoint extractor if enabled
        self.keypoint_extractor = None
        if self.extract_keypoints:
            try:
                self.keypoint_extractor = ALIKEDExtractor(
                    max_num_keypoints=max_num_keypoints,
                    detection_threshold=keypoint_detection_threshold,
                    device=device
                )
                print(f"🔍 ALIKED keypoint extractor initialized with {max_num_keypoints} max keypoints")
            except ImportError:
                print("⚠️  Warning: lightglue not available, keypoint extraction disabled")
                self.extract_keypoints = False
        
        # Initialize chunk reconstruction if enabled
        self.chunk_reconstructor = None
        if self.create_chunk_reconstruction:
            try:
                self.chunk_reconstructor = ChunkPTRecon()
                print(f"🔧 ChunkPTRecon initialized ")
            except ImportError:
                print("⚠️  Warning: pytheia not available, chunk reconstruction disabled")
                self.create_chunk_reconstruction = False
        
        # Initialize data structures
        self.chunk_results = []
        self.camera_ids = []
        self.aligned_points = []
        self.aligned_camera_poses = []
        self.aligned_camera_ids = []
        self.aligned_colors = []
        
        # Keypoint data structures for visualization
        self.aligned_keypoint_points = None
        self.aligned_keypoint_colors = None
        
        # Camera trajectory tracking
        self.full_camera_trajectory = []
        self.full_camera_orientations = []
        self.chunk_end_poses = []
        
        # Timestamp tracking
        self.timestamps = []
        
        # Configuration for correspondence search
        self.correspondence_subsample_factor = 1 if self.extract_keypoints else 10
        
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
            print(f"target_size: {self.target_size}")
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

    def interpolate_world_points_for_ref(self, result, keypoints):
        # interpolate like the colors for keypoints
        H, W = result['images'].shape[-2:]
        grid_x = (keypoints[:, :, 0] / (W - 1)) * 2 - 1  # Convert to [-1, 1]
        grid_y = (keypoints[:, :, 1] / (H - 1)) * 2 - 1  # Convert to [-1, 1]
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1).cpu()  # Shape: (1, 1, N, 2)
        points = torch.nn.functional.grid_sample(
            result['points'].permute(0, 3, 1, 2), grid, mode="bilinear", align_corners=False, padding_mode="border")
        local_points = torch.nn.functional.grid_sample(
            result['local_points'].permute(0, 3, 1, 2), grid, mode="bilinear", align_corners=False, padding_mode="border")
        confidences = torch.nn.functional.grid_sample(
            result['conf'].permute(0, 3, 1, 2), grid, mode="nearest", align_corners=False, padding_mode="border")
        masks = torch.nn.functional.grid_sample(
            result['masks'].unsqueeze(1).float(), grid, mode="nearest", align_corners=False, padding_mode="border")
        return_result = {
            'points': points.squeeze().permute(0, 2, 1),
            'local_points': local_points.squeeze().permute(0, 2, 1),
            'conf': confidences.squeeze(1).permute(0, 2, 1),
            'masks': masks.squeeze(1).permute(0, 2, 1).bool()
        }
        return return_result
    
    def _process_chunk_with_images(self, chunk_images: torch.Tensor, chunk_paths: List, 
                                  start_idx: int, end_idx: int) -> Dict:
        """Process a chunk using pre-loaded images."""
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
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                result = self.model(chunk_images.to(self.device))
        
        # Estimate camera parameters for this chunk (if enabled)
        camera_params = None
        if self.estimate_camera_params:
            camera_params = estimate_camera_parameters_single_chunk(result)
            print_camera_parameters_summary(camera_params)
        
        # Extract keypoints for this chunk (if enabled)
        keypoint_results = None
        if self.extract_keypoints and self.keypoint_extractor is not None:
            keypoint_results = self.keypoint_extractor.extract_with_colors(chunk_images)
            print(f"🔍 Extracted {keypoint_results['keypoints'].shape[-2]} keypoints per frame")
        
        # Process masks
        masks = torch.sigmoid(result['conf'][..., 0]) > 0.1
        non_edge = ~depth_edge(result['local_points'][..., 2], rtol=0.03)
        masks = torch.logical_and(masks, non_edge)[0]
        
        # Move all results to CPU to save GPU memory
        cpu_result = {
            'points': result['points'][0].cpu(),
            'local_points': result['local_points'][0].cpu(),
            'camera_poses': result['camera_poses'][0].cpu(),
            'conf': result['conf'][0].cpu(),
            'masks': masks.cpu(),
            'image_paths': chunk_paths,
            'images': chunk_images.squeeze(0).cpu()
        }
        
        # Add camera parameters if estimated
        if camera_params is not None:
            cpu_result['camera_params'] = {k: v.cpu() for k, v in camera_params.items()}
        
        # Add keypoint results if extracted
        if keypoint_results is not None:
            # interpolate 3D points for each keypoint
            res = self.interpolate_world_points_for_ref(cpu_result, keypoint_results['keypoints'])
            cpu_result['points_kp'] = res['points']
            cpu_result['local_points_kp'] = res['local_points']
            cpu_result['conf_kp'] = res['conf']
            cpu_result['masks_kp'] = res['masks']

            cpu_result['keypoints'] = keypoint_results['keypoints'].cpu()
            cpu_result['descriptors'] = keypoint_results['descriptors'].cpu()
            cpu_result['scores'] = keypoint_results['scores'].cpu()
            cpu_result['colors'] = keypoint_results['colors'].cpu()
        
        # Create chunk reconstruction if enabled
        chunk_reconstruction = None
        if self.create_chunk_reconstruction and self.chunk_reconstructor is not None and keypoint_results is not None:
            # Set target size for reconstruction
            if hasattr(self, 'target_size') and self.target_size is not None:
                self.chunk_reconstructor.set_target_size(self.target_size[0], self.target_size[1])
            
            # Add camera intrinsics if available
            if camera_params is not None:
                cpu_result['intrinsics'] = camera_params['intrinsics']
            
            # Create reconstruction
            chunk_reconstruction = self.chunk_reconstructor.create_recon_from_chunk(cpu_result)
            self.chunk_reconstructor.print_reconstruction_summary()
            
            # Save chunk reconstruction if enabled
            if self.save_chunk_reconstructions:
                self._save_chunk_reconstruction(chunk_reconstruction, len(self.chunk_results))
        
        # Print camera parameters summary (if estimated)
        if camera_params is not None:
            print_camera_parameters_summary(camera_params)
        
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
        aligned_chunk = self._align_chunk_online(cpu_result, chunk_camera_ids)
        
        # Update visualization
        if self.vis_running:
            self._update_visualization()
        
        # Print progress
        print(f"📊 Chunk {len(self.chunk_results)}: {len(chunk_paths)} frames processed")
        print(f"   Total points: {len(self.aligned_points[-1]) if self.aligned_points else 0}")
        
        return {
            'chunk_result': cpu_result,
            'aligned_chunk': aligned_chunk,
            'camera_ids': chunk_camera_ids,
            'timestamps': chunk_timestamps
        }
    
    def _optimize_sim3_pytheia(self, source_points: np.ndarray, target_points: np.ndarray, confidences: np.ndarray) -> np.ndarray:

        import pytheia as pt
        options = pt.sfm.Sim3AlignmentOptions()
        
        # Test default values
        options.alignment_type == pt.sfm.Sim3AlignmentType.ROBUST_POINT_TO_POINT
        options.max_iterations == 20
        options.huber_threshold == 1.0
        options.verbose == False
        
        point_weights = confidences
        options.set_point_weights(point_weights)

        summary = pt.sfm.OptimizeAlignmentSim3(source_points, target_points, options)
                
        # Check that estimated parameters are close to ground truth
        T_source_to_target = pt.math.Sim3d.exp(summary.sim3_params).matrix()
        
        return T_source_to_target
    
    def _align_chunk_online(self, chunk_result: Dict, chunk_camera_ids: List[int]) -> Dict:
        """Align a new chunk with previously processed chunks."""
        if len(self.chunk_results) == 1:
            # First chunk, no alignment needed
            aligned_points = chunk_result['points']
            aligned_poses = chunk_result['camera_poses']
            aligned_colors = extract_colors_from_chunk(chunk_result)
            
            # Use keypoint points for visualization if available
            if 'points_kp' in chunk_result:
                aligned_keypoint_points = chunk_result['points_kp']
                aligned_keypoint_colors = chunk_result['colors']  # Use keypoint colors
            else:
                aligned_keypoint_points = aligned_points
                aligned_keypoint_colors = aligned_colors
            
            # Store aligned data
            self.aligned_points.append(aligned_points.reshape(-1, 3).cpu().numpy())
            self.aligned_camera_poses.append(aligned_poses.cpu().numpy())
            self.aligned_camera_ids.extend(chunk_camera_ids)
            self.aligned_colors.append(aligned_colors.reshape(-1, 3).cpu().numpy())
            
            # Store keypoint data for visualization
            if 'points_kp' in chunk_result:
                self.aligned_keypoint_points = aligned_keypoint_points.reshape(-1, 3).cpu().numpy()
                self.aligned_keypoint_colors = aligned_keypoint_colors.reshape(-1, 3).cpu().numpy()
            else:
                self.aligned_keypoint_points = aligned_points.reshape(-1, 3).cpu().numpy()
                self.aligned_keypoint_colors = aligned_colors.reshape(-1, 3).cpu().numpy()
            
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
                'masks': chunk_result['masks'],
            }
            # Store keypoint data in reference chunk if available
            if 'points_kp' in chunk_result:
                self.current_reference_chunk['points_kp'] = aligned_keypoint_points
                self.current_reference_chunk['colors_kp'] = aligned_keypoint_colors
                self.current_reference_chunk['conf_kp'] = chunk_result['conf_kp']
                self.current_reference_chunk['masks_kp'] = chunk_result['masks_kp']
            
            self.current_reference_camera_ids = chunk_camera_ids
            
            return {
                'points': aligned_points,
                'camera_poses': aligned_poses,
                'colors': aligned_colors,
                'transformation': np.eye(4)
            }
        
        # Align with current reference chunk (like offline version)
        current_chunk = self.current_reference_chunk
        current_camera_ids = self.current_reference_camera_ids
        
        # Find overlapping cameras
        overlap_cameras_current = current_camera_ids[-self.overlap:] if len(current_camera_ids) >= self.overlap else current_camera_ids
        overlap_cameras_next = chunk_camera_ids[:self.overlap] if len(chunk_camera_ids) >= self.overlap else chunk_camera_ids
        
        common_camera_ids = set(overlap_cameras_current) & set(overlap_cameras_next)
        
        if len(common_camera_ids) >= 2:
            # Use keypoint points for alignment if available
            if 'points_kp' in current_chunk and 'points_kp' in chunk_result:
                # Use keypoint points for correspondence finding
                points1 = current_chunk['points_kp']
                points2 = chunk_result['points_kp']
                conf1 = current_chunk.get('conf_kp', None)
                conf2 = chunk_result.get('conf_kp', None)
                print(f"🔍 Using keypoint points for alignment: {points1.shape} -> {points2.shape}")
            else:
                # Fallback to full point clouds
                points1 = current_chunk['points']
                points2 = chunk_result['points']
                conf1 = current_chunk.get('conf', None)
                conf2 = chunk_result.get('conf', None)
                print(f"🔍 Using full point clouds for alignment: {points1.shape} -> {points2.shape}")
            
            # Find corresponding points
            corresponding_points1, corresponding_points2, mean_conf = find_corresponding_points(
                points1, points2,
                current_camera_ids, chunk_camera_ids,
                conf1=conf1,
                conf2=conf2,
                subsample_factor=self.correspondence_subsample_factor,  # Use configurable subsampling
                conf_threshold=self.conf_threshold,
            )
            
            if len(corresponding_points1) >= 10:
                # Estimate SIM3 transformation using robust Open3D RANSAC + ICP
                T_source_to_target = self._optimize_sim3_pytheia(
                    corresponding_points2, corresponding_points1, mean_conf)
                
                # Apply transformation to both full points and keypoint points
                N, H, W, _ = chunk_result['points'].shape
                transformed_points = apply_transformation_to_points(chunk_result['points'].reshape(N, H*W, 3), T_source_to_target)
                transformed_poses = apply_transformation_to_poses(chunk_result['camera_poses'], T_source_to_target)
                
                # Transform keypoint points if available
                if 'points_kp' in chunk_result:
                    transformed_keypoint_points = apply_transformation_to_points(chunk_result['points_kp'], T_source_to_target)
                else:
                    transformed_keypoint_points = transformed_points
                
                # Extract colors for transformed points
                aligned_colors = extract_colors_from_chunk(chunk_result)
                
                # Store aligned data (excluding overlap)
                overlap_size = self.overlap
                self.aligned_points.append(transformed_points[overlap_size:].reshape(-1, 3).cpu().numpy())
                self.aligned_camera_poses.append(transformed_poses[overlap_size:].cpu().numpy())
                self.aligned_camera_ids.extend(chunk_camera_ids[overlap_size:])
                self.aligned_colors.append(aligned_colors[overlap_size:].reshape(-1, 3).cpu().numpy())
                
                # Store keypoint data for visualization
                if 'points_kp' in chunk_result:
                    self.aligned_keypoint_points = transformed_keypoint_points.reshape(-1, 3).cpu().numpy()
                    self.aligned_keypoint_colors = chunk_result['colors'].reshape(-1, 3).cpu().numpy()
                else:
                    self.aligned_keypoint_points = transformed_points.reshape(-1, 3).cpu().numpy()
                    self.aligned_keypoint_colors = aligned_colors.reshape(-1, 3).cpu().numpy()
                
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
                # Store keypoint data in reference chunk if available
                if 'points_kp' in chunk_result:
                    self.current_reference_chunk['points_kp'] = transformed_keypoint_points
                    self.current_reference_chunk['colors_kp'] = chunk_result['colors']
                    self.current_reference_chunk['conf_kp'] = chunk_result['conf_kp']
                    self.current_reference_chunk['masks_kp'] = chunk_result['masks_kp']
                
                self.current_reference_camera_ids = chunk_camera_ids
                
                return {
                    'points': transformed_points,
                    'camera_poses': transformed_poses,
                    'colors': aligned_colors,
                    'transformation': T_source_to_target
                }
            else:
                print(f"⚠️  Warning: Insufficient corresponding points ({len(corresponding_points1)}), using identity transformation")
        else:
            print(f"⚠️  Warning: Insufficient overlapping cameras ({len(common_camera_ids)}), using identity transformation")
    
        # Fallback: no transformation
        aligned_points = chunk_result['points']
        aligned_poses = chunk_result['camera_poses']
        aligned_colors = extract_colors_from_chunk(chunk_result)
        
        # Use keypoint points for visualization if available
        if 'points_kp' in chunk_result:
            aligned_keypoint_points = chunk_result['points_kp']
            aligned_keypoint_colors = chunk_result['colors']  # Use keypoint colors
        else:
            aligned_keypoint_points = aligned_points
            aligned_keypoint_colors = aligned_colors
        
        # Store aligned data
        self.aligned_points.append(aligned_points.reshape(-1, 3).cpu().numpy())
        self.aligned_camera_poses.append(aligned_poses.cpu().numpy())
        self.aligned_camera_ids.extend(chunk_camera_ids)
        self.aligned_colors.append(aligned_colors.reshape(-1, 3).cpu().numpy())
        
        # Store keypoint data for visualization
        if 'points_kp' in chunk_result:
            self.aligned_keypoint_points = aligned_keypoint_points.reshape(-1, 3).cpu().numpy()
            self.aligned_keypoint_colors = aligned_keypoint_colors.reshape(-1, 3).cpu().numpy()
        else:
            self.aligned_keypoint_points = aligned_points.reshape(-1, 3).cpu().numpy()
            self.aligned_keypoint_colors = aligned_colors.reshape(-1, 3).cpu().numpy()
        
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
        # Store keypoint data in reference chunk if available
        if 'points_kp' in chunk_result:
            self.current_reference_chunk['points_kp'] = aligned_keypoint_points
            self.current_reference_chunk['colors_kp'] = aligned_keypoint_colors
            self.current_reference_chunk['conf_kp'] = chunk_result['conf_kp']
            self.current_reference_chunk['masks_kp'] = chunk_result['masks_kp']
        
        self.current_reference_camera_ids = chunk_camera_ids
        
        return {
            'points': aligned_points,
            'camera_poses': aligned_poses,
            'colors': aligned_colors,
            'transformation': np.eye(4)
        }
    
    def _save_chunk_reconstruction(self, reconstruction, chunk_idx: int):
        """
        Save chunk reconstruction to disk.
        
        Args:
            reconstruction: PyTheia reconstruction object
            chunk_idx: Chunk index for filename
        """
        if not hasattr(self, 'output_dir') or self.output_dir is None:
            print("⚠️  Warning: No output directory set, skipping chunk reconstruction save")
            return
        
        try:
            # Create reconstructions subdirectory
            recon_dir = os.path.join(self.output_dir, 'reconstructions')
            os.makedirs(recon_dir, exist_ok=True)
            
            # Save reconstruction file
            recon_filename = f"chunk_{chunk_idx:06d}.sfm"
            recon_path = os.path.join(recon_dir, recon_filename)
            
            # Use the reconstructor's save method
            self.chunk_reconstructor.reconstruction = reconstruction
            self.chunk_reconstructor.save_reconstruction(recon_path)
            
            print(f"💾 Saved chunk reconstruction {chunk_idx} to: {recon_path}")
            
        except Exception as e:
            print(f"❌ Error saving chunk reconstruction {chunk_idx}: {e}")
    
    def set_output_directory(self, output_dir: str):
        """
        Set the output directory for saving chunk reconstructions.
        
        Args:
            output_dir: Directory path for output files
        """
        self.output_dir = output_dir
        print(f"📁 Output directory set to: {output_dir}")
    
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
            frame_info = f"Frame: {len(self.timestamps)}, Chunk: {len(self.chunk_results)}"
            stats_text = f"Chunks: {len(self.chunk_results)}, Frames: {len(self.timestamps)}, Points: {len(all_points)}"
            
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
        # Use keypoint points for visualization if available
        if hasattr(self, 'aligned_keypoint_points') and self.aligned_keypoint_points is not None:
            # Use keypoint points for visualization
            all_points = [self.aligned_keypoint_points]
            all_colors = [self.aligned_keypoint_colors]
        elif self.aligned_points:
            # Fallback to full point clouds
            all_points = []
            all_colors = []
            
            for i, points in enumerate(self.aligned_points):
                if points is not None:
                    all_points.append(points)
                    all_colors.append(self.aligned_colors[i])
        else:
            return np.array([]), np.array([])
        
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
        print(f"✅ Saved trajectory with {len(all_points)} points from {len(self.chunk_results)} chunks")
        
        # Save camera poses
        if self.full_camera_trajectory:
            camera_positions = np.array(self.full_camera_trajectory)
            camera_colors = np.full((len(camera_positions), 3), [1.0, 0.0, 0.0])
            
            camera_filename = save_path.replace('.ply', '_camera_poses.ply')
            write_ply(torch.from_numpy(camera_positions), torch.from_numpy(camera_colors), camera_filename)
            print(f"📷 Saved {len(camera_positions)} camera poses to: {camera_filename}")
    
    def save_trajectory_tum(self, save_path: str, timestamps: List[float] = None, integer_timestamp: bool = False):
        """
        Save the final aligned trajectory in TUM format.
        
        Args:
            save_path: Path to save the TUM format trajectory file
            timestamps: Optional list of timestamps for each pose (if None, will use stored timestamps or frame indices)
            integer_timestamp: If True, saves timestamps as integers (for 7-scenes). Otherwise, saves as float (for EuRoC).
        """
        if not self.full_camera_trajectory:
            print("No camera trajectory available to save")
            return
        
        try:
            from scipy.spatial.transform import Rotation
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Override timestamps with a simple range if integer_timestamp is requested
            if integer_timestamp:
                timestamps_to_use = list(range(len(self.full_camera_trajectory)))
                print("   Using integer frame indices as timestamps.")
            elif timestamps is not None:
                timestamps_to_use = timestamps
            elif self.timestamps and len(self.timestamps) >= len(self.full_camera_trajectory):
                timestamps_to_use = self.timestamps[:len(self.full_camera_trajectory)]
            else:
                timestamps_to_use = list(range(len(self.full_camera_trajectory)))

            
            print(f"💾 Saving trajectory in TUM format to: {save_path}")
            print(f"   Using {len(timestamps_to_use)} timestamps for {len(self.full_camera_trajectory)} poses")
            
            with open(save_path, 'w') as f:
                # TUM format header
                f.write("# timestamp tx ty tz qx qy qz qw\n")
                
                for i, (camera_pos, camera_rot) in enumerate(zip(self.full_camera_trajectory, self.full_camera_orientations)):
                    # Get timestamp
                    t = timestamps_to_use[i] if i < len(timestamps_to_use) else i
                    
                    # Extract translation (x, y, z)
                    x, y, z = camera_pos
                    
                    # Convert rotation matrix to quaternion
                    quat = Rotation.from_matrix(camera_rot).as_quat()
                    qx, qy, qz, qw = quat
                    
                    # Write TUM format line: timestamp tx ty tz qx qy qz qw
                    if integer_timestamp:
                        f.write(f"{int(t)} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
                    else:
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
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        return {
            'total_chunks': len(self.chunk_results),
            'total_frames': len(self.timestamps)
        } 
"""
Dataset classes for Pi3SLAM image loading.
"""

import torch
import numpy as np
import os
from typing import List, Tuple
from torch.utils.data import Dataset
from pi3.utils.basic import TORCHCODEC_AVAILABLE


class ChunkImageDataset(Dataset):
    """
    Custom dataset for loading entire chunks of images at once.
    This is optimized for batch processing where we need chunk_length images at a time.
    """
    
    def __init__(self, image_paths: List, chunk_length: int, overlap: int, 
                 target_size: Tuple[int, int], device: str = 'cpu', 
                 undistortion_maps=None):
        """
        Initialize the chunk dataset.
        
        Args:
            image_paths: List of image paths or (video_path, frame_idx) tuples
            chunk_length: Number of frames per chunk
            overlap: Number of overlapping frames between chunks
            target_size: Target size (height, width) for resizing
            device: Device to load images on
            undistortion_maps: Optional UndistortionMaps object for applying undistortion
        """
        self.image_paths = image_paths
        self.chunk_length = chunk_length
        self.overlap = overlap
        self.target_size = target_size
        self.device = device
        self.undistortion_maps = undistortion_maps
        
        # Calculate chunk indices
        self.chunk_indices = []
        chunk_idx = 0
        while chunk_idx < len(image_paths):
            end_idx = min(chunk_idx + chunk_length, len(image_paths))
            if end_idx - chunk_idx >= 2:  # At least 2 frames required
                self.chunk_indices.append((chunk_idx, end_idx))
            chunk_idx += chunk_length - overlap
        
        # Check if we're dealing with video files for optimization
        self.is_video = isinstance(image_paths[0], tuple) if image_paths else False
        if self.is_video:
            self.video_path = image_paths[0][0]  # All frames should be from the same video
            print(f"🎬 Video detected: {self.video_path}")
            print(f"   Optimizing for video chunk loading with {len(self.chunk_indices)} chunks")
            
            # Initialize video undistortion loader if undistortion maps are provided
            if undistortion_maps is not None:
                try:
                    from pi3.utils.undistortion import VideoUndistortionLoader
                    self.video_loader = VideoUndistortionLoader(undistortion_maps, device=device)
                    print(f"   Using VideoUndistortionLoader for efficient video processing")
                except ImportError:
                    print(f"   VideoUndistortionLoader not available, using standard loading")
                    self.video_loader = None
            else:
                self.video_loader = None
        
    def __len__(self):
        return len(self.chunk_indices)
    
    def __getitem__(self, idx):
        """Load and preprocess an entire chunk of images."""
        start_idx, end_idx = self.chunk_indices[idx]
        chunk_paths = self.image_paths[start_idx:end_idx]
        
        if self.is_video:
            # Optimized video chunk loading
            images = self._load_video_chunk(chunk_paths)
        else:
            # Regular image loading
            images = self._load_image_chunk(chunk_paths)
        
        # Stack images into a chunk tensor
        chunk_tensor = torch.stack(images)
        return {
            'chunk': chunk_tensor.to(self.device),
            'start_idx': torch.tensor([start_idx]),
            'end_idx': torch.tensor([end_idx]),
            'chunk_paths': [chunk_paths]  # Wrap in list for batching
        }
    
    def _load_video_chunk(self, chunk_paths: List) -> List[torch.Tensor]:
        """Load a chunk of video frames efficiently using bulk loading when possible."""
        if not chunk_paths:
            return []
        
        # Extract video path and frame indices
        video_path = chunk_paths[0][0]  # All frames should be from the same video
        frame_indices = [path_info[1] for path_info in chunk_paths]
        
        # Use VideoUndistortionLoader if available (most efficient for undistorted video)
        if hasattr(self, 'video_loader') and self.video_loader is not None:
            try:
                # Load frames with undistortion in bulk
                frames_tensor = self.video_loader.load_and_undistort_frames(
                    video_path, frame_indices, self.target_size
                )
                # Return individual frames as list
                return [frames_tensor[i] for i in range(frames_tensor.shape[0])]
            except Exception as e:
                print(f"Warning: VideoUndistortionLoader failed, falling back to standard loading: {e}")
        
        # Try to use bulk loading with torchcodec if available
        try:
            if TORCHCODEC_AVAILABLE:
                from torchcodec.decoders import VideoDecoder
                
                # Load frames in bulk using torchcodec
                decoder = VideoDecoder(video_path, device="cpu")
                frames_tensor = decoder.get_frames_at(indices=frame_indices)  # Shape: [N, C, H, W], uint8
                del decoder  # Clean up decoder
                
                # Convert to float [0, 1] range
                frames_tensor = frames_tensor.data.float() / 255.0
                
                # Resize if target size is specified
                if self.target_size is not None:
                    frames_tensor = torch.nn.functional.interpolate(
                        frames_tensor, size=self.target_size, mode='bilinear', align_corners=False
                    )
                
                # Apply undistortion if maps are provided
                if self.undistortion_maps is not None:
                    undistorted_frames = []
                    for i in range(frames_tensor.shape[0]):
                        # Convert tensor to numpy for undistortion
                        frame_np = frames_tensor[i].permute(1, 2, 0).numpy()  # CHW to HWC
                        undistorted_np = self.undistortion_maps.undistort_image(frame_np, self.target_size)
                        # Convert back to tensor
                        undistorted_frame = torch.from_numpy(undistorted_np).float() / 255.0
                        undistorted_frame = undistorted_frame.permute(2, 0, 1)  # HWC to CHW
                        undistorted_frames.append(undistorted_frame)
                    return undistorted_frames
                else:
                    # Return individual frames as list
                    return [frames_tensor[i] for i in range(frames_tensor.shape[0])]
        
        except Exception as e:
            print(f"Warning: Bulk video loading failed, falling back to individual loading: {e}")
        
        # Fallback to individual frame loading
        images = []
        for path_info in chunk_paths:
            video_path, frame_idx = path_info
            from pi3.utils.basic import load_video_frame
            
            # Load frame using torchcodec (with fallback to OpenCV)
            image = load_video_frame(video_path, frame_idx, self.target_size, use_torchcodec=True)
            
            # Apply undistortion if maps are provided
            if self.undistortion_maps is not None:
                # Convert tensor to numpy for undistortion
                image_np = image.permute(1, 2, 0).numpy()  # CHW to HWC
                undistorted_np = self.undistortion_maps.undistort_image(image_np, self.target_size)
                # Convert back to tensor
                image = torch.from_numpy(undistorted_np).float() / 255.0
                image = image.permute(2, 0, 1)  # HWC to CHW
            
            images.append(image)
        
        return images
    
    def _load_image_chunk(self, chunk_paths: List) -> List[torch.Tensor]:
        """Load a chunk of image files."""
        images = []
        
        for path_info in chunk_paths:
            # Image file - handle None case
            if path_info is None:
                raise ValueError(f"Invalid path_info: {path_info}")
            
            # Check if file exists
            if not os.path.exists(path_info):
                raise ValueError(f"Image file not found: {path_info}")
            
            # Use PIL for image loading
            from PIL import Image
            import torchvision.transforms as transforms
            
            pil_image = Image.open(path_info).convert('RGB')
            
            # Apply undistortion if maps are provided
            if self.undistortion_maps is not None:
                # Convert PIL to numpy for undistortion
                image_np = np.array(pil_image)
                undistorted_np = self.undistortion_maps.undistort_image(image_np, self.target_size)
                # Convert back to tensor
                image = torch.from_numpy(undistorted_np).float() / 255.0
                image = image.permute(2, 0, 1)  # HWC to CHW
            else:
                # Use PIL transforms for regular loading
                transform = transforms.Compose([
                    transforms.Resize(self.target_size),  # (height, width)
                    transforms.ToTensor()
                ])
                image = transform(pil_image)
            
            images.append(image)
        
        return images
    
    def __del__(self):
        """Cleanup video loader if it exists."""
        if hasattr(self, 'video_loader') and self.video_loader is not None:
            try:
                # Close any open video decoders
                if hasattr(self.video_loader, 'decoders'):
                    for video_path in list(self.video_loader.decoders.keys()):
                        self.video_loader.close_decoder(video_path)
            except:
                pass  # Ignore errors during cleanup


class AsyncImageDataset(Dataset):
    """
    Custom dataset for asynchronous image loading with optional undistortion support.
    """
    
    def __init__(self, image_paths: List, target_size: Tuple[int, int], device: str = 'cpu', 
                 undistortion_maps=None):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of image paths or (video_path, frame_idx) tuples
            target_size: Target size (height, width) for resizing
            device: Device to load images on
            undistortion_maps: Optional UndistortionMaps object for applying undistortion
        """
        self.image_paths = image_paths
        self.target_size = target_size
        self.device = device
        self.undistortion_maps = undistortion_maps
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Load and preprocess a single image."""
        path_info = self.image_paths[idx]
        
        if isinstance(path_info, tuple):
            # Video frame
            video_path, frame_idx = path_info
            from pi3.utils.basic import load_video_frame
            
            # Load frame using torchcodec (with fallback to OpenCV)
            image = load_video_frame(video_path, frame_idx, self.target_size, use_torchcodec=True)
            
            # Apply undistortion if maps are provided
            if self.undistortion_maps is not None:
                # Convert tensor to numpy for undistortion
                image_np = image.permute(1, 2, 0).numpy()  # CHW to HWC
                undistorted_np = self.undistortion_maps.undistort_image(image_np, self.target_size)
                # Convert back to tensor
                image = torch.from_numpy(undistorted_np).float() / 255.0
                image = image.permute(2, 0, 1)  # HWC to CHW
        else:
            # Image file - handle None case
            if path_info is None:
                raise ValueError(f"Invalid path_info at index {idx}: {path_info}")
            
            # Check if file exists
            if not os.path.exists(path_info):
                raise ValueError(f"Image file not found: {path_info}")
            
            # Use PIL for image loading
            from PIL import Image
            import torchvision.transforms as transforms
            
            pil_image = Image.open(path_info).convert('RGB')
            
            # Apply undistortion if maps are provided
            if self.undistortion_maps is not None:
                # Convert PIL to numpy for undistortion
                image_np = np.array(pil_image)
                undistorted_np = self.undistortion_maps.undistort_image(image_np, self.target_size)
                # Convert back to tensor
                image = torch.from_numpy(undistorted_np).float() / 255.0
                image = image.permute(2, 0, 1)  # HWC to CHW
            else:
                # Use PIL transforms for regular loading
                transform = transforms.Compose([
                    transforms.Resize(self.target_size),  # (height, width)
                    transforms.ToTensor()
                ])
                image = transform(pil_image)
        
        return image.to(self.device) 
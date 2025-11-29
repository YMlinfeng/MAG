### data_loader.py
import os
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Tuple, Optional, Dict, Any
from config import Config

class MAGDataLoader(Dataset):
    """Dataset loader for MAG training stages following paper specifications"""
    
    def __init__(self, stage: int, config: Config):
        """
        Initialize dataset loader for specified training stage
        
        Args:
            stage: Training stage (1, 2, or 3)
            config: Configuration handler with dataset paths
            
        Raises:
            ValueError: For invalid stages
            RuntimeError: If dataset path not found
        """
        self.stage = stage
        self.config = config
        self.clip_frames = self.config.config_data['evaluation']['frames']  # Default: 60/120
        self.target_res = tuple(self.config.config_data['evaluation']['resolution'])  # (1280, 720)
        
        # Get dataset path
        try:
            self.root = config.get_dataset_path()
        except RuntimeError as e:
            raise RuntimeError(f"Dataset path error: {str(e)}")
        
        # Load metadata and setup transformations
        self.metadata = self._load_metadata()
        self.transform = self._create_transforms()
        
        # Stage-specific initialization
        if stage in (2, 3):
            self._prefilter_videos()
        if stage == 3:
            self.action_dim = self.config.config_data['model']['apb'].get('action_dim', 5)  # Default: 5 keys

    def _load_metadata(self) -> List[Dict[str, str]]:
        """Load metadata JSON for current stage"""
        meta_path = os.path.join(self.root, 'metadata.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"metadata.json not found at {self.root}")
        
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Metadata JSON error: {str(e)}")
        
        return metadata

    def _create_transforms(self) -> transforms.Compose:
        """Create image transformations pipeline"""
        return transforms.Compose([
            transforms.ToTensor(),  # Converts to [0,1] range
        ])

    def _prefilter_videos(self):
        """Filter videos that meet minimum frame requirement for Stage 2/3"""
        valid_metadata = []
        video_frame_counts = []
        
        for item in self.metadata:
            video_path = os.path.join(self.root, item['video'])
            
            # Get video frame count
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Keep videos with sufficient frames
            if total_frames >= self.clip_frames:
                valid_metadata.append(item)
                video_frame_counts.append(total_frames)
        
        self.metadata = valid_metadata
        self.video_frame_counts = video_frame_counts

    def __len__(self) -> int:
        return len(self.metadata)

    def _map_keyboard_to_vector(self, key_str: str) -> np.ndarray:
        """
        Map keyboard input to binary action vector (W=0, A=1, S=2, D=3, Space=4)
        
        Args:
            key_str: Keyboard input string (e.g., "W+A")
            
        Returns:
            Binary action vector of shape (action_dim,)
        """
        keys = key_str.upper().split('+')
        action_vec = np.zeros(self.action_dim, dtype=np.float32)
        key_mapping = {'W': 0, 'A': 1, 'S': 2, 'D': 3, 'SPACE': 4}
        
        for key in keys:
            if key in key_mapping:
                action_vec[key_mapping[key]] = 1.0
        return action_vec

    def __getitem__(self, idx: int) -> Tuple:
        try:
            if self.stage == 1:
                return self._load_image_text_pair(idx)
            elif self.stage == 2:
                return self._load_video_text_clip(idx)
            elif self.stage == 3:
                return self._load_gameplay_clip(idx)
            else:
                raise ValueError(f"Invalid stage: {self.stage}")
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            return None

    def _load_image_text_pair(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Load image-text pair for Stage 1 training"""
        item = self.metadata[idx]
        img_path = os.path.join(self.root, item['image'])
        
        # Load and process image
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_res)
        img_tensor = self.transform(img)  # Normalized [0,1] tensor
        
        return img_tensor, item['text']

    def _load_video_text_clip(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Load video clip and text for Stage 2 training"""
        item = self.metadata[idx]
        video_path = os.path.join(self.root, item['video'])
        
        # Random start frame within valid range
        total_frames = self.video_frame_counts[idx]
        start_idx = random.randint(0, total_frames - self.clip_frames)
        
        # Extract video clip
        frames = self._extract_video_clip(video_path, start_idx)
        
        # Convert to tensor [T, C, H, W]
        frame_tensors = torch.stack([self.transform(frame) for frame in frames])
        return frame_tensors, item['text']

    def _load_gameplay_clip(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load gameplay clip and actions for Stage 3 training"""
        item = self.metadata[idx]
        video_path = os.path.join(self.root, item['video'])
        action_path = os.path.join(self.root, item['actions'])
        
        # Random start frame
        total_frames = self.video_frame_counts[idx]
        start_idx = random.randint(0, total_frames - self.clip_frames)
        
        # Extract video clip
        frames = self._extract_video_clip(video_path, start_idx)
        
        # Extract action segment
        actions = self._load_actions(action_path, start_idx, total_frames)
        
        # Convert to tensors
        frame_tensors = torch.stack([self.transform(frame) for frame in frames])
        action_tensors = torch.tensor(actions, dtype=torch.float32)
        
        return frame_tensors, action_tensors

    def _extract_video_clip(self, video_path: str, start_idx: int) -> List[np.ndarray]:
        """Extract consecutive video frames starting from start_idx"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        frames = []
        
        for _ in range(self.clip_frames):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {start_idx} from {video_path}")
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.target_res)
            frames.append(frame)
            
        cap.release()
        return frames

    def _load_actions(self, action_path: str, start_idx: int, total_frames: int) -> np.ndarray:
        """Load action segment ensuring frame-action alignment"""
        if not os.path.exists(action_path):
            raise FileNotFoundError(f"Action file not found: {action_path}")
            
        try:
            # Load action data
            with open(action_path, 'r') as f:
                action_logs = [line.strip() for line in f.readlines()]
                
            # Skip if misalignment
            if len(action_logs) != total_frames:
                raise RuntimeError(
                    f"Action-video misalignment: "
                    f"{len(action_logs)} actions != {total_frames} frames"
                )
                
            # Extract segment and convert to vectors
            actions = []
            for key_str in action_logs[start_idx : start_idx + self.clip_frames]:
                actions.append(self._map_keyboard_to_vector(key_str))
                
            return np.array(actions)
        except Exception as e:
            raise RuntimeError(f"Error loading actions: {str(e)}")

# Example usage for testing
if __name__ == "__main__":
    # Initialize configuration
    config = Config(stage=1)
    print(f"Stage 1 dataset path: {config.get_dataset_path()}")
    
    # Test Stage1 loader
    stage1_loader = MAGDataLoader(stage=1, config=config)
    img, text = stage1_loader[0]
    print(f"Stage1 sample - Image shape: {img.shape}, Text: '{text[:20]}...'")
    
    # Test Stage2 loader
    config = Config(stage=2)
    stage2_loader = MAGDataLoader(stage=2, config=config)
    video, text = stage2_loader[0]
    print(f"Stage2 sample - Video shape: {video.shape}, Text: '{text[:20]}...'")
    
    # Test Stage3 loader
    config = Config(stage=3)
    stage3_loader = MAGDataLoader(stage=3, config=config)
    video, actions = stage3_loader[0]
    print(f"Stage3 sample - Video shape: {video.shape}, Actions shape: {actions.shape}")

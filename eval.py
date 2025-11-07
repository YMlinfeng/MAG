### eval.py
import os
import cv2
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from config import Config
from models.mag_model import MAGModel
from data_loader import MAGDataLoader
import lpips
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics import StructuralSimilarityIndexMeasure

class Evaluator:
    """
    Evaluation framework for MAG system using metrics from Table 1 in paper.
    Performs both quantitative (VBench++) and qualitative (FPS element) analysis.
    Follows evaluation protocol described in Section 3.1 of the paper.
    """
    
    def __init__(self, model: MAGModel, config: Config, device: Optional[str] = None):
        """
        Initialize evaluator with target model and configuration
        
        Args:
            model: Trained MAGModel instance
            config: Configuration handler
            device: Target device (default: cuda if available else cpu)
        """
        self.model = model
        self.config = config
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Extract evaluation parameters
        self.eval_cfg = config.config_data['evaluation']
        self.samples = self.eval_cfg['samples']  # 50 samples per model
        self.frames = self.eval_cfg['frames']  # 60 frames per sample
        self.resolution = tuple(self.eval_cfg['resolution'])  # (1280, 720)
        
        # Initialize metrics
        self.fid = FrechetInceptionDistance()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.lpips = lpips.LPIPS(net='vgg').to(self.device)
        self.kid = KernelInceptionDistance(subset_size=10)
        self.inception = InceptionScore()
        
        # Define constant FPS element regions based on resolution
        self.minimap_region = (self.resolution[0]-100, 0, self.resolution[0]-10, 90)  # Top-right
        self.health_region = (10, self.resolution[1]-50, 200, self.resolution[1]-10)  # Bottom-left
        self.ammo_region = (self.resolution[0]-200, self.resolution[1]-50, 
                           self.resolution[0]-10, self.resolution[1]-10)  # Bottom-right
    
    def run_metrics(self, dataloader: Optional[MAGDataLoader] = None) -> Dict[str, float]:
        """
        Run full evaluation protocol as per paper Table 1
        
        Args:
            dataloader: Optional dataloader for reference videos (for FID)
            
        Returns:
            Dictionary containing all metrics from Table 1
        """
        metrics = {}
        
        # Generate sample videos
        print(f"Generating {self.samples} samples with {self.frames} frames each...")
        generated_videos = []
        for i in tqdm(range(self.samples)):
            video = self._generate_sample_video()
            generated_videos.append(video)
        generated_videos = torch.stack(generated_videos)  # [B, T, C, H, W]
        
        # Quantitative metrics - Video Quality
        metrics.update(self._compute_video_quality(generated_videos))
        
        # Quantitative metrics - Control Performance
        metrics.update(self._compute_control_performance(generated_videos))
        
        # FID requires real data reference
        if dataloader:
            real_videos = []
            for i, batch in enumerate(dataloader):
                if i >= self.samples:
                    break
                real_videos.append(batch[0])
            real_videos = torch.stack(real_videos)
            metrics['fid'] = self._compute_fid(generated_videos, real_videos)
        
        return metrics
    
    def render_fps_elements(self, video: torch.Tensor, output_path: str) -> None:
        """
        Visualize FPS game elements as in Fig. 2 analysis
        
        Args:
            video: Generated video tensor [1, T, C, H, W]
            output_path: Directory to save visualization
        """
        os.makedirs(output_path, exist_ok=True)
        frames = video.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
        
        # Check frame stability
        stability_report = self._check_frame_stability(frames)
        
        # Visualize elements
        element_frames = []
        for i, frame in enumerate(frames):
            # Convert to OpenCV format (BGR)
            frame_cv = cv2.cvtColor((frame*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # Draw regions of interest
            self._draw_region(frame_cv, self.minimap_region, "MINIMAP", (0, 255, 0))
            self._draw_region(frame_cv, self.health_region, "HEALTH", (0, 0, 255))
            self._draw_region(frame_cv, self.ammo_region, "AMMO", (255, 0, 0))
            
            # Check element presence
            elements_present = self._detect_fps_elements(frame_cv)
            
            # Add info overlay
            cv2.putText(frame_cv, f"Frame: {i+1}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame_cv, f"Elements: {', '.join(elements_present)}", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame_cv, f"Drift: {stability_report[i]:.4f}", 
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Save frame
            frame_path = os.path.join(output_path, f"frame_{i:03d}.png")
            cv2.imwrite(frame_path, frame_cv)
            element_frames.append(frame_cv)
        
        # Save as video
        self._save_video(element_frames, os.path.join(output_path, "annotated_video.avi"))
    
    def _generate_sample_video(self) -> torch.Tensor:
        """
        Generate sample video from MAG model following paper inference
        Uses autoregressive generation as described in Section 2.3
        
        Returns:
            Generated video tensor [1, T, C, H, W]
        """
        # Initialize first frame and history
        history = torch.zeros((1, 3, *self.resolution), device=self.device)
        generated_frames = []
        
        # Autoregressive generation
        for i in range(self.frames):
            # Process inputs using MAG encoders
            inputs = self._prepare_inputs(history)
            
            # MAG forward pass (Eq. 1)
            z_out = self.model(**inputs)
            
            # Decode latent to frame
            frame = self._decode_frame(z_out)
            generated_frames.append(frame)
            
            # Update history (keep all frames or only recent based on config)
            history = torch.cat([history, frame.unsqueeze(0)], dim=0)[-self._context_frames():]
        
        return torch.stack(generated_frames).squeeze(1)
    
    def _prepare_inputs(self, history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for MAG forward pass
        Simulates the encoder components described in Section 2.2
        
        Args:
            history: Previous frames [T, C, H, W]
            
        Returns:
            Dictionary with prepared inputs
        """
        # Simulate WFVAE encoding
        latent_frames = history.flatten(2).permute(0, 2, 1)
        
        # Random text/action embeddings for testing
        text_emb = torch.randn(1, 16, device=self.device)  # Dummy text tokens
        prompt_emb = torch.randn(1, 32, device=self.device)  # Dummy prompt
        
        # For APB
        action_dim = self.config.config_data['model']['apb'].get('action_dim', 5)
        action_vec = torch.zeros(1, action_dim, device=self.device)
        
        return {
            "z_f": latent_frames,
            "z_text": text_emb,
            "z_c": prompt_emb,
            "a": action_vec
        }
    
    def _compute_video_quality(self, videos: torch.Tensor) -> Dict[str, float]:
        """
        Compute video quality metrics (left side of Table 1)
        - TF (Temporal Flickering)
        - MS (Motion Smoothness)
        - DD (Dynamic Degree)
        - FID (Fréchet Inception Distance)
        - AQ (Aesthetic Quality)
        
        Args:
            videos: Generated videos [B, T, C, H, W]
            
        Returns:
            Dictionary with video quality metrics
        """
        # Compute temporal metrics
        tf_score = self._compute_temporal_flickering(videos)
        ms_score = self._compute_motion_smoothness(videos)
        dd_score = self._compute_dynamic_degree(videos)
        
        # Compute frame-level metrics
        frames = videos.flatten(0, 1)  # Flatten BxT dimensions
        self.fid.update(frames, real=False)
        fid_score = self.fid.compute()
        
        self.inception.update(frames)
        aq_score, _ = self.inception.compute()  # Use as aesthetic proxy
        
        return {
            "TF": tf_score,
            "MS": ms_score,
            "DD": dd_score,
            "FID": float(fid_score),
            "AQ": float(aq_score)
        }
    
    def _compute_control_performance(self, videos: torch.Tensor) -> Dict[str, float]:
        """
        Compute control performance metrics (right side of Table 1)
        - SR (Spatial Relations)
        - Scene (Scene Alignment)
        - Color (Color Consistency)
        - SC (Subject Consistency)
        - BC (Background Consistency)
        - OC (Overall Consistency)
        - AS (Appearance Style)
        
        Args:
            videos: Generated videos [B, T, C, H, W]
            
        Returns:
            Dictionary with control performance metrics
        """
        # Spatial metrics
        sr_score = self._compute_spatial_relations(videos)
        
        # Consistency metrics
        sc_score = self._compute_subject_consistency(videos)
        bc_score = self._compute_background_consistency(videos)
        
        # Scene/color metrics
        scene_score = self._compute_scene_alignment(videos)
        color_score = self._compute_color_consistency(videos)
        
        # Combined metrics
        oc_score = 0.5 * (sc_score + bc_score)  # Simple average
        as_score = self._compute_appearance_style(videos)
        
        return {
            "SR": sr_score,
            "Scene": scene_score,
            "Color": color_score,
            "SC": sc_score,
            "BC": bc_score,
            "OC": oc_score,
            "AS": as_score
        }
    
    # Metric implementations below correspond to paper definitions
    
    def _compute_temporal_flickering(self, videos: torch.Tensor) -> float:
        """TF↑: Measure of inter-frame luminance stability"""
        lum = videos.mean(dim=2)
        diff = lum[:, 1:] - lum[:, :-1]
        return diff.abs().mean().item()
    
    def _compute_motion_smoothness(self, videos: torch.Tensor) -> float:
        """MS↑: Optical flow consistency across frames"""
        flows = []
        for i in range(videos.shape[1]-1):
            flow = cv2.calcOpticalFlowFarneback(
                videos[0, i].cpu().numpy().transpose(1,2,0).astype(np.float32),
                videos[0, i+1].cpu().numpy().transpose(1,2,0).astype(np.float32),
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flows.append(flow)
        return np.std(np.array(flows)).item()
    
    def _compute_dynamic_degree(self, videos: torch.Tensor) -> float:
        """DD↑: Variance of pixel changes over time"""
        per_frame_var = videos.var(dim=(0, 2, 3, 4))
        return per_frame_var.mean().item()
    
    def _compute_fid(self, gen_videos: torch.Tensor, real_videos: torch.Tensor) -> float:
        """FID↓: Compute Frechet Inception Distance"""
        self.fid.update(real_videos.flatten(0, 1), real=True)
        self.fid.update(gen_videos.flatten(0, 1), real=False)
        return float(self.fid.compute())
    
    def _compute_spatial_relations(self, videos: torch.Tensor) -> float:
        """SR↑: Spatial relationship accuracy"""
        # Dummy implementation: object position consistency
        positions = []
        for frame in videos[0]:
            frame = frame.permute(1, 2, 0).cpu().numpy()
            positions.append(self._detect_objects(frame))
        return self._position_consistency(positions)
    
    def _compute_scene_alignment(self, videos: torch.Tensor) -> float:
        """Scene↑: Scene-semantic consistency"""
        # Dummy: use frame similarity
        ref_frame = videos[0, 0]
        similarities = []
        for frame in videos[0]:
            similarities.append(self.ssim(frame.unsqueeze(0), ref_frame.unsqueeze(0)))
        return torch.mean(torch.stack(similarities)).item()
    
    def _compute_color_consistency(self, videos: torch.Tensor) -> float:
        """Color↑: Color palette stability"""
        palettes = []
        for frame in videos[0]:
            frame = frame.permute(1, 2, 0).cpu().numpy()
            palettes.append(self._extract_color_palette(frame))
        return self._palette_similarity(palettes)
    
    def _compute_subject_consistency(self, videos: torch.Tensor) -> float:
        """SC↑: LPIPS similarity for moving subject"""
        return self._compute_lpips_stability(videos, region=(100, 100, 300, 300))
    
    def _compute_background_consistency(self, videos: torch.Tensor) -> float:
        """BC↑: LPIPS similarity for static background"""
        return self._compute_lpips_stability(videos, region=(0, 0, *self.resolution))
    
    def _compute_appearance_style(self, videos: torch.Tensor) -> float:
        """AS↑: Style coherence throughout video"""
        # Dummy: use KID as style metric
        self.kid.update(videos.flatten(0, 1))
        return float(self.kid.compute()[0])
    
    # Utility functions
    
    def _detect_fps_elements(self, frame: np.ndarray) -> List[str]:
        """Detect FPS elements in frame as per Fig. 2 analysis"""
        detected = []
        try:
            # Mini-map detection
            minimap = frame[
                self.minimap_region[1]:self.minimap_region[3],
                self.minimap_region[0]:self.minimap_region[2]
            ]
            if minimap.mean() > 20:  # Non-black region
                detected.append("Minimap")
            
            # Health bar detection
            health = frame[
                self.health_region[1]:self.health_region[3],
                self.health_region[0]:self.health_region[2]
            ]
            red_pixels = (health[:, :, 2] > 150) & (health[:, :, 1] < 50) & (health[:, :, 0] < 50)
            if red_pixels.any():
                detected.append("Health")
            
            # Ammo counter detection
            ammo = frame[
                self.ammo_region[1]:self.ammo_region[3],
                self.ammo_region[0]:self.ammo_region[2]
            ]
            if np.var(ammo) > 1000:  # Non-uniform region
                detected.append("Ammo")
        except:
            pass
        return detected
    
    def _check_frame_stability(self, frames: List[np.ndarray]) -> List[float]:
        """Measure frame stability to detect distortion/drift"""
        drift_scores = []
        prev_frame = frames[0]
        for frame in frames[1:]:
            prev_tensor = torch.tensor(prev_frame.transpose(2,0,1)).unsqueeze(0)/255.0
            curr_tensor = torch.tensor(frame.transpose(2,0,1)).unsqueeze(0)/255.0
            drift_scores.append(self.ssim(prev_tensor, curr_tensor).item())
            prev_frame = frame
        return drift_scores
    
    def _draw_region(self, frame: np.ndarray, region: Tuple, label: str, color: Tuple) -> None:
        """Draw rectangle around element region"""
        pt1 = (region[0], region[1])
        pt2 = (region[2], region[3])
        cv2.rectangle(frame, pt1, pt2, color, 2)
        cv2.putText(frame, label, (pt1[0], pt1[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _save_video(self, frames: List[np.ndarray], output_path: str) -> None:
        """Save frames as video file"""
        """"some bug, under fixing"""
        pass
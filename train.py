### train.py
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import lpips  # LPIPS perceptual loss
from config import Config
from models.foundational import FoundationalModel
from models.mag_model import MAGModel
from data_loader import MAGDataLoader
from typing import Dict, Tuple, List, Any

class Trainer:
    """
    Trainer class implementing the three-stage training pipeline for MAG.
    Handles stage-specific training procedures according to the paper methodology.
    """
    
    def __init__(self, stage: int, config: Config, device: str = "cuda"):
        """
        Initialize trainer for specific training stage
        
        Args:
            stage: Training stage (1, 2, or 3)
            config: Configuration handler instance
            device: Target device for training
        """
        self.stage = stage
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.hyperparams = self.config.get_hyperparams()
        
        # Load dataset
        self.dataset = MAGDataLoader(stage, config)
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.hyperparams["batch_size"],
            shuffle=True,
            num_workers=4
        )
        
        # Build stage-specific model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Configure loss functions
        self.loss_functions = self._configure_losses()
        
        # Setup optimizer
        self.optimizer = self._configure_optimizer()
        
        # LPIPS model for perceptual loss
        self.lpips = lpips.LPIPS(net='vgg').to(self.device)
        
    def _build_model(self) -> nn.Module:
        """Build model based on current training stage"""
        if self.stage == 1:
            # Stage 1 uses foundational model
            wfvae = WFVAE(levels=self.config.config_data['encoders']['video'].get('levels', 3))
            mt5 = TextEncoder(model_type='mT5')
            clip = TextEncoder(model_type='CLIP')
            return FoundationalModel(wfvae, mt5, clip, self.config)
        
        # Stage 2 and 3 use MAGModel
        mag_model = MAGModel(self.config.config_data)
        
        if self.stage == 3:
            # Freeze backbone for stage 3
            for param in mag_model.backbone.parameters():
                param.requires_grad = False
                
        return mag_model
        
    def _configure_losses(self) -> Dict[str, callable]:
        """Configure stage-specific loss functions"""
        losses = {"l1": nn.L1Loss()}
        
        if self.stage in (1, 2):
            losses["temporal_smoothness"] = self._temporal_smoothness_loss
        
        if self.stage == 3:
            losses["lpips"] = self._background_lpips_loss
            
        return losses
    
    def _configure_optimizer(self) -> optim.Optimizer:
        """Configure stage-specific optimizer"""
        lr = self.hyperparams["learning_rate"]
        params = self.model.parameters()
        
        if self.stage == 3:
            # Only optimize APB parameters in stage 3
            params = self.model.apb.parameters()
        
        return optim.AdamW(params, lr=lr, weight_decay=self.hyperparams.get("weight_decay", 0.0))
    
    def _temporal_smoothness_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal smoothness loss (Stage 1 and 2)
        
        Args:
            predictions: Predicted frame sequence [B, T, C, H, W]
            targets: Ground truth frame sequence [B, T, C, H, W]
            
        Returns:
            Smoothness loss scalar
        """
        # Compute frame differences
        pred_diff = predictions[:, 1:] - predictions[:, :-1]
        target_diff = targets[:, 1:] - targets[:, :-1]
        
        # Calculate L1 loss on differences
        return nn.L1Loss()(pred_diff, target_diff)
    
    def _background_lpips_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute background LPIPS loss (Stage 3)
        Focuses on non-moving regions to prevent degradation
        
        Args:
            predictions: Predicted frames [B, T, C, H, W]
            targets: Ground truth frames [B, T, C, H, W]
            
        Returns:
            LPIPS perceptual loss on background regions
        """
        # Randomly select fixed position (background remains constant)
        bg_h = torch.randint(0, predictions.shape[3] - 16, (1,)).item()
        bg_w = torch.randint(0, predictions.shape[4] - 16, (1,)).item()
        
        # Extract background patches (first and last frames)
        pred_bg = predictions[:, [0, -1], :, bg_h:bg_h+16, bg_w:bg_w+16]
        target_bg = targets[:, [0, -1], :, bg_h:bg_h+16, bg_w:bg_w+16]
        
        # Compute perceptual loss
        return self.lpips(pred_bg, target_bg).mean()
    
    def _get_masking_params(self, timesteps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create mask for APB modulation (Stage 3)
        Implements Eq.8 from paper
        
        Args:
            timesteps: Number of frames in sequence
            
        Returns:
            Tuple containing:
            - mask_tensor: Binary mask [1, T, 1, 1, 1]
            - noise: Gaussian noise with same shape as inputs
        """
        # Masking parameters from config
        context_frames = self.config.config_data["model"]["apb"].get("context_frames", 4)
        alpha = self.config.config_data["model"]["apb"].get("alpha", 0.3)
        
        # Create mask with context frames (0) and prediction frames (1)
        mask = torch.zeros(timesteps)
        mask[context_frames:] = 1.0
        
        # Expand dimensions for broadcasting
        mask = mask.view(1, timesteps, 1, 1, 1).to(self.device)
        
        # Generate noise
        noise = torch.randn(mask.shape).to(self.device)
        
        return mask, noise * alpha
    
    def train(self, total_steps: Optional[int] = None) -> List[float]:
        """
        Execute training for current stage
        
        Args:
            total_steps: Override steps from config if provided
            
        Returns:
            List of loss values per iteration
        """
        total_steps = total_steps or self.hyperparams.get("steps", 1000)
        progress = tqdm(range(total_steps), desc=f"Training Stage {self.stage}")
        loss_history = []
        
        for step in progress:
            # Load batch data
            batch = next(iter(self.data_loader))
            if batch is None:
                continue  # Skip invalid batches
            
            # Stage-specific data preparation
            inputs, loss_inputs = self._prepare_batch(batch)
            
            # Forward pass
            outputs = self.model(**inputs)
            
            # Calculate loss
            loss = self._compute_loss(outputs, **loss_inputs)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Record and display loss
            loss_value = loss.item()
            loss_history.append(loss_value)
            progress.set_postfix(loss=loss_value)
            
            # Save checkpoint periodically
            if step % 1000 == 0:
                self._save_checkpoint(step)
        
        return loss_history
    
    def _prepare_batch(self, batch_data: Any) -> Tuple[Dict, Dict]:
        """
        Prepare batch data for model inputs and loss calculation
        
        Args:
            batch_data: Raw batch from data loader
            
        Returns:
            Tuple containing:
            - model_inputs: Dictionary for model forward pass
            - loss_inputs: Dictionary for loss calculation
        """
        if self.stage == 1:
            images, texts = batch_data
            return (
                {"x": images.to(self.device), "text": texts},
                {"targets": images.to(self.device)}
            )
        
        if self.stage == 2:
            videos, texts = batch_data
            return (
                {"z_f": videos.to(self.device), "z_text": texts},
                {"targets": videos.to(self.device)}
            )
        
        if self.stage == 3:
            frames, actions = batch_data
            batch_size, timesteps = frames.shape[0], frames.shape[1]
            
            # Apply masking and noise
            mask, noise = self._get_masking_params(timesteps)
            noisy_frames = frames * (1 - mask) + noise
            
            return (
                {"z_f": noisy_frames.to(self.device), "a": actions.to(self.device)},
                {"predictions": frames.to(self.device), "targets": frames.to(self.device)}
            )
        
        raise RuntimeError(f"Invalid training stage: {self.stage}")
    
    def _compute_loss(self, outputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute combined loss for current stage
        
        Args:
            outputs: Model predictions
            **kwargs: Additional inputs for loss calculation
            
        Returns:
            Combined loss tensor
        """
        if self.stage == 1:
            targets = kwargs["targets"]
            return self.loss_functions["l1"](outputs, targets)
        
        if self.stage == 2:
            targets = kwargs["targets"]
            l1_loss = self.loss_functions["l1"](outputs, targets)
            smooth_loss = self.loss_functions["temporal_smoothness"](outputs, targets)
            return 0.8 * l1_loss + 0.2 * smooth_loss
        
        if self.stage == 3:
            predictions = outputs
            targets = kwargs["targets"]
            l1_loss = self.loss_functions["l1"](predictions, targets)
            bg_loss = self.loss_functions["lpips"](predictions, targets)
            return 0.7 * l1_loss + 0.3 * bg_loss
        
        raise RuntimeError(f"Invalid training stage: {self.stage}")
    
    def _save_checkpoint(self, step: int) -> None:
        """
        Save model checkpoint
        
        Args:
            step: Current training step
        """
        checkpoint_dir = self.config.config_data.get("training", {}).get("checkpoint_dir", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": step,
            "stage": self.stage,
            "loss_history": []
        }
        
        filename = f"{checkpoint_dir}/stage_{self.stage}_step_{step:06d}.pt"
        torch.save(checkpoint, filename)
        print(f"Saved checkpoint to {filename}")

# Usage example
if __name__ == "__main__":
    # Create configuration for stage 1
    config = Config(stage=1)
    
    # Initialize trainer
    trainer = Trainer(stage=1, config=config)
    
    # Execute training
    loss_history = trainer.train()
    
    print(f"Stage 1 training complete. Final loss: {loss_history[-1]:.4f}")

### encoders/wfvae.py
import torch
import torch.nn as nn
import pywt
import numpy as np
from typing import Tuple
from config import Config  # Import configuration handler

class WFVAE(nn.Module):
    """
    Wavelet-Featured Variational Autoencoder (WFVAE) implementation using Haar wavelet transform.
    Performs multi-level wavelet decomposition as described in paper Section 2.2 (Eq. 2).
    This is a non-trainable module that performs fixed wavelet transformations.
    """
    
    def __init__(self, levels: int = 3, wavelet: str = "haar"):
        """
        Initialize WFVAE with wavelet parameters.
        
        Args:
            levels: Number of decomposition levels (from config.yaml)
            wavelet: Wavelet type (fixed to "haar" per paper)
        """
        super().__init__()
        self.levels = levels
        self.wavelet = wavelet
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if not isinstance(self.levels, int) or self.levels < 1:
            raise ValueError("Levels must be positive integer")
        if self.wavelet != "haar":
            raise ValueError("Only Haar wavelet is supported per paper specification")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input tensor into wavelet approximation coefficients (low-frequency energy)
        
        Args:
            x: Input tensor (images: [B, C, H, W], videos: [B, C, T, H, W])
            
        Returns:
            Approximation coefficients tensor at deepest decomposition level
        """
        with torch.no_grad():
            return self._transform(x, mode='decomposition')
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode approximation coefficients back to pixel space
        
        Args:
            z: Approximation coefficients tensor
            
        Returns:
            Reconstructed tensor in original shape
        """
        with torch.no_grad():
            return self._transform(z, mode='reconstruction')
    
    def _transform(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Internal method handling wavelet transformations
        
        Args:
            x: Input tensor
            mode: 'decomposition' or 'reconstruction'
            
        Returns:
            Transformed tensor
        """
        # Convert to CPU and numpy for pywt operations
        device = x.device
        x_np = x.detach().cpu().numpy()
        
        # Handle different dimensionalities
        if x.dim() == 4:  # Images [B, C, H, W]
            result = self._process_2d(x_np, mode)
        elif x.dim() == 5:  # Videos [B, C, T, H, W]
            result = self._process_3d(x_np, mode)
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")
        
        # Return to original device
        return torch.from_numpy(result).to(device)
    
    def _process_2d(self, x: np.ndarray, mode: str) -> np.ndarray:
        """
        Process 2D image data (per-channel)
        
        Args:
            x: Input array [B, C, H, W]
            mode: Transformation mode
            
        Returns:
            Transformed array
        """
        batch_size, channels, height, width = x.shape
        
        # Process each batch and channel independently
        result = np.zeros_like(x) if mode == 'reconstruction' else []
        for b in range(batch_size):
            for c in range(channels):
                channel_data = x[b, c]
                
                if mode == 'decomposition':
                    coeffs = self._wavelet_decompose_2d(channel_data)
                    # Store deepest approximation coefficients
                    result.append(coeffs[self.levels - 1])
                else:  # reconstruction
                    approx = channel_data  # Input is the approximation coefficients
                    reconstructed = self._wavelet_reconstruct_2d(approx)
                    result[b, c] = reconstructed
        
        if mode == 'decomposition':
            return np.stack(result)  # [B, C, H//2^L, W//2^L]
        return result
    
    def _process_3d(self, x: np.ndarray, mode: str) -> np.ndarray:
        """
        Process 3D video data (per-channel)
        
        Args:
            x: Input array [B, C, T, H, W]
            mode: Transformation mode
            
        Returns:
            Transformed array
        """
        batch_size, channels, frames, height, width = x.shape
        
        # Process each batch, channel and frame independently
        result = np.zeros_like(x) if mode == 'reconstruction' else []
        for b in range(batch_size):
            for c in range(channels):
                frame_results = []
                for t in range(frames):
                    frame_data = x[b, c, t]
                    
                    if mode == 'decomposition':
                        coeffs = self._wavelet_decompose_2d(frame_data)
                        # Store deepest approximation coefficients
                        frame_results.append(coeffs[self.levels - 1])
                    else:  # reconstruction
                        approx = frame_data  # Input is approximation coefficients
                        reconstructed = self._wavelet_reconstruct_2d(approx)
                        result[b, c, t] = reconstructed
                
                if mode == 'decomposition':
                    result.append(np.stack(frame_results))
        
        if mode == 'decomposition':
            return np.stack(result)  # [B, C, T, H//2^L, W//2^L]
        return result
    
    def _wavelet_decompose_2d(self, data: np.ndarray) -> list:
        """
        Perform multi-level 2D wavelet decomposition
        
        Args:
            data: 2D input array [H, W]
            
        Returns:
            List of wavelet coefficients at each level
        """
        coeffs = [data]
        for _ in range(self.levels):
            coeffs = pywt.wavedec2(coeffs[0], self.wavelet, level=1)
            # Only keep approximation for the next level
            coeffs = [coeffs[0]]
        return coeffs
    
    def _wavelet_reconstruct_2d(self, approx: np.ndarray) -> np.ndarray:
        """
        Reconstruct 2D data from approximation coefficients
        Detail coefficients are set to zero as per paper specification
        
        Args:
            approx: Approximation coefficients [H, W]
            
        Returns:
            Reconstructed 2D array
        """
        coeffs = [approx]
        for _ in range(self.levels):
            # Create detail coefficients filled with zeros
            details = (np.zeros_like(approx), 
                       np.zeros_like(approx), 
                       np.zeros_like(approx))
            coeffs = (coeffs[0], details)
            approx = pywt.waverec2(coeffs, self.wavelet)
            coeffs = [approx]
        return approx

# Example configuration retrieval (for standalone testing)
if __name__ == "__main__":
    # Create config with default stage
    config = Config(stage=2)
    # Get wavelet levels from configuration (fallback to default 3)
    enc_config = config.config_data.get('encoders', {})
    params = enc_config.get('video', {}).get('parameters', {})
    levels = params.get('levels', 3)
    
    # Initialize WFVAE with config parameters
    wfvae = WFVAE(levels=levels)
    
    # Test encoding/decoding with dummy data
    dummy_image = torch.randn(1, 3, 64, 64)  # [B, C, H, W]
    latent = wfvae.encode(dummy_image)
    reconstructed = wfvae.decode(latent)
    
    print(f"Input shape: {dummy_image.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")

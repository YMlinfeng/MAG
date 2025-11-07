### utils/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class ZeroConv1D(nn.Module):
    """
    Implementation of zero-convolution operation for bias matrix Δγ as defined in paper Eq.6-7
    Applies piecewise conditioning to video-control token interactions with zero-initialization
    
    Attributes:
        in_features (int): Input dimension (sequence length)
        out_features (int): Output dimension (sequence length)
        weight (nn.Parameter): Convolution weights
        bias (nn.Parameter): Convolution bias
    """
    
    def __init__(self, in_features: int, out_features: Optional[int] = None):
        """
        Initialize ZeroConv1D module
        
        Args:
            in_features: Input sequence length dimension
            out_features: Output sequence length dimension (defaults to in_features)
        """
        super().__init__()
        out_features = out_features or in_features
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize parameters to zero per paper Eq.7
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights and bias to zero per paper requirement"""
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, base_bias: torch.Tensor) -> torch.Tensor:
        """
        Apply zero-convolution operation to base bias matrix
        
        Args:
            base_bias: Base matrix I scaled by log(1+γ) 
                       Shape: (batch_size, seq_len, seq_len)
                       
        Returns:
            Δγ bias matrix of shape (batch_size, seq_len, seq_len)
        """
        # Apply linear transformation: base_bias @ weight.T + bias
        return F.linear(base_bias, self.weight, self.bias)

def apply_3d_rope(
    x: torch.Tensor, 
    dim: Optional[int] = None, 
    theta: float = 10000.0
) -> torch.Tensor:
    """
    Apply 3D Rotary Position Embedding (RoPE) to video tokens as specified in paper
    
    Args:
        x: Input tensor of shape (batch_size, tokens, features)
        dim: Feature dimension (automatically inferred if None)
        theta: Frequency base parameter (default=10000.0)
    
    Returns:
        Tensor with rotary embeddings applied to video tokens
    
    Raises:
        ValueError: If feature dimension is not even
    """
    if dim is None:
        dim = x.shape[-1]
    if dim % 2 != 0:
        raise ValueError(f"Feature dimension must be even for RoPE, got {dim}")
    
    # Generate positions (linear indices for video tokens)
    device, dtype = x.device, x.dtype
    tokens = x.size(1)
    positions = torch.arange(tokens, device=device, dtype=dtype)  # (tokens)
    
    # Compute frequencies
    half_dim = dim // 2
    exponents = torch.arange(half_dim, device=device, dtype=dtype) / half_dim
    freqs = 1.0 / (theta ** exponents)  # (half_dim)
    
    # Compute angles
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (tokens, half_dim)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    # Split feature dimension
    x1 = x[..., 0::2]  # (batch, tokens, half_dim)
    x2 = x[..., 1::2]
    
    # Apply rotation
    y1 = cos * x1 - sin * x2
    y2 = sin * x1 + cos * x2
    
    # Interleave results
    return torch.stack([y1, y2], dim=-1).flatten(2)  # (batch, tokens, dim)

# Unit tests for module functionality
if __name__ == "__main__":
    # Test ZeroConv1D functionality
    batch_size, seq_len = 2, 5
    base_bias = torch.randn(batch_size, seq_len, seq_len)
    zero_conv = ZeroConv1D(seq_len)
    
    # Verify initial weights are zero
    assert torch.all(zero_conv.weight == 0), "Weight not initialized to zero"
    assert torch.all(zero_conv.bias == 0), "Bias not initialized to zero"
    
    # Test forward pass
    output = zero_conv(base_bias)
    assert output.shape == (batch_size, seq_len, seq_len), "Output shape mismatch"
    assert torch.allclose(output, torch.zeros_like(output)), "Non-zero output with zero-init"
    
    # Test non-zero behavior
    zero_conv.weight.data.fill_(1.0)
    zero_conv.bias.data.fill_(1.0)
    output = zero_conv(base_bias)
    manual_output = base_bias @ zero_conv.weight.T + zero_conv.bias
    assert torch.allclose(output, manual_output), "Forward computation mismatch"
    
    # Test 3DRoPE
    batch_size, tokens, dim = 2, 16, 64
    x = torch.randn(batch_size, tokens, dim)
    rotated = apply_3d_rope(x.clone())
    
    # Verify shape preservation
    assert rotated.shape == x.shape, "RoPE shape mismatch"
    
    # Verify non-trivial transformation
    assert not torch.allclose(rotated, x), "RoPE failed to transform inputs"
    
    print("All attention utils tests passed!")

### models/mag_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from config import Config  # For configuration handling
from utils.attention import ZeroConv1D, apply_3d_rope  # Import attention utilities

class UC3DMMAttn(nn.Module):
    """
    Unified Control 3D Mamba-DiT Attention (UC-3DMMAttn) module.
    Implements attention with bias control (Δγ) for unified spatial/non-spatial alignment
    as described in paper Eqs. (4-7).
    """
    
    def __init__(self, config: dict):
        super().__init__()
        # Get configuration parameters
        self.d_model = config["model"]["uc_3dmm_attn"]["d_model"]
        self.bias_init = config["model"]["uc_3dmm_attn"]["bias_control"]
        self.control_intensity = nn.Parameter(torch.tensor(0.01))  # Initialize γ to 0.01
        
        # Query/Key/Value projections
        self.Wq = nn.Linear(self.d_model, self.d_model)
        self.Wk = nn.Linear(self.d_model, self.d_model)
        self.Wv = nn.Linear(self.d_model, self.d_model)
        
        # Zero-convolution layer for bias matrix Δγ
        self.zero_conv = ZeroConv1D(
            in_features=1,  # Scalar input for each position
            out_features=1  # Output scalar per position
        )
        
        # Layer normalization
        self.ln = nn.LayerNorm(self.d_model)
        
        # Initialize with small values
        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)
        nn.init.xavier_uniform_(self.Wv.weight)

    def compute_region_mask(
        self, 
        frame_len: int, 
        text_len: int,
        prompt_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create region mask for 𝒯, 𝒯', 𝒞 regions as defined in paper Eq. (5)
        
        Args:
            frame_len: Number of frame tokens
            text_len: Number of text tokens
            prompt_len: Number of prompt tokens
            device: Target device
            
        Returns:
            Bias matrix Δγ base tensor
        """
        total_len = text_len + frame_len + prompt_len
        mask = torch.zeros(total_len, total_len, device=device)
        
        # Define regions based on token types
        text_end = text_len
        frame_end = text_end + frame_len
        prompt_end = frame_end + prompt_len
        
        # 𝒯 ∪ 𝒯' region (0)
        mask[:text_end, :] = 0         # Text tokens (𝒯)
        mask[text_end:frame_end, :text_end] = 0  # Frames to text (𝒯')
        
        # 𝒞 region (log(1+γ))
        mask[text_end:frame_end, frame_end:prompt_end] = torch.log(1 + self.control_intensity)
        
        return mask

    def forward(
        self, 
        Z: torch.Tensor, 
        text_len: int,
        frame_len: int,
        prompt_len: int
    ) -> torch.Tensor:
        """
        Forward pass of UC-3DMMAttn with bias control.
        
        Args:
            Z: Concatenated tokens [batch, seq_len, d_model]
            text_len: Number of text tokens
            frame_len: Number of frame tokens
            prompt_len: Number of prompt tokens
            
        Returns:
            Output after attention and modulation
        """
        batch_size, seq_len, _ = Z.shape
        device = Z.device
        
        # Apply layer normalization
        Z = self.ln(Z)
        
        # Compute base matrix for bias control
        base_matrix = self.compute_region_mask(frame_len, text_len, prompt_len, device)
        base_matrix = base_matrix.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, seq_len, seq_len]
        
        # Apply zero-convolution to create Δγ (Eqs. 6-7)
        base_matrix = base_matrix.view(batch_size * seq_len, seq_len)  # Flatten for conv1D
        Δγ = self.zero_conv(base_matrix)  # Apply zero convolution
        Δγ = Δγ.view(batch_size, seq_len, seq_len)  # Reshape to original
        
        # Apply 3D Rotary Position Embedding
        Z = apply_3d_rope(Z)
        
        # Project to Query/Key/Value
        Q = self.Wq(Z)
        K = self.Wk(Z)
        V = self.Wv(Z)
        
        # Compute attention scores (Eq. 4)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        attn_scores += Δγ  # Add bias control
        
        # Apply softmax and attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        return output


class UC3DMMDiTBlock(nn.Module):
    """
    Unified Control 3D Mamba-DiT Block that integrates UC-3DMMAttn and MLP layers.
    Part of the backbone architecture shown in Fig. 1(c) of the paper.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        # Get dimensions from configuration
        self.d_model = config["model"]["uc_3dmm_attn"]["d_model"]
        self.mlp_ratio = config["model"]["mlp_ratio"]  # Typically 4
        
        # Attention module
        self.attn = UC3DMMAttn(config)
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * self.mlp_ratio),
            nn.GELU(),
            nn.Linear(self.d_model * self.mlp_ratio, self.d_model)
        )
        
        # Layer normalizations
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)

    def forward(
        self, 
        Z: torch.Tensor,
        text_len: int,
        frame_len: int,
        prompt_len: int
    ) -> torch.Tensor:
        # Self-attention branch (Eq. 4)
        attn_output = self.attn(Z, text_len, frame_len, prompt_len)
        Z = Z + attn_output
        Z = self.ln1(Z)
        
        # MLP branch
        mlp_output = self.mlp(Z)
        Z = Z + mlp_output
        Z = self.ln2(Z)
        
        return Z


class UC3DMMDiT(nn.Module):
    """
    UC-3DMMDiT backbone comprising multiple UC3DMMDiT blocks.
    Implements the environment modeling component described in Sec. 2.2 of the paper.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        # Get configuration parameters
        self.config = config
        self.num_blocks = config["model"]["num_blocks"]
        self.d_model = config["model"]["uc_3dmm_attn"]["d_model"]
        
        # Create stacked transformer blocks
        self.blocks = nn.ModuleList([
            UC3DMMDiTBlock(config) for _ in range(self.num_blocks)
        ])
        
        # Input/output projection layers
        self.in_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

    def forward(
        self, 
        z_f: torch.Tensor,  # Frame tokens
        z_text: torch.Tensor,  # Text tokens
        z_c: torch.Tensor,  # Prompt tokens
    ) -> torch.Tensor:
        """
        Forward pass of UC-3DMMDiT backbone.
        
        Args:
            z_f: Frame tokens [batch, frame_tokens, d_model]
            z_text: Text tokens [batch, text_tokens, d_model]
            z_c: Prompt tokens [batch, prompt_tokens, d_model]
            
        Returns:
            Environment latent representation
        """
        # Concatenate tokens (Eq. Z = [z_f; z_text; z_c])
        Z = torch.cat([z_text, z_f, z_c], dim=1)
        Z = self.in_proj(Z)
        
        # Get token lengths for attention masks
        batch_size = z_f.size(0)
        text_len = z_text.size(1)
        frame_len = z_f.size(1)
        prompt_len = z_c.size(1)
        
        # Process through transformer blocks
        for block in self.blocks:
            Z = block(Z, text_len, frame_len, prompt_len)
        
        # Output projection
        return self.out_proj(Z)


class APB(nn.Module):
    """
    Action Prompt Block (APB) for real-time entity manipulation.
    Implements Eqs. (8-10) from Sec. 2.2 of the paper.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        # Get dimensions from configuration
        self.action_dim = config["model"]["apb"]["action_dim"]
        self.feature_dim = config["model"]["apb"]["feature_dim"]
        
        # MLP for scale/shift factors (α, β)
        self.affine_mlp = nn.Sequential(
            nn.Linear(self.action_dim, 4 * self.feature_dim),
            nn.ReLU(),
            nn.Linear(4 * self.feature_dim, 2 * self.feature_dim)
        )
        
        # MLP for gating factor g_t ∈ (0,1)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.action_dim, self.feature_dim),
            nn.Sigmoid()
        )
        
        # Initialize with small values
        nn.init.normal_(self.affine_mlp[0].weight, mean=0.0, std=0.02)
        nn.init.normal_(self.gate_mlp[0].weight, mean=0.0, std=0.02)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Modulate features based on action input.
        
        Args:
            z: Input features [batch, ..., feature_dim]
            a: Action vector [batch, action_dim]
            
        Returns:
            Modulated features according to Eqs. (9-10)
        """
        # Save original dimensions for reshaping
        orig_dims = z.shape
        flat_z = z.view(orig_dims[0], -1, orig_dims[-1])  # Flatten non-feature dims
        
        # Normalize features across channels (Eq. 9)
        μ = flat_z.mean(dim=-1, keepdim=True)
        σ = flat_z.std(dim=-1, keepdim=True)
        z_norm = (flat_z - μ) / (σ + 1e-6)
        
        # Compute affine transformation parameters (α, β)
        aff_params = self.affine_mlp(a)
        α, β = torch.split(aff_params, self.feature_dim, dim=-1)
        
        # Expand parameters to match feature dimensions
        α = α.view(z_norm.shape[0], 1, α.size(1))  # [batch, 1, feature_dim]
        β = β.view(z_norm.shape[0], 1, β.size(1))  # [batch, 1, feature_dim]
        
        # Modulate features
        z_mod = α * z_norm + β  # Element-wise affine transform
        
        # Compute gating factor (Eq. 10)
        g_t = self.gate_mlp(a)  # [batch, feature_dim]
        g_t = g_t.view(z_mod.shape[0], 1, g_t.size(1))  # Reshape for broadcasting
        
        # Blend modulated and original features
        z_out = g_t * z_mod + (1 - g_t) * z_norm
        
        # Restore original dimensions
        return z_out.view(orig_dims)


class MAGModel(nn.Module):
    """
    Full MAG model integrating UC-3DMMDiT backbone and APB module.
    Implements overall architecture shown in Fig. 1(a) and Eq. (1).
    """
    
    def __init__(self, config: dict):
        super().__init__()
        # Store configuration
        self.config = config
        
        # Initialize backbone and branch modules
        self.backbone = UC3DMMDiT(config["model"])
        self.apb = APB(config["model"]) if config["model"]["apb"]["enabled"] else None

    def forward(
        self, 
        z_f: torch.Tensor,  # Frame tokens
        z_text: torch.Tensor,  # Text tokens
        z_c: torch.Tensor,  # Prompt tokens
        a: Optional[torch.Tensor] = None  # Optional action vector
    ) -> torch.Tensor:
        """
        Forward pass with optional action modulation.
        
        Args:
            z_f: Frame tokens [batch, frame_tokens, d_model]
            z_text: Text tokens [batch, text_tokens, d_model]
            z_c: Prompt tokens [batch, prompt_tokens, d_model]
            a: Optional action vector [batch, action_dim]
            
        Returns:
            Environment latent with optional entity modulation
        """
        # Backbone processing (f_UC in Eq. 1)
        z_env = self.backbone(z_f, z_text, z_c)
        
        # APB modulation if provided (APB(a) in Eq. 1)
        if a is not None and self.apb is not None:
            z_out = z_env + self.apb(z_env, a)
        else:
            z_out = z_env
        
        return z_out


# Testing and configuration handling
if __name__ == "__main__":
    # Sample configuration
    sample_config = {
        "model": {
            "uc_3dmm_attn": {"d_model": 512},
            "num_blocks": 8,
            "apb": {
                "enabled": True,
                "action_dim": 16,
                "feature_dim": 512
            }
        }
    }
    
    # Create test model
    mag = MAGModel(sample_config)
    
    # Test inputs
    batch_size, frame_tokens, text_tokens, prompt_tokens = 2, 100, 30, 50
    z_f = torch.randn(batch_size, frame_tokens, 512)
    z_text = torch.randn(batch_size, text_tokens, 512)
    z_c = torch.randn(batch_size, prompt_tokens, 512)
    a = torch.randn(batch_size, 16)
    
    # Test forward pass
    out = mag(z_f, z_text, z_c, a)
    print(f"Input shape: {z_f.shape}, {z_text.shape}, {z_c.shape}")
    print(f"Output shape: {out.shape}")
    
    # Test without action
    out_no_action = mag(z_f, z_text, z_c)
    print(f"Output without action: {out_no_action.shape}")

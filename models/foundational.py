### models/foundational.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from config import Config  # Import configuration handler

class LayerNormZero(nn.Module):
    """
    Adaptive LayerNormZero module with scalar modulation as defined in the MAG paper.
    Uses α, β, γ scalars to modulate normalization behavior while maintaining zero-initialization.
    
    Inputs:
        x: Input tensor of shape [B, L, D]
        alpha: Scalar for multiplicative modulation [B]
        beta: Scalar for additive modulation [B]
        gamma: Scalar for input scaling [B]
        
    Output: Modulated tensor of shape [B, L, D]
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim
        
    def forward(self, x: torch.Tensor, 
                gamma: torch.Tensor, 
                alpha: torch.Tensor, 
                beta: torch.Tensor) -> torch.Tensor:
        # Apply input scaling with gamma
        x_scaled = x * (1 + gamma.unsqueeze(1).unsqueeze(1))
        
        # Standard LayerNorm without affine parameters
        u = x_scaled.mean(-1, keepdim=True)
        s = (x_scaled - u).pow(2).mean(-1, keepdim=True)
        x_norm = (x_scaled - u) / torch.sqrt(s + self.eps)
        
        # Apply affine transformation with alpha and beta
        return alpha.unsqueeze(1).unsqueeze(1) * x_norm + beta.unsqueeze(1).unsqueeze(1)

class DiTBlock(nn.Module):
    """
    DiT Block with text conditioning and adaptive normalization
    Implements self-attention, cross-attention, and MLP operations modulated by text conditioning
    and the global scalars α, β, γ from CLIP embeddings.
    """
    def __init__(self, 
                 hidden_size: int, 
                 num_heads: int, 
                 mt5_hidden_size: int,
                 text_proj: Optional[nn.Module] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # LayerNorm with scalar modulation
        self.ln_zero = LayerNormZero(hidden_size)
        
        # Self-attention module
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Cross-attention module for text conditioning
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            kdim=mt5_hidden_size,
            vdim=mt5_hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        
        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Projection for text embeddings
        self.text_proj = text_proj or nn.Linear(mt5_hidden_size, hidden_size)

    def forward(self, 
               x: torch.Tensor, 
               text_emb: torch.Tensor,
               scalars: torch.Tensor) -> torch.Tensor:
        # Split scalars into gamma, alpha, beta
        gamma, alpha, beta = scalars[:, 0], scalars[:, 1], scalars[:, 2]
        
        # Self-attention branch
        residual = x
        x = self.ln_zero(x, gamma, alpha, beta)
        x_attn, _ = self.self_attn(x, x, x)
        x = residual + x_attn
        
        # Cross-attention branch
        residual = x
        x_norm = self.norm1(x)
        text_emb_proj = self.text_proj(text_emb)
        x_cross, _ = self.cross_attn(x_norm, text_emb_proj, text_emb_proj)
        x = residual + x_cross
        
        # MLP branch
        residual = x
        x = self.norm2(x)
        x_mlp = self.mlp(x)
        x = residual + x_mlp
        
        return x

class ScalarsProjection(nn.Module):
    """
    Projects CLIP embeddings to global scalars α, β, γ for all DiT blocks
    Follows zero-initialization requirements from paper
    
    Input: Combined CLIP and timestep embedding [B, D_clip]
    Output: Scalars for all blocks [B, num_blocks, 3]
    """
    def __init__(self, 
                 clip_hidden_size: int, 
                 num_blocks: int,
                 hidden_dim: int = 512):
        super().__init__()
        self.num_blocks = num_blocks
        self.mlp = nn.Sequential(
            nn.Linear(clip_hidden_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3 * num_blocks)
        )
        self._init_weights()
        
    def _init_weights(self):
        """Zero initialization with alpha preset to 1.0"""
        # Initialize weights to zero
        nn.init.zeros_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        
        # Initialize bias for alpha parameters to 1.0
        bias = torch.zeros(3 * self.num_blocks)
        for block_idx in range(self.num_blocks):
            bias[3 * block_idx + 1] = 1.0  # Set alpha dimension to 1.0
        self.mlp[2].bias.data = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scalars = self.mlp(x)
        return scalars.view(x.size(0), self.num_blocks, 3)  # [B, num_blocks, 3]

class FoundationalModel(nn.Module):
    """
    Stage 1 Foundational Model for MAG framework
    DiT-based image generator fine-tuned on 3D game images with OpenSora-style techniques
    Combines WFVAE encoding, mT5 text conditioning, and CLIP semantic modulation
    """
    def __init__(self, 
                 wfvae: nn.Module,
                 text_encoder_mt5: nn.Module,
                 text_encoder_clip: nn.Module,
                 config: Config):
        super().__init__()
        # Store and freeze encoders
        self.wfvae = wfvae
        self.text_encoder_mt5 = text_encoder_mt5
        self.text_encoder_clip = text_encoder_clip
        self._freeze_encoders()
        
        # Get hyperparameters from config
        stage_config = config.get_hyperparams()
        self.batch_size = stage_config["batch_size"]
        self.hidden_size = stage_config.get("hidden_size", 1024)
        self.num_blocks = stage_config.get("num_blocks", 12)
        self.num_heads = stage_config.get("num_heads", 16)
        
        # Get encoder dimensions
        self.mt5_hidden_size = text_encoder_mt5.model.config.d_model
        self.clip_hidden_size = text_encoder_clip.model.config.hidden_size
        
        # Timestep embedding
        self.timestep_embedder = nn.Embedding(
            num_embeddings=1000, 
            embedding_dim=256  # Fixed for compatibility with CLIP
        )
        
        # Projection modules
        self.timestep_proj = nn.Sequential(
            nn.Linear(256, self.clip_hidden_size),
            nn.SiLU(),
            nn.Linear(self.clip_hidden_size, self.clip_hidden_size)
        )
        self.scalars_proj = ScalarsProjection(
            clip_hidden_size=self.clip_hidden_size,
            num_blocks=self.num_blocks
        )
        self.text_proj = nn.Linear(self.mt5_hidden_size, self.hidden_size)
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mt5_hidden_size=self.mt5_hidden_size,
                text_proj=self.text_proj
            ) for _ in range(self.num_blocks)
        ])
        
        # Input/output projections
        self.input_proj = nn.Linear(wfvae.latent_dim, self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, wfvae.latent_dim)
        
    def _freeze_encoders(self):
        """Freeze encoder parameters"""
        for param in self.wfvae.parameters():
            param.requires_grad = False
        for param in self.text_encoder_mt5.parameters():
            param.requires_grad = False
        for param in self.text_encoder_clip.parameters():
            param.requires_grad = False

    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor,
                text: list) -> torch.Tensor:
        # Encode input image
        with torch.no_grad():
            z = self.wfvae.encode(x)  # [B, C, H, W]
        B, C, H, W = z.shape
        
        # Rearrange to token format
        z = z.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, L, C]
        z = self.input_proj(z)  # [B, L, hidden_size]
        
        # Text embeddings
        with torch.no_grad():
            env_text_emb = self.text_encoder_mt5(text)  # [B, T, D_text]
            clip_emb = self.text_encoder_clip(text)  # [B, D_clip]
        
        # Timestep processing
        t_embed = self.timestep_embedder(t)  # [B, 256]
        t_embed = self.timestep_proj(t_embed)  # [B, D_clip]
        
        # Combine embeddings and project to scalars
        combined = t_embed + clip_emb
        scalars = self.scalars_proj(combined)  # [B, num_blocks, 3]
        
        # Process through DiT blocks
        for block_idx, block in enumerate(self.blocks):
            z = block(z, env_text_emb, scalars[:, block_idx, :])
        
        # Project back to latent space
        z = self.output_proj(z)  # [B, L, C]
        z = z.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        return z  # Predicted noise in latent space

# Load configuration for testing
if __name__ == "__main__":
    # Initialize dummy encoders and config
    class DummyWFVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_dim = 16
        def encode(self, x):
            return x
    class DummyTextEncoder(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.model = nn.Identity()
            self.config = type('obj', (object,), {'d_model': hidden_size, 'hidden_size': hidden_size})
        def forward(self, text):
            return torch.randn(len(text), 5, self.config.d_model)
    
    config = Config(stage=1)
    wfvae = DummyWFVAE()
    text_encoder_mt5 = DummyTextEncoder(512)
    text_encoder_clip = DummyTextEncoder(768)
    
    # Initialize and test model
    model = FoundationalModel(wfvae, text_encoder_mt5, text_encoder_clip, config)
    dummy_x = torch.randn(2, 3, 64, 64)
    dummy_t = torch.randint(0, 1000, (2,))
    text = ["dummy"] * 2
    output = model(dummy_x, dummy_t, text)
    
    print(f"Input shape: {dummy_x.shape}")
    print(f"Output shape: {output.shape}")

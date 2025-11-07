### encoders/text_encoder.py
import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer, MT5EncoderModel, MT5Tokenizer
from typing import List, Optional, Tuple

class TextEncoder(nn.Module):
    """
    Text encoder module handling both environmental text (mT5) and auxiliary semantics (CLIP)
    as described in the MAG paper section 2.2. Outputs either sequence embeddings or scalar projections
    depending on configured encoder type.
    """
    def __init__(self, model_type: str, model_name: Optional[str] = None, 
                 max_seq_len: int = 77, device: Optional[str] = None):
        """
        Initialize text encoder with specified parameters
        
        Args:
            model_type: Either 'mT5' for environmental text or 'CLIP' for auxiliary semantics
            model_name: Pretrained model identifier (defaults to 'google/mt5-base' or 'openai/clip-vit-base-patch32')
            max_seq_len: Maximum sequence length for tokenization
            device: Target device ('cuda' or 'cpu')
        """
        super().__init__()
        self.model_type = model_type
        self.max_seq_len = max_seq_len
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set default models if not specified
        if not model_name:
            model_name = "google/mt5-base" if model_type == "mT5" else "openai/clip-vit-base-patch32"
        
        # Initialize tokenizer and model
        if model_type == "mT5":
            self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
            self.model = MT5EncoderModel.from_pretrained(model_name)
        elif model_type == "CLIP":
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.model = CLIPTextModel.from_pretrained(model_name)
            # Projection layer for global scalars (α,β,γ)
            self.scalar_proj = nn.Linear(self.model.config.hidden_size, 3)
            # Initialize projection layer to near-zero values
            nn.init.constant_(self.scalar_proj.weight, 1e-4)
            nn.init.constant_(self.scalar_proj.bias, 0.0)
        else:
            raise ValueError(f"Invalid model type: {model_type}. Must be 'mT5' or 'CLIP'")
        
        # Move models to target device
        self.model = self.model.to(self.device)
        if model_type == "CLIP":
            self.scalar_proj = self.scalar_proj.to(self.device)

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Process batch of input texts to generate embeddings or scalars
        
        Args:
            texts: List of input text strings
            
        Returns:
            For mT5: Tensor of shape [batch_size, seq_len, hidden_dim]
            For CLIP: Tensor of shape [batch_size, 3] containing α,β,γ scalars
        """
        # Tokenize input texts
        tokenized = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        ).to(self.device)
        
        # Process through model
        outputs = self.model(**tokenized)
        
        if self.model_type == "mT5":
            # Return full sequence embeddings
            return outputs.last_hidden_state
        else:  # CLIP
            # Mean pooling across sequence dimension
            pooled = outputs.last_hidden_state.mean(dim=1)
            # Project to 3 scalars (α,β,γ)
            return self.scalar_proj(pooled)

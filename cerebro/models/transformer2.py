import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple

from cerebro.models.modules import FeatureExtractor, RevIn, DynamicTanh
from cerebro.models.base_model import BaseModel

def rmsnorm(x, eps):
    def _norm(y):
        return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)

    return _norm(x.float()).type_as(x)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return rmsnorm(x, self.eps) * self.weight

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time series sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for time series."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)



class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = nn.GELU(approximate="tanh")
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with pre-norm architecture."""
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int, 
        dropout: float = 0.1,
        activation: str = "gelu",
        norm: Optional[nn.Module] = nn.LayerNorm,
        epsilon: float = 1e-6,
        qv_bias: bool = False,
        pre_norm: bool = False
    ):
        super().__init__()
        self.pre_norm = pre_norm
        
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout, add_bias_kv=qv_bias, batch_first=True)
        self.ff = FeedForward(d_model, d_ff, dropout, activation)
        
        self.norm1 = norm(d_model, eps=epsilon) if norm else nn.Identity()
        self.norm2 = norm(d_model, eps=epsilon) if norm else nn.Identity()
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        
        if self.pre_norm:
            # Pre-norm: norm before attention/ff
            if context is not None:
                x_q = self.norm1(x)
                x_kv = self.norm1(context)
                attn_out = self.attn(x_q, x_kv, x_kv, mask,  need_weights=return_attention)
            else:
                x = self.norm1(x)
                attn_out = self.attn(x, x, x, mask, need_weights=return_attention)
                
            if return_attention:
                attn_out, attn_weights = attn_out
            x = x + self.dropout1(attn_out)
            x = x + self.dropout2(self.ff(self.norm2(x)))
        else:
            # Post-norm: norm after attention/ff
            if context is not None:
                attn_out, attn_weights = self.attn(x, context, context, mask, need_weights=return_attention)
            else:
                attn_out, attn_weights = self.attn(x, x, x, mask, need_weights=return_attention)


            x = self.norm1(x + self.dropout1(attn_out))
            x = self.norm2(x + self.dropout2(self.ff(x)))
        
        if return_attention:
            return x, attn_weights
        return x
    
class TransformerCore(nn.Module):   
    """Transformer-based model for cryptocurrency price prediction."""
    
    def __init__(
        self, 
        input_dim: int = 4, 
        hidden_dim: int = 64, 
        n_heads: int = 4,
        activation: str = "gelu",
        norm: Optional[nn.Module] = RMSNorm,
        pre_norm: bool = False,
        num_layers: int = 2,
        dropout: float = 0.0,
        epsilon: float = 1e-6,
        only_last: bool = True,
        seg_length: int = 60, 
        output_dim: int = 3,
        **kwargs
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            output_dim: Number of output features
            seg_length: Length of input sequences
            num_layers: Number of Transformer layers
            conv_kernel: Kernel size for convolution (None to disable)
            pool_kernel: Kernel size for pooling (None to disable)
            dropout: Dropout probability for Transformer
            num_symbols: Maximum number of unique symbols for embedding
            loss_fn: Loss function to use
        """
        super().__init__()
        
        self.only_last = only_last
        
        print("TransformerCore only_last =", only_last)

        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, seg_length)


        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=25)
        
        self.symbol_emb = nn.Embedding(100, hidden_dim)  # Assuming max 100 unique symbols
        # Transformer layers
        self.transformer = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=hidden_dim, 
                n_heads=n_heads, 
                d_ff=hidden_dim*4, 
                dropout=dropout,
                activation=activation,
                norm=norm,
                pre_norm=pre_norm,
                epsilon=epsilon
            ) for _ in range(num_layers)
        ])
        
        self.start_idx = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
                
        self.fc = nn.Linear(hidden_dim, output_dim)
        

    def forward(self, sources, labels=None, symbols=None, **kwargs):
        
        sources = self.feature_extractor(sources)

        sources = self.positional_encoding(sources)

        if symbols is not None:

            x = torch.cat([self.symbol_emb(symbols).unsqueeze(1), sources], dim=1)  
        
        if self.only_last:
            x = sources[:, -1:, :]
            
            for layer in self.transformer:
                x = layer(x, context=sources)

        else:
            x = sources
            for layer in self.transformer:
                x = layer(x)
                
        x = self.fc(x[:, -1, :]).unsqueeze(1)  # (B, 1, output_dim)

        return x
    
        

class TransformerModel(BaseModel):
    def __init__(self, input_features, hidden_dim=64, output_dim=3, seg_length=60, loss_fn=nn.MSELoss(), num_layers=6, norm=RMSNorm, dropout=0.1, n_heads=4, activation="gelu", pre_norm=False, only_last=True, epsilon=1e-6, **kwargs):
        super().__init__(input_features)
        
        self.loss_fn = loss_fn


        self.encoder = TransformerCore(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            activation=activation,
            norm=norm,
            num_layers=num_layers,
            dropout=dropout,
            pre_norm=pre_norm,
            only_last=only_last
        )



    def forward(self, sources, volumes=None, labels=None, symbols=None, **kwargs):
        B, T, C = sources.shape


        x = self.pre_forward(sources, volumes)
        

        

        
        x = self.encoder(x)

     


        return self.post_forward(x, labels)

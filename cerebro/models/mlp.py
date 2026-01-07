from typing import Optional, Tuple
import torch
import torch.nn as nn

from cerebro.models.modules import FeatureExtractor, RevIn
from cerebro.models.base_model import BaseModel


class MLPCore(nn.Module):
    """MLP-based model for cryptocurrency price prediction."""
    
    def __init__(
        self, 
        input_dim: int = 4, 
        hidden_dim: int = 64, 
        output_dim: int = 3, 
        seg_length: int = 60, 
        num_layers: int = 2,
        conv_kernel: int = 5,
        pool_kernel: int = 1,
        dropout: float = 0.0,


        **kwargs
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            output_dim: Number of output features
            seg_length: Length of input sequences
            num_layers: Number of LSTM layers
            conv_kernel: Kernel size for convolution (None to disable)
            pool_kernel: Kernel size for pooling (None to disable)
            dropout: Dropout probability for LSTM
            num_symbols: Maximum number of unique symbols for embedding
            loss_fn: Loss function to use
        """
        super().__init__()


        # Feature extraction
        self.feature_extractor = FeatureExtractor(
            input_dim, 
            hidden_dim, 
            seg_length,
            conv_kernel=conv_kernel,
            pool_kernel=pool_kernel
        )
        
        self.activation = nn.ReLU()
        
        # MLP layers
        mlp_layers = []
        for i in range(num_layers):
            mlp_layers.append(nn.Linear(hidden_dim if i == 0 else hidden_dim, hidden_dim))
            mlp_layers.append(self.activation)
            if dropout > 0.0:
                mlp_layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*mlp_layers)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

    def forward(
        self, 
        sources: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        symbols: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            src: Source tensor of shape (B, T, C)
            tgt: Target tensor of shape (B, 1, C) or (B, output_dim) (optional)
            symbol: Symbol indices of shape (B,) (optional)
        Returns:
            Tuple of (predictions, loss)
            - predictions: Shape (B, 1, output_dim)
            - loss: Scalar loss value (None if tgt is not provided)
        """
        B, T, C = sources.shape

        # Extract features
        src = self.feature_extractor(sources)  # (B, num_segments, hidden_dim)
        
        # src = torch.cat([src, self.start_idx.repeat(B, 1, 1), ], dim=1)  # (B, num_segments+1, hidden_dim)
        x = self.mlp(src)  # (B, num_segments, hidden_dim)


        return x[:, -1, :].unsqueeze(1)

class MLPModel(BaseModel):
    """MLP-based model for cryptocurrency price prediction."""
    
    def __init__(
        self, 
        input_features: list,
        hidden_dim: int = 64, 
        output_dim: int = 3, 
        seg_length: int = 60, 
        num_layers: int = 2,
        conv_kernel: int = 5,
        pool_kernel: int = 1,
        dropout: float = 0.0,
        num_symbols: int = 100,
        loss_fn: nn.Module = None,
        **kwargs
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            output_dim: Number of output features
            seg_length: Length of input sequences
            num_layers: Number of LSTM layers
            conv_kernel: Kernel size for convolution (None to disable)
            pool_kernel: Kernel size for pooling (None to disable)
            dropout: Dropout probability for LSTM
            num_symbols: Maximum number of unique symbols for embedding
            loss_fn: Loss function to use
        """
        super().__init__(input_features=input_features, loss_fn=loss_fn)
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Symbol embedding for conditioning
        self.symbol_emb = nn.Embedding(num_symbols, hidden_dim)

        self.mlp = MLPCore(
            input_dim=len(input_features),
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            seg_length=seg_length,
            num_layers=num_layers,
            conv_kernel=conv_kernel,
            pool_kernel=pool_kernel,
            dropout=dropout
        )


        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.static = False

    def forward(
        self, 
        sources: torch.Tensor,
        volumes: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        symbols: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            src: Source tensor of shape (B, T, C)
            tgt: Target tensor of shape (B, 1, C) or (B, output_dim) (optional)
            symbol: Symbol indices of shape (B,) (optional)
        Returns:
            Tuple of (predictions, loss)
            - predictions: Shape (B, 1, output_dim)
            - loss: Scalar loss value (None if tgt is not provided)
        """
        
        B, T, C = sources.shape

        # Normalize input
        
        x = self.pre_forward(sources, volumes=volumes)  # (B, T, C)

        x = self.mlp(x)  # (B, 1, hidden_dim)

        # Get prediction from last hidden state
        x = self.fc(x)  # (B, 1, output_dim)
        
        if not self.training and self.static:
            x = torch.zeros(B, 1, self.output_dim, device=x.device)
            
        out = self.post_forward(x, labels)

        # Calculate loss if target is provided
        return out


import torch
import torch.nn as nn
from typing import Optional, Tuple
from cerebro.models.base_model import BaseModel
from cerebro.models.mlp import MLPCore


models_dict = {
    'mlp': MLPCore,
    # 'transformer': TransformerCore,
}



class InvWrapper(BaseModel):
    """MLP-based model for cryptocurrency price prediction."""
    
    def __init__(
        self, 
        input_features: list,
        type: str = 'mlp',
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
        super().__init__()

        # Output projection
        self.encoder = models_dict[type](
            input_dim=len(input_features) + (0 if num_symbols is None else 1),
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            seg_length=seg_length,
            num_layers=num_layers,
            conv_kernel=conv_kernel,
            pool_kernel=pool_kernel,
            dropout=dropout,
            num_symbols=num_symbols,
            loss_fn=loss_fn,
            **kwargs
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
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
        
        x = self.pre_forward(sources, volumes=volumes)
        
        enc_out = self.encoder(sources, volumes, symbols)  # (B, T, hidden_dim)
        
        enc_out = enc_out[:, -1, :]  # (B, hidden_dim)
        
        enc_out = torch.cat([enc_out, self.rev_in.scale], dim=-1)  # (B, hidden_dim)
        
        out = self.fc(enc_out).unsqueeze(1)  # (B, 1, output_dim)
        
        
        # Calculate loss if target is provided
        return self.post_forward(out, labels)
from typing import Optional
import torch
import torch.nn as nn
from cerebro.models.modules import FeatureExtractor, RevIn
from cerebro.models.mlp import MLPCore
from typing import Optional, Tuple

class WeightedNorm(nn.Module):
    def __init__(self, output_dim: int, dim: int = -1):
        super().__init__()
        shape = [1] * 3 
        shape[dim] = output_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.zeros(*shape)) # -1 for limit the exponential growth of softmax
        self.dim = dim
        self.softmax = nn.Softmax(dim=self.dim)
        self.tanh = nn.Tanh()
        # self.leverage = nn.Parameter(torch.ones(1, 1, 1))  # learnable leverage parameter
        
    def forward(self, x):
        return self.tanh(x) * (self.softmax(self.weight) * self.output_dim)

class BaseModel(nn.Module):
    def __init__(self, input_features, output_dim, loss_fn=None,  output_norm=None, **kwargs):
        super().__init__()
        
        self.loss_fn = loss_fn 
        
        output_norm = output_norm or self.loss_fn.output_norm
        
        self.ohlc = {'open', 'high', 'low', 'close'}.intersection(set(input_features))
        
        self.vol = {'volume', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume'}.intersection(set(input_features))
        
        self.rev_in = RevIn(len(self.ohlc), scaling_idx=input_features.index('close')) 
        if len(self.vol) > 0:
            self.vol_rev_in = RevIn(len(self.vol), scaling_idx=None)
        else:
            self.vol_rev_in = None
            
        if output_norm == 'rev_in':
            self.output_norm = lambda x: self.rev_in(x, mode='denorm')
        elif output_norm == 'tanh':
            self.output_norm = nn.Tanh()
        elif output_norm == 'sigmoid':
            self.output_norm = nn.Sigmoid()
        elif output_norm == 'weighted':
            print("Using WeightedNorm as output normalization.")
            self.output_norm = WeightedNorm(output_dim=output_dim, dim=-1)
        elif output_norm is None:
            self.output_norm = nn.Identity()
        else:
            raise ValueError(f"Unknown output_norm: {output_norm}")


    def pre_forward(self, sources, volumes=None, **kwargs):

        x = self.rev_in(sources, mode='norm')
        
        if self.vol_rev_in is not None and volumes is not None:
            vol_x = self.vol_rev_in(volumes, mode='norm')
            x = torch.cat([x, vol_x], dim=-1)

        return x
    
    def post_forward(self, x, labels=None):
        output = {}
        output["pred"] = self.output_norm(x)
        output["last"] = self.rev_in.last
        output["scale"] = self.rev_in.scale
        if labels is not None:
            output["loss"] = self.loss_fn(output, labels)
        return output
    
models_dict = {
    'mlp': MLPCore,
    # 'transformer': TransformerCore,
}

class BaseWrapper(BaseModel):
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
        self.fc = nn.Linear(hidden_dim, output_dim)
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
        
        enc_out = self.encoder(x, symbols=symbols)  # (B, 1, hidden_dim)
        
        enc_out = enc_out[:, -1, :]  # (B, hidden_dim)
        
        out = self.fc(enc_out).unsqueeze(1)  # (B, 1, output_dim)
        
        
        # Calculate loss if target is provided
        return self.post_forward(out, labels)
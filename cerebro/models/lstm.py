from typing import Optional, Tuple
import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """Extracts features using convolution and pooling operations."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        seg_length: int, 
        conv_kernel: Optional[int] = 3, 
        pool_kernel: Optional[int] = 2
    ):
        super().__init__()
        
        self.activation = nn.ReLU()

        self.conv1 = nn.Conv1d(
            input_dim, 
            hidden_dim, 
            kernel_size=conv_kernel, 
            padding=conv_kernel//2
        ) if conv_kernel else nn.Identity()

        self.pool = nn.AvgPool1d(
            kernel_size=pool_kernel, 
            stride=pool_kernel
        ) if pool_kernel else nn.Identity()
        
        if pool_kernel and seg_length % pool_kernel != 0:
            raise ValueError(f"seg_length ({seg_length}) must be divisible by pool_kernel ({pool_kernel})")

        self.framing = nn.Conv1d(
            hidden_dim, 
            hidden_dim, 
            kernel_size=seg_length//pool_kernel, 
            stride=seg_length//pool_kernel
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C)
        Returns:
            Output tensor of shape (B, num_segments, hidden_dim)
        """
        B, T, C = x.shape
        
        x = x.transpose(1, 2)  # (B, C, T)
        
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        
        x = self.framing(x)  # (B, hidden_dim, num_segments)
        x = x.transpose(1, 2)  # (B, num_segments, hidden_dim)
        return x


class RevIn(nn.Module):
    """Reversible Instance Normalization for time series."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, std_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        
        self.std = None
        self.last = None
        
        self.std_scale = std_scale

    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C)
            mode: 'norm' for normalization, 'denorm' for denormalization
        Returns:
            Normalized or denormalized tensor
        """
        B, T, C = x.shape
        
        if mode == 'norm':
            # Use close price (index 3) for normalization
            self.std = x[:, :, 3].std(dim=1, keepdim=False) + self.eps
            self.last = x[:, -1, 3]
            
            x_centered = (x - self.last.view(B, 1, 1))
            if self.std_scale:
                x_centered = x_centered / self.std.view(B, 1, 1)
            return x_centered
            
        elif mode == 'denorm':
            if self.std is None or self.last is None:
                raise ValueError("Must call in 'norm' mode before 'denorm' mode")
            
            if self.std_scale:
                x = x * self.std.view(B, 1, 1)
            return x + self.last.view(B, 1, 1)
            
        else:
            raise ValueError(f"Mode must be 'norm' or 'denorm', got '{mode}'")


class LSTMModel(nn.Module):
    """LSTM-based model for cryptocurrency price prediction."""
    
    def __init__(
        self, 
        input_dim: int = 4, 
        hidden_dim: int = 64, 
        output_dim: int = 3, 
        seg_length: int = 60, 
        num_layers: int = 2,
        conv_kernel: int = 3,
        pool_kernel: int = 2,
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
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        
        # Normalization layer
        self.rev_in = RevIn(input_dim)
        
        # Feature extraction
        self.feature_extractor = FeatureExtractor(
            input_dim, 
            hidden_dim, 
            seg_length,
            conv_kernel=conv_kernel,
            pool_kernel=pool_kernel
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Symbol embedding for conditioning
        self.symbol_emb = nn.Embedding(num_symbols, hidden_dim)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

    def forward(
        self, 
        src: torch.Tensor, 
        tgt: Optional[torch.Tensor] = None, 
        symbol: Optional[torch.Tensor] = None
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
        B, T, C = src.shape

        # Normalize input
        src = self.rev_in(src, mode='norm')
        
        # Extract features
        src = self.feature_extractor(src)  # (B, num_segments, hidden_dim)

        # Initialize hidden states with symbol embedding if provided
        if symbol is not None:
            h0 = self.symbol_emb(symbol).unsqueeze(0).repeat(self.num_layers, 1, 1)
        else:
            h0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=src.device)
        
        c0 = torch.zeros_like(h0)

        # LSTM forward pass
        _, (hidden, _) = self.lstm(src, (h0, c0))

        # Get prediction from last hidden state
        x = self.fc(hidden[-1].unsqueeze(1))  # (B, 1, output_dim)
        
        # Denormalize prediction
        x = self.rev_in(x, mode='denorm')
        
        # Calculate loss if target is provided
        loss = None
        if tgt is not None:
            # Handle different target shapes
            if tgt.dim() == 2:
                tgt = tgt.unsqueeze(1)  # (B, output_dim) -> (B, 1, output_dim)
            loss = self.loss_fn(x, tgt)

        return x, loss

    def predict(self, src: torch.Tensor, symbol: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inference-only forward pass without loss calculation.
        
        Args:
            src: Source tensor of shape (B, T, C)
            symbol: Symbol indices of shape (B,) (optional)
        Returns:
            Predictions of shape (B, 1, output_dim)
        """
        with torch.no_grad():
            pred, _ = self.forward(src, tgt=None, symbol=symbol)
        return pred
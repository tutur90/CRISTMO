from typing import Optional, Tuple
import torch
import torch.nn as nn

from cerebro.models.features import FeatureExtractor, RevIn


class LSTMCore(nn.Module):
    """LSTM-based model for cryptocurrency price prediction."""
    
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
        num_symbols: int = 100,
        loss_fn: nn.Module = None,
        symbol_emb: nn.Embedding = None,
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
        
        self.start_idx = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Output projection
        # self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Symbol embedding for conditioning
        self.symbol_emb = nn.Embedding(num_symbols, hidden_dim) if symbol_emb is None else symbol_emb
        
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
        src = self.feature_extractor(src)  # (B, num_segments, hidden_dim)
        
        # src = torch.cat([src, self.start_idx.repeat(B, 1, 1), ], dim=1)  # (B, num_segments+1, hidden_dim)

        # Initialize hidden states with symbol embedding if provided
        if symbols is not None:
            h0 = self.symbol_emb(symbols).unsqueeze(0).repeat(self.num_layers, 1, 1)
        else:
            h0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=src.device)
        
        c0 = torch.zeros_like(h0)

        # LSTM forward pass
        _, (hidden, _) = self.lstm(src, (h0, c0))


        return hidden[-1].unsqueeze(1)


class LSTMModel(nn.Module):
    """LSTM-based model for cryptocurrency price prediction."""
    
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
        
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Symbol embedding for conditioning
        self.symbol_emb = nn.Embedding(num_symbols, hidden_dim)

        self.lstm = LSTMCore(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            conv_kernel=conv_kernel,
            pool_kernel=pool_kernel,
            seg_length=seg_length,
            num_symbols=num_symbols,
            loss_fn=loss_fn,
            symbol_emb=self.symbol_emb,
        )
        
        self.rev_in = RevIn(input_dim)

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

        # Normalize input
        src = self.rev_in(sources, mode='norm')

        x = self.lstm(src, symbols=symbols)  # (B, 1, hidden_dim)

        # Get prediction from last hidden state
        x = self.fc(x)  # (B, 1, output_dim)

        # Calculate loss if target is provided
        loss = None
        if labels is not None:
            loss = self.loss_fn(x, labels, self.rev_in)

        return {"pred": x, "loss": loss}

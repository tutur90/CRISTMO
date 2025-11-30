from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from cerebro.models.modules import RevIn
from pytorch_tcn import TCN






class TCN2Core(nn.Module):
    """TCN core using pytorch_tcn library for feature extraction."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        seg_length: int = 60,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_symbols: int = 100,
        **kwargs
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            seg_length: Length of input sequences
            num_layers: Number of TCN layers
            kernel_size: Kernel size for TCN convolutions
            dropout: Dropout probability
            num_symbols: Maximum number of unique symbols for embedding
        """
        super().__init__()

        self.tcn = TCN(
            num_inputs=input_dim,
            num_channels=[hidden_dim]*num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            activation='leaky_relu',
            use_norm='batch_norm',
        )

        self.hidden_dim = hidden_dim

    def forward(
        self,
        sources: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        symbols: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            sources: Source tensor of shape (B, T, C)
            labels: Optional labels (unused in core)
            symbols: Optional symbol embeddings for hidden state init
        Returns:
            Hidden state tensor of shape (B, 1, hidden_dim)
        """
        # TCN expects (B, C, T) format
        x = self.tcn(sources.permute(0, 2, 1))  # (B, hidden_dim, T)

        # Take last timestep and return as (B, 1, hidden_dim)
        return x[:, :, -1].unsqueeze(1)  # (B, 1, hidden_dim)


class TCN2Model(nn.Module):
    """TCN-based model using pytorch_tcn library for cryptocurrency price prediction."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 3,
        seg_length: int = 60,
        num_layers: int = 4,
        conv_kernel: int = 3,
        dropout: float = 0.2,
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
            num_layers: Number of TCN layers
            kernel_size: Kernel size for TCN convolutions
            dropout: Dropout probability
            num_symbols: Maximum number of unique symbols for embedding
            loss_fn: Loss function to use
        """
        super().__init__()
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()

        # Output projection
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Symbol embedding for conditioning
        self.symbol_emb = nn.Embedding(num_symbols, hidden_dim)

        # TCN core
        self.tcn = TCN2Core(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            seg_length=seg_length,
            num_layers=num_layers,
            kernel_size=conv_kernel,
            dropout=dropout,
            num_symbols=num_symbols,
        )

        self.rev_in = RevIn(input_dim)

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(
        self,
        sources: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        symbols: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            sources: Source tensor of shape (B, T, C)
            labels: Target tensor of shape (B, 1, C) or (B, output_dim) (optional)
            symbols: Symbol indices of shape (B,) (optional)
        Returns:
            Dictionary with:
            - pred: Predictions of shape (B, 1, output_dim)
            - loss: Scalar loss value (None if labels is not provided)
        """
        B, T, C = sources.shape

        # Normalize input
        src = self.rev_in(sources, mode='norm')

        # Get symbol embeddings if provided
        embed_symbols = self.symbol_emb(symbols) if symbols is not None else None

        # TCN forward pass
        x = self.tcn(src, symbols=embed_symbols)  # (B, 1, hidden_dim)

        # Get prediction from hidden state
        x = self.fc(x)  # (B, 1, output_dim)

        # Calculate loss if target is provided
        loss = None
        if labels is not None:
            loss = self.loss_fn(x, labels, self.rev_in)

        return {"pred": x, "loss": loss}

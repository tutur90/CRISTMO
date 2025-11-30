import torch
import torch.nn as nn
from typing import Optional, Tuple



class FeatureExtractor(nn.Module):
    """Extracts features using convolution and pooling operations."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        seg_length: int, 
        conv_kernel: Optional[int] = 5,
        pool_kernel: Optional[int] = 1
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
        ) if pool_kernel > 1 else nn.Identity()
        
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



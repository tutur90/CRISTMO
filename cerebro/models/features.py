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
            # self.std = x[:, :, 3].std(dim=1, keepdim=False) + self.eps
            
            
            
            self.last = x[:, -1, 3].view(B, 1, 1)
            
            x_centered = (x - self.last)

            x_centered = x_centered / self.last

            # print(x_centered.shape)

            self.std = x_centered.abs().max(dim=1, keepdim=False).values[:, 3] + self.eps
            
            # print(self.std.shape)

            if self.std_scale:
                x_centered = x_centered / self.std.view(B, 1, 1)
            # x_centered = x_centered / self.last.view(B, 1, 1)
            return x_centered
            
        elif mode == 'denorm':
            if self.std is None or self.last is None:
                raise ValueError("Must call in 'norm' mode before 'denorm' mode")
            
            if self.std_scale:
                x = x * self.std.view(B, 1, 1)
                
            # x = x * self.last.view(B, 1, 1)
            return x + self.last
            
        else:
            raise ValueError(f"Mode must be 'norm' or 'denorm', got '{mode}'")
from typing import Optional, Tuple
import torch
import torch.nn as nn

class RevIn(nn.Module):
    """Reversible Instance Normalization for time series."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, scale_type: str = 'max', scaling_idx: int = None):
        """
        Args:
            num_features: Number of features (channels) in the input tensor
            eps: Small value to avoid division by zero
            scale_type: Type of scaling to apply ('std' or 'minmax')
            std_scale: Whether to apply standard deviation scaling
        """
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        # self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
        # self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        self.scaling_idx = scaling_idx
        
        self.scale = None
        self.last = None
        
        self.scale_type = scale_type
        
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
            if self.scaling_idx is None:
                self.last = x[:, -1:, :].view(B, 1, C)
            else:
                try:
                    self.last = x[:, -1, self.scaling_idx].view(B, 1, 1)
                except Exception as e:
                    raise ValueError(f"Error accessing scaling_idx {self.scaling_idx}: {e}")
                
                
            x_centered = (x - self.last)

            if self.scale_type == 'std':
                self.scale = x_centered.std(dim=1, keepdim=False) + self.eps
            elif self.scale_type == 'minmax':
                self.scale = self.std
            elif self.scale_type == 'max':
                if self.scaling_idx is not None:
                    self.scale = x_centered.abs().max(dim=1, keepdim=False).values[:, self.scaling_idx] + self.eps
                else:
                    self.scale = x_centered.abs().max(dim=1, keepdim=False).values + self.eps
            elif self.scale_type == 'diff':
                if self.scaling_idx is not None:
                    self.scale = (x_centered.diff(dim=1).max(dim=1, keepdim=False).values[:, self.scaling_idx] )
                else:
                    self.scale = (x_centered.diff(dim=1).max(dim=1, keepdim=False).values ) + self.eps
            elif self.scale_type == 'none':
                self.scale = torch.ones_like(self.std)
            else:
                raise ValueError(f"Unknown scale type: {self.scale_type}")

            x_centered = x_centered / self.scale.view(B, 1, -1)
            return x_centered
            
        elif mode == 'denorm':
            if self.scale is None or self.last is None:
                raise ValueError("Must call in 'norm' mode before 'denorm' mode")

            x = x * self.scale.view(B, 1, 1)
            return x + self.last
            
        else:
            raise ValueError(f"Mode must be 'norm' or 'denorm', got '{mode}'")
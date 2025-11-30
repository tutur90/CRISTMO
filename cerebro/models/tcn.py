from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from cerebro.models.modules import FeatureExtractor, RevIn


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"



class ResidualBlock(nn.Module):
    '''Residual block to use in TCN'''

    def __init__(self, input_size, hidden_size, kernel_size, dilation, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad1 = nn.ConstantPad1d(((self.kernel_size-1)*dilation, 0), 0.0)
        self.conv1 = torch.nn.utils.parametrizations.weight_norm(nn.Conv1d(
            in_channels=self.input_size,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            dilation=dilation,
        ))
        self.dropout1 = nn.Dropout1d(p=dropout)
        self.pad2 = nn.ConstantPad1d(((self.kernel_size-1)*dilation, 0), 0.0)
        self.conv2 = torch.nn.utils.parametrizations.weight_norm(nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            dilation=dilation,
        ))
        self.dropout2 = nn.Dropout1d(p=dropout)
        self.identity_conv = None
        if self.input_size > 1 and self.input_size != self.hidden_size:
            self.identity_conv = nn.Conv1d(
                in_channels=self.input_size,
                out_channels=self.hidden_size,
                kernel_size=1,
            )
            
        self.activation = nn.LeakyReLU()
        
        print(self.hidden_size)
        self.norm = nn.BatchNorm1d(self.hidden_size)

    def forward(self, x):
        '''One step of computation'''
        output = self.pad1(x)
        output = self.activation(self.conv1(output))
        output = self.dropout1(output)
        output = self.pad2(output)
        output = self.activation(self.conv2(output))
        output = self.dropout2(output)
        if self.input_size > 1 and self.input_size != self.hidden_size:
            x = self.identity_conv(x)

        return self.norm(output + x)

class TCN(nn.Module):
    '''Temporal Convolutional Network'''

    def __init__(self, input_size, num_filters, kernel_sizes, dilations, dropout=0.0):
        super(TCN, self).__init__()
        if len(num_filters) != len(kernel_sizes):
            raise ValueError('output_sizes and kernel_sizes must be of the same size')
        if len(kernel_sizes) != len(dilations):
            raise ValueError('kernel_sizes and dilations must be of the same size')
        self.input_size = input_size
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.residuals = nn.Sequential()
        for n in range(len(kernel_sizes)):
            if n == 0:
                self.residuals.append(ResidualBlock(self.input_size, self.num_filters[n], self.kernel_sizes[n], self.dilations[n], dropout=dropout))
            else:
                self.residuals.append(ResidualBlock(self.num_filters[n-1], self.num_filters[n], self.kernel_sizes[n], self.dilations[n], dropout=dropout))
  
    def forward(self, value):
        '''One step of computation'''
        output = self.residuals(value)
        return output


class TCNCore(nn.Module):
    """TCN core model for feature extraction."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        seg_length: int = 60,
        kernels: list = [2,2,2,2],
        dilations: list = [1,2,4,8],
        conv_kernel: int = 5,
        pool_kernel: int = 1,
        num_symbols: int = 100,
        **kwargs
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            seg_length: Length of input sequences
            kernels: List of kernel sizes for TCN layers
            dilations: List of dilation factors for TCN layers
            conv_kernel: Kernel size for feature extraction convolution
            pool_kernel: Kernel size for pooling
            num_symbols: Maximum number of unique symbols for embedding
        """
        super().__init__()

        self.tcn = TCN(
            input_size=input_dim,
            num_filters=[hidden_dim]* (len(kernels)),
            kernel_sizes=kernels,
            dilations=dilations,
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
        x = self.tcn(sources.permute(0, 2, 1))  # (B, hidden_dim, num_segments)

        # Take last timestep and return as (B, 1, hidden_dim)
        return x[:, :, -1].unsqueeze(1)  # (B, 1, hidden_dim)


class TCNModel(nn.Module):
    """TCN-based model for cryptocurrency price prediction."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 3,
        seg_length: int = 60,
        kernels: list = [2,2,2,2],
        dilations: list = [1,2,4,8],
        conv_kernel: int = 5,
        pool_kernel: int = 1,
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
            kernels: List of kernel sizes for TCN layers
            dilations: List of dilation factors for TCN layers
            conv_kernel: Kernel size for convolution
            pool_kernel: Kernel size for pooling
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
        self.tcn = TCNCore(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            seg_length=seg_length,
            kernels=kernels,
            dilations=dilations,
            conv_kernel=conv_kernel,
            pool_kernel=pool_kernel,
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

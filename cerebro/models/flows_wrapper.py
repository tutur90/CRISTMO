from typing import Optional, Tuple
import torch
import torch.nn as nn
from cerebro.models.base_model import BaseModel, BaseWrapper
from cerebro.models.mlp import MLPCore
from cerebro.models.transformer import TransformerCore
import zuko


class NSFWrapper(BaseModel):
    """Neural Spline Flow (NSF) wrapper for probabilistic cryptocurrency price prediction."""

    def __init__(
        self,
        input_features: list,
        model_type: str = 'mlp',
        hidden_dim: int = 64,
        output_dim: int = 3,
        seg_length: int = 60,
        num_layers: int = 2,
        conv_kernel: int = 5,
        pool_kernel: int = 1,
        dropout: float = 0.0,
        num_symbols: int = 100,
        loss_fn: nn.Module = None,
        flow_transforms: int = 3,
        flow_bins: int = 8,
        flow_hidden_features: list = None,
        **kwargs
    ):
        """
        Args:
            input_features: List of input feature names
            model_type: Type of encoder ('mlp' or 'transformer')
            hidden_dim: Hidden dimension size for encoder
            output_dim: Number of output features (distribution dimension)
            seg_length: Length of input sequences
            num_layers: Number of encoder layers
            conv_kernel: Kernel size for convolution (None to disable)
            pool_kernel: Kernel size for pooling (None to disable)
            dropout: Dropout probability
            num_symbols: Maximum number of unique symbols for embedding
            loss_fn: Loss function to use
            flow_transforms: Number of spline transformations in the flow
            flow_bins: Number of bins for the spline
            flow_hidden_features: Hidden layer sizes for flow MLP [default: [hidden_dim, hidden_dim]]
        """
        super().__init__(
            input_features=input_features,
            output_dim=output_dim,
            loss_fn=loss_fn,
            **kwargs
        )

        # Encoder for extracting features from time series
        from cerebro.models.base_model import models_dict
        self.encoder = models_dict[model_type](
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

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.static = False

        # Neural Spline Flow configuration
        if flow_hidden_features is None:
            flow_hidden_features = [hidden_dim, hidden_dim]

        # Create NSF using zuko
        # NSF is a normalizing flow that transforms a simple base distribution
        # (e.g., Gaussian) into a complex distribution using spline transformations
        self.flow = zuko.flows.NSF(
            features=output_dim,  # dimension of the distribution
            context=hidden_dim,   # dimension of conditioning context
            transforms=flow_transforms,  # number of coupling layers
            bins=flow_bins,  # number of bins for spline
            hidden_features=flow_hidden_features  # MLP architecture
        )

    def forward(
        self,
        sources: torch.Tensor,
        volumes: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        symbols: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            sources: Source tensor of shape (B, T, C)
            volumes: Volume tensor (optional)
            labels: Target tensor of shape (B, 1, output_dim) (optional)
            symbols: Symbol indices of shape (B,) (optional)
        Returns:
            Dictionary containing:
            - pred: Predicted mean of the distribution (B, 1, output_dim)
            - loss: Negative log-likelihood loss (if labels provided)
            - last: Last value from RevIn
            - scale: Scale from RevIn
        """
        B, T, C = sources.shape

        # Normalize input
        x = self.pre_forward(sources, volumes=volumes)

        # Extract features using encoder
        enc_out = self.encoder(x, symbols=symbols)  # (B, T, hidden_dim)
        context = enc_out[:, -1, :]  # (B, hidden_dim) - use last timestep as context

        output = {}

        if labels is not None:
            # Training mode: compute negative log-likelihood
            # labels shape: (B, 1, output_dim)
            labels_flat = (labels.squeeze(1) - self.rev_in.last) / self.rev_in.scale  # (B, output_dim)

            # Create flow distribution conditioned on context
            flow_dist = self.flow(context)  # returns a Distribution

            # Compute negative log-likelihood
            nll = -flow_dist.log_prob(labels_flat)  # (B,)
            output["loss"] = nll.mean()

            # For compatibility, also compute prediction (mean of the flow)
            # Sample from the flow and compute mean
            with torch.no_grad():
                samples = flow_dist.sample((100,))  # (100, B, output_dim)
                pred = samples.mean(dim=0).unsqueeze(1)  # (B, 1, output_dim)
        else:
            # Inference mode: sample from the flow
            flow_dist = self.flow(context)

            # Generate prediction as mean of samples
            samples = flow_dist.sample((100,))  # (100, B, output_dim)
            pred = samples.mean(dim=0).unsqueeze(1)  # (B, 1, output_dim)

        output["pred"] = self.output_norm(pred)
        output["last"] = self.rev_in.last
        output["scale"] = self.rev_in.scale


        return output


class FlowsWrapper(BaseWrapper):
    """MLP-based model for cryptocurrency price prediction."""

    def __init__(
        self,
        input_features: list,
        model_type: str = 'mlp',
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
        super().__init__(
                         input_features= input_features,
                        model_type= model_type,
                        hidden_dim= hidden_dim,
                        output_dim= output_dim,
                        seg_length= seg_length,
                    num_layers= num_layers,
                    conv_kernel = conv_kernel,
                    pool_kernel = pool_kernel,
                    dropout = dropout,
                    num_symbols = num_symbols,
                    loss_fn = loss_fn,
                    **kwargs)



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
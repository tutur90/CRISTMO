import torch
import torch.nn as nn
from typing import Optional
from cerebro.models.modules import RevIn

class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, distribution, levels, price):
        takes = (levels > price[:, 1].unsqueeze(1)) * (levels < price[:, 2].unsqueeze(1))
        taken = takes * distribution
        pnl = taken * ((levels - price[:, 3].unsqueeze(1))/levels)
        pnl = pnl.sum(dim=1)
        pnl = torch.log(pnl.clamp(min=1e-8))
        pnl = pnl.mean()
        return -pnl # minimize negative log-likelihood


class RelativeMSELoss(nn.Module):
    """
    Relative Mean Squared Error Loss
    """
    def __init__(self):
        super(RelativeMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: predicted values (tensor)
            y_true: actual values (tensor)

        Returns:
            Relative MSE loss
        """
        return nn.functional.mse_loss(y_pred, y_true)


class MAPE(nn.Module):
    """
    Mean Absolute Percentage Error Loss
    """
    def __init__(self, epsilon=1e-8, use_close=True):
        """
        Args:
            epsilon: small constant to avoid division by zero
        """
        super(MAPE, self).__init__()
        self.epsilon = epsilon
        self.use_close = use_close

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: predicted values (tensor)
            y_true: actual values (tensor)

        Returns:
            MAPE loss as percentage
        """
        if self.use_close:
            y_pred = y_pred[:, -1]  # (B, T)
            y_true = y_true[:, -1]  # (B, T)
        # Add epsilon to avoid division by zero
        loss = torch.mean(torch.abs((y_true - y_pred) / (y_true + self.epsilon))) * 100
        return loss
    
class MSPE(nn.Module):
    """
    Mean Squared Percentage Error Loss
    """
    def __init__(self, epsilon=1e-8, use_close=True):
        """
        Args:
            epsilon: small constant to avoid division by zero
        """
        super(MSPE, self).__init__()
        self.epsilon = epsilon
        self.use_close = use_close  
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: predicted values (tensor)
            y_true: actual values (tensor)
        
        Returns:
            MSPE loss as percentage
        """
        if self.use_close:
            y_pred = y_pred[:, -1]  # (B, T)
            y_true = y_true[:, -1]  # (B, T)
        loss = torch.mean(((y_true - y_pred) / (y_true + self.epsilon) * 100) ** 2) 
        return loss




class BasicInvLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor, rev_in: RevIn, leverage=1, num_items_in_batch: int= None) -> torch.Tensor:
        ret = target[:, 2] - rev_in.last.squeeze()  # (B, T)
        inv = torch.tanh(output.mean(dim=-1).squeeze()) * leverage
        pnl = inv * (ret.exp() - 1) + 1  # (B, T)
        log_pnl = torch.log(pnl.clamp(min=1e-8))

        return -log_pnl.mean() * 24 * 364  # minimize negative log-pnl
    
    
class InvLoss(torch.nn.Module):
    def __init__(self, leverage=1.0, grid_scale: float=1.0, softmax: bool=False, fee=0.01, log_scale: bool=True, **kwargs):
        super().__init__()
        self.leverage = leverage
        self.grid_scale = grid_scale
        self.softmax = softmax
        self.fee = fee
        self.log_scale = log_scale

    def forward(self, output: torch.Tensor, target: torch.Tensor, rev_in: RevIn, num_items_in_batch: int= None) -> torch.Tensor:


        scale = rev_in.scale * self.grid_scale

        grid = torch.linspace(-1, 1, steps=output.shape[-1], device=output.device).unsqueeze(0) * scale.unsqueeze(1)   # (B, D)

        if self.softmax:

            inv = torch.softmax(output, dim=-1) * -torch.sign(grid)

        else:
            inv = output/output.abs().sum(dim=-1, keepdim=True)


        
        grid = grid + rev_in.last.view(-1, 1)  # (B, D)
        
        if self.log_scale:
            ret = target[:, 2].unsqueeze(-1) - (grid)  # (B, T)
        else:
            ret = target[:, 2].unsqueeze(-1)/ (grid) 


        taken_order = (grid > target[:, 0].unsqueeze(-1).repeat(1, grid.shape[1])) * (grid < target[:, 1].unsqueeze(-1).repeat(1, grid.shape[1]))  # (B, D)


        inv = inv * taken_order * self.leverage  # (B, D)

        if self.log_scale:
            pnl = (inv * (ret.exp() - 1)).sum(dim=-1) + 1  # (B, T)
        else:
            pnl = (inv * (ret - 1)).sum(dim=-1) + 1  # (B, T)
            
        pnl = pnl - self.fee/100*inv.abs().sum(dim=-1) # - 4* self.fee/100*inv.sum(dim=-1).abs()

        log_pnl = torch.log(pnl.clamp(min=1e-8)) 

        return -log_pnl.mean() * 24 * 364  # minimize negative log-pnl
import torch
from typing import Optional
from cerebro.models.features import RevIn

class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, distribution, levels, price):
        takes = (levels > price[:, 1].unsqueeze(1)) * (levels < price[:, 2].unsqueeze(1))
        print("takes", takes)
        taken = takes * distribution
        pnl = taken * ((levels - price[:, 3].unsqueeze(1))/levels)
        pnl = pnl.sum(dim=1)
        pnl = torch.log(pnl.clamp(min=1e-8))
        pnl = pnl.mean()
        return -pnl # minimize negative log-likelihood




class RelativeMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor, rev_in: RevIn, num_items_in_batch: int= None) -> torch.Tensor:
        
        target = target.unsqueeze(1)

        output = rev_in(output, mode='denorm')

        if output.shape != target.shape:
            raise ValueError(f"Output shape {output.shape} does not match target shape {target.shape}")
        
        # Compute the mean squared error
        mse = torch.mean((output - target) ** 2)

        # Compute the relative mean squared error
        relative_mse = mse

        return relative_mse * 1e4  # scale factor to keep loss values manageable


class BasicInvLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor, rev_in: RevIn, leverage=1, num_items_in_batch: int= None) -> torch.Tensor:
        

        # if output.shape != target.shape:
        #     raise ValueError(f"Output shape {output.shape} does not match target shape {target.shape}")
        
        
        ret = target[:, 2] - rev_in.last.squeeze()  # (B, T)
        
        # print("ret", ret)
        
        inv = torch.tanh(output.mean(dim=-1).squeeze()) * leverage
        
        
        pnl = inv * (ret.exp() - 1) + 1  # (B, T)
        
        log_pnl = torch.log(pnl.clamp(min=1e-8))    
        
        # print("log_pnl", log_pnl)
        
        # inv = output.mean(dim=-1) * (target[:, :, 2]-last.view(-1, 1)) / (last.view(-1, 1)) 
        # inv = -torch.log(1 + inv.clamp(min=-0.999))  # prevent log(0)


        return -log_pnl.mean() * 24 * 364  # minimize negative log-pnl
    
    
class InvLoss(torch.nn.Module):
    def __init__(self, leverage=1.0, grid_scale: float=1.0, softmax: bool=False, fee=0.01, log_scale: bool=False, **kwargs):
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

            inv = torch.softmax(output, dim=-1)

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
            pnl = (inv * (ret.exp() - 1)).sum(dim=-1) - self.fee/100*inv.abs().sum(dim=-1) + 1  # (B, T)
        else:
            pnl = (inv * (ret - 1)).sum(dim=-1) - self.fee/100*inv.abs().sum(dim=-1) + 1  # (B, T)

        log_pnl = torch.log(pnl.clamp(min=1e-8))

        return -log_pnl.mean() * 24 * 364  # minimize negative log-pnl
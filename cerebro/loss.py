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
        relative_mse = mse / (torch.mean(target ** 2) + 1e-8)

        return relative_mse * 1e5
    
    
class BasicInvLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor, rev_in: RevIn, leverage=1, num_items_in_batch: int= None) -> torch.Tensor:
        

        # if output.shape != target.shape:
        #     raise ValueError(f"Output shape {output.shape} does not match target shape {target.shape}")
        
        
        ret = target[:, 2]/rev_in.last.squeeze()  # (B, T)
        
        # print("ret", ret)
        
        inv = torch.tanh(output.mean(dim=-1).squeeze()) * leverage
        
        
        pnl = inv * (ret - 1) + 1  # (B, T)
        
        log_pnl = torch.log(pnl.clamp(min=1e-8))    
        
        # print("log_pnl", log_pnl)
        
        # inv = output.mean(dim=-1) * (target[:, :, 2]-last.view(-1, 1)) / (last.view(-1, 1)) 
        # inv = -torch.log(1 + inv.clamp(min=-0.999))  # prevent log(0)


        return -log_pnl.mean() * 24 * 364  # minimize negative log-pnl
    
    
class InvLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor, rev_in: RevIn, leverage=1, grid_scale: float=1.0, num_items_in_batch: int= None) -> torch.Tensor:
        

        # if output.shape != target.shape:
        #     raise ValueError(f"Output shape {output.shape} does not match target shape {target.shape}")

        scale = rev_in.scale * grid_scale

        # print("scale", scale.shape)

        grid = torch.linspace(-1, 1, steps=output.shape[-1], device=output.device).unsqueeze(0) * scale.unsqueeze(1) + rev_in.last.view(-1, 1)  # (B, D)

        ret = target[:, 2].unsqueeze(-1)/( grid)  # (B, T)


        taken_order = (grid > target[:, 0].unsqueeze(-1).repeat(1, grid.shape[1])) * (grid < target[:, 1].unsqueeze(-1).repeat(1, grid.shape[1]))  # (B, D)


        # print("ret", ret)

        # print("taken_order", taken_order.sum(dim=1))

        inv = output/output.abs().sum(dim=-1, keepdim=True) * leverage * taken_order
        
        # print("inv", inv)


        pnl = (inv * (ret - 1)).mean(dim=-1) + 1  # (B, T)

        log_pnl = torch.log(pnl.clamp(min=1e-8))

        # print("log_pnl", log_pnl)
        
        # inv = output.mean(dim=-1) * (target[:, :, 2]-last.view(-1, 1)) / (last.view(-1, 1)) 
        # inv = -torch.log(1 + inv.clamp(min=-0.999))  # prevent log(0)


        return -log_pnl.mean() * 24 * 364  # minimize negative log-pnl
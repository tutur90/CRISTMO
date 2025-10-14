import torch

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

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        target = target.unsqueeze(1)
        
        if output.shape != target.shape:
            raise ValueError(f"Output shape {output.shape} does not match target shape {target.shape}")
        
        # Compute the mean squared error
        mse = torch.mean((output - target) ** 2)

        # Compute the relative mean squared error
        relative_mse = mse / (torch.mean(target ** 2) + 1e-8)

        return relative_mse
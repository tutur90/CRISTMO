from typing import Optional
import torch
import torch.nn as nn
from cerebro.models.modules import FeatureExtractor, RevIn

class WeightedNorm(nn.Module):
    def __init__(self, output_dim: int, dim: int = -1):
        super().__init__()
        shape = [-1] * 3 # -1 for limit the exponential growth of softmax
        shape[dim] = output_dim
        self.weight = nn.Parameter(torch.ones(*shape))
        self.dim = dim
        self.softmax = nn.Softmax(dim=self.dim)
        
    def forward(self, x):
        return x * self.softmax(self.weight)

class BaseModel(nn.Module):
    def __init__(self, input_features, output_dim, loss_fn=None,  output_norm=None, **kwargs):
        super().__init__()
        
        self.loss_fn = loss_fn 
        
        output_norm = output_norm or self.loss_fn.output_norm
        
        self.ohlc = {'open', 'high', 'low', 'close'}.intersection(set(input_features))
        
        self.vol = {'volume', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume'}.intersection(set(input_features))
        
        self.rev_in = RevIn(len(self.ohlc), scaling_idx=input_features.index('close')) 
        if len(self.vol) > 0:
            self.vol_rev_in = RevIn(len(self.vol), scaling_idx=None)
        else:
            self.vol_rev_in = None
            
        if output_norm == 'rev_in':
            self.output_norm = lambda x: self.rev_in(x, mode='denorm')
        elif output_norm == 'tanh':
            self.output_norm = nn.Tanh()
        elif output_norm == 'sigmoid':
            self.output_norm = nn.Sigmoid()
        elif output_norm == 'weighted':
            self.output_norm = WeightedNorm(output_dim=output_dim, dim=-1)
        else:
            self.output_norm = lambda x, mode: x  # identity


    def pre_forward(self, sources, volumes=None, **kwargs):

        x = self.rev_in(sources, mode='norm')
        
        if self.vol_rev_in is not None and volumes is not None:
            vol_x = self.vol_rev_in(volumes, mode='norm')
            x = torch.cat([x, vol_x], dim=-1)

        return x
    
    def post_forward(self, x, labels=None):
        output = {}
        output["pred"] = self.output_norm(x)
        output["last"] = self.rev_in.last
        output["scale"] = self.rev_in.scale
        if labels is not None:
            output["loss"] = self.loss_fn(output, labels)
        return output
    
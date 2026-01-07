from typing import Optional
import torch
import torch.nn as nn
from cerebro.models.modules import FeatureExtractor, RevIn

class BaseModel(nn.Module):
    def __init__(self, input_features, loss_fn=None, **kwargs):
        super().__init__()
        
        self.loss_fn = loss_fn 
        
        self.ohlc = {'open', 'high', 'low', 'close'}.intersection(set(input_features))
        
        self.vol = {'volume', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume'}.intersection(set(input_features))
        
        self.rev_in = RevIn(len(self.ohlc), scaling_idx=input_features.index('close')) 
        if len(self.vol) > 0:
            self.vol_rev_in = RevIn(len(self.vol), scaling_idx=None)
        else:
            self.vol_rev_in = None


    def pre_forward(self, sources, volumes=None, **kwargs):

        x = self.rev_in(sources, mode='norm')
        
        if self.vol_rev_in is not None and volumes is not None:
            vol_x = self.vol_rev_in(volumes, mode='norm')
            x = torch.cat([x, vol_x], dim=-1)

        return x
    
    def post_forward(self, x, labels=None):
        output = self.loss_fn.post_forward(x, rev_in=self.rev_in)
        if labels is not None:
            output["loss"] = self.loss_fn(output, labels)
        return output
    
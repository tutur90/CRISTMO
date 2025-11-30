import math
import torch
import torch.nn as nn
from cerebro.models.features import FeatureExtractor, RevIn
from cerebro.models.modules.dytanh import DynamicTanh

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=24):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
        

class TransformerModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3, seg_length=60, loss_fn=nn.MSELoss(), num_layers=6, norm=nn.LayerNorm, dropout=0.1, **kwargs):
        super().__init__()
        self.loss_fn = loss_fn
        
        self.rev_in = RevIn(input_dim)
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, seg_length)

    
        
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=25)
        
        self.symbol_emb = nn.Embedding(100, hidden_dim)  # Assuming max 100 unique symbols

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*4, dropout=dropout, activation='relu', batch_first=True),
            num_layers=num_layers, norm=norm(hidden_dim) if norm else None
        )
        
        
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, sources, labels=None, symbols=None, **kwargs):
        B, T, C = sources.shape

        x = self.rev_in(sources, mode='norm')
        x = self.feature_extractor(x)

        # if symbols is not None:

        #     x = torch.cat([x, self.symbol_emb(symbols).unsqueeze(1)], dim=1)

        x = self.positional_encoding(x)

        x = self.encoder(x, is_causal=False)

     
        x = self.fc(x[:, -1, :]).unsqueeze(1)  # (B, 1, output_dim)

        loss = None
        if labels is not None:
            loss = self.loss_fn(x, labels, self.rev_in)

        return {"pred": x, "loss": loss}

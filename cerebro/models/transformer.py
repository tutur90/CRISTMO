import math
import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, seg_length=60, conv_kernel=5, pool_kernel=1):
        super().__init__()
        
        self.activation = nn.ReLU()

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=conv_kernel, padding=conv_kernel//2)
        # self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=conv_kernel, padding=conv_kernel//2)

        self.pool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_kernel)
        
        if seg_length % pool_kernel != 0:
            raise ValueError("seg_length must be divisible by pool_kernel")

        self.framing = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=seg_length//pool_kernel, stride=seg_length//pool_kernel)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        
        x = x.transpose(1, 2)  # (B, C, T)
        
        x = self.activation(self.conv1(x))
        # x = self.activation(self.conv2(x))
        x = self.pool(x)
        
        x = self.framing(x)  # (B, hidden_dim, num_segments)
        x = x.transpose(1, 2)  # (B, num_segments, hidden_dim)
        return x
    
class RevIn(nn.Module):
    def __init__(self, num_features, eps=1e-5, std_scale=True):
        super(RevIn, self).__init__()
        self.eps = eps
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        
        self.std = None
        self.last = None
        
        self.std_scale = std_scale

    def forward(self, x, mode='norm'):
        B, T, C = x.shape
        if mode == 'norm':
            self.std = x[:, :, 3].std(dim=1, keepdim=False) + 1e-5
            self.last = x[:, -1, 3]
            x_centered = (x - self.last.view(B, 1, 1))
            if self.std_scale:
                x_centered = x_centered / self.std.view(B, 1, 1)
            return x_centered
        elif mode == 'denorm':
            if self.std is None or self.last is None:
                raise ValueError("Must call in 'norm' mode before 'denorm' mode")
            if self.std_scale:
                x = x * self.std.view(B, 1, 1)
            return x + self.last.view(B, 1, 1)
        else:
            raise ValueError("Mode must be 'norm' or 'denorm'")
        
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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
        

class TransformerModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3, seg_length=60, loss_fn=nn.MSELoss(), num_layers=6, norm=None, **kwargs):
        super().__init__()
        self.loss_fn = loss_fn
        
        self.rev_in = RevIn(input_dim)
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, seg_length)


        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=25)
        
        self.symbol_emb = nn.Embedding(100, hidden_dim)  # Assuming max 100 unique symbols

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*4, dropout=0.1, activation='relu', batch_first=True),
            num_layers=num_layers, norm=norm(hidden_dim) if norm else None
        )
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*4, dropout=0.1, activation='relu', batch_first=True),
            num_layers=num_layers, norm=norm(hidden_dim) if norm else None
        )
        
        self.bos = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor = None, **kwargs):
        B, T, C = src.shape

        x = self.rev_in(src, mode='norm')
        x = self.feature_extractor(x)
        
        if "symbol" in kwargs and kwargs["symbol"] is not None:

            x = torch.cat([x, self.symbol_emb(kwargs["symbol"]).unsqueeze(1)], dim=1)

        x = self.positional_encoding(x)

        x = self.encoder(x, is_causal=False)

        context = x.mean(dim=1, keepdim=True)  # (B, 1, hidden_dim)

        context = torch.cat([context, self.bos.expand(B, -1, -1)], dim=1)  # (B, 2, hidden_dim)


        x = self.decoder(context, x, tgt_is_causal=False, memory_is_causal=False)

        # print(x.shape)

        x = self.fc(x[:, :1])  # (B, 1, output_dim)

        x = self.rev_in(x, mode='denorm')
        
        
        if tgt is not None:
            loss = self.loss_fn(x, tgt)
            return x, loss
        else:
            return x, None




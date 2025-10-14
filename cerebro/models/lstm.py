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
        
        

class LSTMModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3, seg_length=60, loss_fn=nn.MSELoss(), **kwargs):
        super().__init__()
        self.loss_fn = loss_fn
        
        self.rev_in = RevIn(input_dim)
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, seg_length)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, tgt=None, symbol=None):
        B, T, C = src.shape

        src = self.rev_in(src, mode='norm')
        src = self.feature_extractor(src)
        _, (hidden, _) = self.lstm(src)
        
        x = self.fc(hidden[-1].unsqueeze(1))
        x = self.rev_in(x, mode='denorm')
        
        loss = self.loss_fn(x, tgt) if tgt is not None else None

        return x, loss



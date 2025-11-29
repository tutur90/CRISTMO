import torch
import torch.nn as nn

from cerebro.loss import Loss

class TimeSeriesModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=32, seg_length=60, **kwargs):
        super().__init__()
        self.embedding = nn.Linear(input_dim * seg_length, hidden_dim)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.loss_fn = Loss()
        self.seg_length = seg_length
        self.latitude = nn.Parameter(torch.linspace(-100, 100, steps=output_dim).unsqueeze(0), requires_grad=False)

    def forward(self, x):
        B, T, C = x.shape
        num_segments = T // self.seg_length
        x_last = x[:, -1, 3].unsqueeze(-1)
        x_centered = x - x_last.unsqueeze(1)
        x_centered = x_centered.view(B, num_segments, self.seg_length * C)
        x_centered = self.relu(self.embedding(x_centered))
        _, (hidden, _) = self.lstm(x_centered)
        return self.fc(hidden[-1]), x_last + self.latitude


    
class LSTMModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3, num_layers=2, seg_length=60, loss_fn=nn.MSELoss(), **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=seg_length, stride=seg_length)
        self.loss_fn = loss_fn
        self.latitude = nn.Parameter(torch.linspace(-100, 100, steps=output_dim).unsqueeze(0), requires_grad=False)
        self.std_scale = True

    def forward(self, x):
        B, T, C = x.shape

        std = x[:, :, 3].std(dim=1, keepdim=False) + 1e-5
        x_last = x[:, -1, 3]
        x_centered = (x - x_last.view(B, 1, 1))

        if self.std_scale:
            x_centered = x_centered / std.view(B, 1, 1)

        x = self.conv(x_centered.permute(0, 2, 1)).permute(0, 2, 1)

        x, (hidden, _) = self.lstm(x)
        
        x = self.fc(hidden[-1])

        if self.std_scale:
            x = x * std.view(B, 1)

        return x_last.view(B, 1) + x, x_last.view(B, 1) + self.latitude

class SimpleModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=3, seg_length=60, **kwargs):
        super().__init__()
        self.latitude = torch.linspace(-100, 100, steps=output_dim).unsqueeze(0)
        self.inv = nn.Linear(input_dim, output_dim)
        self.loss_fn = Loss()

    def forward(self, x):
        
        x_last = x[:, -1, 3].unsqueeze(-1)

        x = self.inv(x[:, -1] - x_last)

        x = x + x_last.unsqueeze(1)

        return x, x_last + self.latitude
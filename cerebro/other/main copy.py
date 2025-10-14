import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import yaml
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import os

from tqdm import tqdm


def custom_ohlc_aggregation(df, window):
    rolled = df.rolling(window, min_periods=window)

    ohlc = rolled.agg({
        'Open': lambda x: x.iloc[0],
        'High': 'max',
        'Low': 'min',
        'Close': lambda x: x.iloc[-1]
    })

    return ohlc.dropna()


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
    
class SimpleLoss(torch.nn.Module):
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



class TimeSeriesModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=32, seg_length=60, **kwargs):
        super().__init__()
        self.embedding = nn.Linear(input_dim * seg_length, hidden_dim)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.loss_fn = Loss()
        self.seg_length = seg_length
        self.latitude = torch.linspace(-100, 100, steps=output_dim).unsqueeze(0)

    def forward(self, x):
        B, T, C = x.shape
        num_segments = T // self.seg_length
        x_last = x[:, -1, 3].unsqueeze(-1)
        x_centered = x - x_last.unsqueeze(1)
        x_centered = x_centered.view(B, num_segments, self.seg_length * C)
        x_centered = self.relu(self.embedding(x_centered))
        _, (hidden, _) = self.lstm(x_centered)
        return self.fc(hidden[-1]), x_last + self.latitude
    
class SimpleModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=32, seg_length=60, **kwargs):
        super().__init__()
        self.latitude = torch.linspace(-100, 100, steps=output_dim).unsqueeze(0)
        self.inv = nn.Parameter(torch.randn(1, 1, output_dim))
        self.loss_fn = Loss()

    def forward(self, x):
        
        x_last = x[:, -1, 3].unsqueeze(-1)

        return self.inv, x_last + self.latitude


class StockDataset(Dataset):
    def __init__(self, data, src_length=24, tgt_length=1, seg_length=60, **kwargs):
        super().__init__()
        print("Loading data...")
        self.src_length = src_length * seg_length
        self.tgt_length = tgt_length * seg_length
        self.data = data.reset_index(drop=True)
        target_df = custom_ohlc_aggregation(data[['Open', 'High', 'Low', 'Close']], self.tgt_length)
        self.features = torch.tensor(self.data.values[:-self.tgt_length], dtype=torch.float32)
        self.targets = torch.tensor(target_df.values[self.src_length:], dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        src = self.features[index : index + self.src_length]
        tgt = self.targets[index]
        return src, tgt


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(model, dataloader, optimizer, writer, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.view(inputs.size(0), -1, 4)  # reshape
        optimizer.zero_grad()
        outputs, levels = model(inputs)
        loss = model.loss_fn(outputs, levels, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        pbar.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar("Loss/train", avg_loss)
    return avg_loss


def evaluate(model, dataloader, writer, device, stage="val"):
    model.eval()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Evaluating ({stage})", leave=False)
    
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1, 4)
            outputs, levels = model(inputs)
            loss = model.loss_fn(outputs, levels, targets)
            total_loss += loss.item()

            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    writer.add_scalar(f"Loss/{stage}", avg_loss)
    return avg_loss

def load_data(hparams, stage="train"):
    data_path = hparams['data']['data_path']
    df = pd.read_feather(f"{data_path}/{stage}_data.feather")
    df.sort_index(inplace=True)
    
    dataset = StockDataset(df, **hparams["data"])
    dataloader = DataLoader(dataset, batch_size=hparams["data"]["batch_size"], shuffle=(stage == "train"))

    return dataloader

def main():
    parser = ArgumentParser()
    parser.add_argument("hparams", type=str, default="configs/train.yaml", help="Path to config file.")
    args = parser.parse_args()

    hparams = load_config(args.hparams)
    torch.manual_seed(hparams["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=hparams["logger"]["save_dir"])

    train_loader = load_data(hparams, stage="train")
    val_loader = load_data(hparams, stage="val")


    model = SimpleModel(**hparams["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["model"]["lr"])

    for epoch in range(hparams["trainer"]["max_epochs"]):
        print(f"Epoch {epoch+1}/{hparams['trainer']['max_epochs']}")
        train_loss = train(model, train_loader, optimizer, writer, device)
        val_loss = evaluate(model, val_loader, writer, device, stage="val")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    test_loader = load_data(hparams, stage="test")
    test_loss = evaluate(model, test_loader, writer, device, stage="test")
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
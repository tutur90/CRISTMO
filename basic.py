import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import random
from typing import Dict, Any, Optional

from cerebro.dataset import CryptoDataset
from cerebro.loss import RelativeMSELoss
from cerebro.models.lstm import LSTMModel
from cerebro.models.transformer import TransformerModel


class CryptoDataModule(pl.LightningDataModule):
    """Lightning DataModule for cryptocurrency dataset."""
    
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        **dataset_kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = dataset_kwargs
        
    def setup(self, stage: Optional[str] = None):
        """Setup train, val, and test datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = CryptoDataset(
                **self.dataset_kwargs,
                split="train"
            )
            self.val_dataset = CryptoDataset(
                **self.dataset_kwargs,
                split="val"
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = CryptoDataset(
                **self.dataset_kwargs,
                split="test"
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )


class LightningCryptoModel(pl.LightningModule):
    """Lightning wrapper for crypto prediction models."""
    
    def __init__(
        self,
        model_type: str,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer: str = "adam",
        scheduler: Optional[str] = None,
        scheduler_kwargs: Optional[Dict] = None,
        **model_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model
        loss_fn = RelativeMSELoss()
        if model_type == "lstm":
            self.model = LSTMModel(**model_kwargs, loss_fn=loss_fn)
        elif model_type == "transformer":
            self.model = TransformerModel(**model_kwargs, loss_fn=loss_fn)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.loss_fn = loss_fn
        
    def forward(self, x, symbol=None):
        return self.model(x, symbol=symbol)
    
    def training_step(self, batch, batch_idx):
        # Unpack batch - expecting (x, y) or (x, y, symbol)
        if len(batch) == 3:
            x, y, symbol = batch
        else:
            x, y = batch
            symbol = None
        
        # Forward pass - model returns (prediction, loss)
        y_hat, loss = self.model(x, tgt=y, symbol=symbol)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Unpack batch
        if len(batch) == 3:
            x, y, symbol = batch
        else:
            x, y = batch
            symbol = None
        
        y_hat, loss = self.model(x, tgt=y, symbol=symbol)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        # Unpack batch
        if len(batch) == 3:
            x, y, symbol = batch
        else:
            x, y = batch
            symbol = None
        
        y_hat, loss = self.model(x, tgt=y, symbol=symbol)
        
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        """For inference without targets."""
        if len(batch) == 3:
            x, _, symbol = batch
        elif len(batch) == 2:
            x, symbol = batch
        else:
            x = batch
            symbol = None
        
        y_hat, _ = self.model(x, tgt=None, symbol=symbol)
        return y_hat
    
    def configure_optimizers(self):
        # Setup optimizer
        if self.hparams.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")
        
        # Setup scheduler if specified
        if self.hparams.scheduler is None:
            return optimizer
        
        scheduler_kwargs = self.hparams.scheduler_kwargs or {}
        
        if self.hparams.scheduler.lower() == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                **scheduler_kwargs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "frequency": 1
                }
            }
        elif self.hparams.scheduler.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                **scheduler_kwargs
            )
        elif self.hparams.scheduler.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                **scheduler_kwargs
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1
            }
        }


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MyCLI(LightningCLI):
    """Custom Lightning CLI with additional setup."""
    
    def add_arguments_to_parser(self, parser):
        # Add seed argument
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        
        # Link arguments
        parser.link_arguments("data.batch_size", "model.init_args.batch_size", apply_on="instantiate")
    
    def before_instantiate_classes(self):
        """Set seed before instantiating classes."""
        set_seed(self.config["seed"])


def cli_main():
    cli = MyCLI(
        LightningCryptoModel,
        CryptoDataModule,
        seed_everything_default=42,
        save_config_callback=None,
        trainer_defaults={
            "max_epochs": 100,
            "accelerator": "auto",
            "devices": "auto",
            "precision": "16-mixed",
            "gradient_clip_val": 1.0,
            "log_every_n_steps": 50,
            "enable_checkpointing": True,
            "enable_progress_bar": True,
            "enable_model_summary": True,
        }
    )


if __name__ == "__main__":
    cli_main()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import yaml
from argparse import ArgumentParser
import os
import numpy as np
import random
from pathlib import Path

from cerebro.dataset import CryptoDataset
from cerebro.loss import RelativeMSELoss
from cerebro.utils import load_config
from cerebro.models.lstm import LSTMModel
from cerebro.models.transformer import TransformerModel


class CryptoLightningModule(pl.LightningModule):
    """PyTorch Lightning wrapper for crypto prediction models."""
    
    def __init__(self, model: nn.Module, config: dict):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(config)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.model.training_step(x, y)
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        loss = self.model.validation_step(batch)
        
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.model.validation_step(batch)
        
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer_config = self.config.get("optimizer", {})
        optimizer_type = optimizer_config.get("type", "adam").lower()
        lr = optimizer_config.get("lr", 1e-3)
        weight_decay = optimizer_config.get("weight_decay", 0.0)
        
        # Initialize optimizer
        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == "sgd":
            momentum = optimizer_config.get("momentum", 0.9)
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Setup learning rate scheduler if configured
        scheduler_config = self.config.get("scheduler", None)
        if scheduler_config is None:
            return optimizer
        
        scheduler_type = scheduler_config.get("type", "none").lower()
        
        if scheduler_type == "none":
            return optimizer
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get("step_size", 10),
                gamma=scheduler_config.get("gamma", 0.1)
            )
        elif scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get("T_max", 50),
                eta_min=scheduler_config.get("eta_min", 0)
            )
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_config.get("factor", 0.5),
                patience=scheduler_config.get("patience", 5)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    pl.seed_everything(seed, workers=True)


def main():
    parser = ArgumentParser(description="Train cryptocurrency price prediction model")
    parser.add_argument(
        "config", 
        type=str, 
        default="configs/vanilla_lstm.yaml", 
        help="Path to config file"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None, 
        help="Path to checkpoint to resume training"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    print("="*70)
    print("CONFIGURATION")
    print("="*70)
    print(yaml.dump(config, default_flow_style=False))
    print("="*70)

    # Set random seed for reproducibility
    set_seed(config.get("seed", 42))

    # Create datasets
    train_dataset = CryptoDataset(**config["data"])
    val_dataset = CryptoDataset(**config["data"], split="val")
    test_dataset = CryptoDataset(**config["data"], split="test")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config["data"].get("num_workers", 0) > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config["data"].get("num_workers", 0) > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config["data"].get("num_workers", 0) > 0
    )

    # Initialize model
    if config["model"]["type"] == "lstm":
        base_model = LSTMModel(**config["model"], loss_fn=RelativeMSELoss())
    elif config["model"]["type"] == "transformer":
        base_model = TransformerModel(**config["model"], loss_fn=RelativeMSELoss())
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")
    
    # Wrap in Lightning module
    model = CryptoLightningModule(base_model, config)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")

    # Setup logger
    save_dir = Path(config["logger"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    wandb_logger = WandbLogger(
        project=config["logger"].get("project", "crypto-prediction"),
        name=config["logger"].get("name", None),
        save_dir=str(save_dir),
        log_model=True
    )
    wandb_logger.experiment.config.update({"trainable_parameters": num_params})
    wandb_logger.watch(model, log="all", log_freq=100)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir / "checkpoints",
        filename='epoch={epoch:02d}-val_loss={val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        patience=config.get("early_stopping_patience", 10),
        mode='min',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    callbacks = [checkpoint_callback, lr_monitor]
    
    if config.get("early_stopping", True):
        callbacks.append(early_stop_callback)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.get("max_epochs", 100),
        accelerator='auto',
        devices='auto',
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=config.get("gradient_clip_val", None),
        accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
        precision=config.get("precision", 32),
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=config.get("log_every_n_steps", 50),
    )
    
    # Train model
    try:
        trainer.fit(
            model, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader,
            ckpt_path=args.resume
        )
        
        # Test evaluation
        print("="*50)
        print("Running final test evaluation...")
        test_results = trainer.test(model, dataloaders=test_loader, ckpt_path='best')
        print("="*50)
        
        # Save final results
        results = {
            'best_val_loss': checkpoint_callback.best_model_score.item(),
            'test_loss': test_results[0]['test/loss'],
            'best_model_path': checkpoint_callback.best_model_path
        }
        
        results_path = save_dir / "results.yaml"
        with open(results_path, 'w') as f:
            yaml.dump(results, f)
        
        print(f"\nResults saved to {results_path}")
        print(f"Best model: {checkpoint_callback.best_model_path}")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    finally:
        wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
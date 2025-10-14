import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from argparse import ArgumentParser
import wandb

import os
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Tuple

from cerebro.dataset import CryptoDataset
from cerebro.loss import RelativeMSELoss
from cerebro.utils import load_config
from cerebro.models.lstm import LSTMModel
from cerebro.models.transformer import TransformerModel

from cerebro.trainer import Trainer


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

def main():
    parser = ArgumentParser(description="Train cryptocurrency price prediction model")
    parser.add_argument(
        "--hparams", 
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
    config = load_config(args.hparams)
    
    print("="*70)
    print("CONFIGURATION")
    print("="*70)
    print(yaml.dump(config, default_flow_style=False))
    print("="*70)

    # Initialize wandb
    wandb.init(
        project=config["logger"].get("project", "crypto-prediction"),
        name=config["logger"].get("name", None),
        config=config,
        dir=config["logger"]["save_dir"],
        resume="allow" if args.resume else None
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = CryptoDataset(**config["data"],
                                   )

    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        CryptoDataset(**config["data"], split="val"),
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available()
    )
    
    # Set random seed for reproducibility
    set_seed(config.get("seed", 42))

    if config["model"]["type"] == "lstm":
        model = LSTMModel(**config["model"],
                          loss_fn=RelativeMSELoss()
                          ).to(device)
    elif config["model"]["type"] == "transformer":
        model = TransformerModel(**config["model"],
                                 loss_fn=RelativeMSELoss()
                                 ).to(device)
        
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    wandb.config.update({"trainable_parameters": num_params})

    # Watch model gradients and parameters
    wandb.watch(model, log="all", log_freq=100)

    # Initialize trainer
    trainer = Trainer(model, config, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train model
    try:
        best_val_loss = trainer.fit(train_loader, val_loader)
        
        # Test evaluation
        trainer.logger.info("="*50)
        trainer.logger.info("Running final test evaluation...")
        
        test_loader = DataLoader(
        CryptoDataset(**config["data"], split="test"),
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available()
    )
        
        test_loss = trainer.evaluate(test_loader, "test")
        trainer.logger.info(f"Test Loss: {test_loss:.6f}")
        trainer.logger.info("="*50)
        
        # Save final results
        results = {
            'best_val_loss': best_val_loss,
            'test_loss': test_loss,
            'total_steps': trainer.global_step
        }
        
        # Log final results to wandb
        wandb.log(results)
        wandb.summary.update(results)
        
        results_path = Path(config["logger"]["save_dir"]) / "results.yaml"
        with open(results_path, 'w') as f:
            yaml.dump(results, f)
        
    except KeyboardInterrupt:
        trainer.logger.info("Training interrupted by user")
        trainer.save_checkpoint("interrupted_checkpoint.pt", trainer.best_val_loss)
    
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
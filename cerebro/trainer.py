import torch
import torch.nn as nn  
from torch.utils.data import DataLoader
from typing import Dict
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np
import random
import logging
import wandb

class Trainer:
    """Enhanced trainer with step-based evaluation and checkpointing."""

    def __init__(self, model: nn.Module, config: Dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Setup logging
        self.setup_logging()
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config["logger"]["save_dir"]) / config["logger"]["name"] / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config["model"]["lr"],
            weight_decay=config["model"].get("weight_decay", 0.0)
        )
        
        scheduler_config = config["model"].get("scheduler", None)
        if scheduler_config:
            if scheduler_config["type"] == "ReduceLROnPlateau":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=scheduler_config.get("factor", 0.5),
                    patience=scheduler_config.get("patience", 5)
                )
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}")
        else:
            self.scheduler = None

        # Early stopping
        self.early_stopping_patience = config["trainer"].get("early_stopping_patience", None)
        self.epochs_without_improvement = 0
        
        # Step-based evaluation interval
        self.eval_every_n_steps = config["trainer"].get("eval_every_n_steps", 100)
        self.log_every_n_steps = config["trainer"].get("log_every_n_steps", 10)

        # Gradient clipping
        self.grad_clip = config["trainer"].get("grad_clip", None)
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config["logger"]["save_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch with step-based evaluation."""
        self.model.train()
        epoch_loss = 0
        epoch_steps = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True)

        for batch_idx, (batch) in enumerate(pbar):
            
            batch = {k: v.to(self.device) for k, v in batch.items()}

            
            # Forward pass
            self.optimizer.zero_grad()
            outputs, loss = self.model(**batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # Update metrics
            batch_loss = loss.item()
            epoch_loss += batch_loss
            epoch_steps += 1
            self.global_step += 1
            
            # Log training metrics
            if self.global_step % self.log_every_n_steps == 0:
                wandb.log({
                    "train/loss_step": batch_loss,
                    "train/loss": epoch_loss / epoch_steps,
                    "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                }, step=self.global_step)
            
            # Step-based evaluation
            if self.global_step % self.eval_every_n_steps == 0:
                val_loss, mae, mse = self.evaluate(self.val_loader, "val")
                self.logger.info(f"Step {self.global_step} - Val Loss: {val_loss:.4E} - MAE: {mae:.4E} - MSE: {mse:.4E}")

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt", val_loss)
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                
                self.model.train()  # Return to training mode
            
            # Update progress bar
            pbar.set_postfix({
                'loss_step': f'{batch_loss:.2E}',
                'loss': f'{epoch_loss/epoch_steps:.2E}',
                'step': self.global_step
            })
        
        return epoch_loss / epoch_steps
    
    def evaluate(self, dataloader: DataLoader, stage: str = "val") -> float:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Eval ({stage})", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs, loss = self.model(**batch)

                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions for analysis
                all_predictions.append(outputs.cpu().flatten())
                all_targets.append(batch["tgt"].cpu().flatten())

        avg_loss = total_loss / num_batches
        
        # Additional metrics
        if stage == "val":
            predictions = torch.cat(all_predictions, dim=0)
            targets = torch.cat(all_targets, dim=0)
            
            # Calculate additional metrics
            mae = torch.mean(torch.abs(predictions - targets)).item()
            mse = torch.mean((predictions - targets) ** 2).item()
            
            # Log to wandb
            wandb.log({
                f"{stage}/loss": avg_loss,
                f"{stage}/mae": mae,
                f"{stage}/mse": mse,
                "global_step": self.global_step
            }, step=self.global_step)
            
            return avg_loss, mae, mse
        
        # Log to wandb for test
        wandb.log({
            f"{stage}/loss": avg_loss,
            "global_step": self.global_step
        }, step=self.global_step)
        
        return avg_loss
    
    def save_checkpoint(self, filename: str, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save checkpoint as wandb artifact
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            description=f"Model checkpoint at step {self.global_step}"
        )
        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch']
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        self.val_loader = val_loader
        max_epochs = self.config["trainer"]["max_epochs"]
        
        self.logger.info("="*50)
        self.logger.info("Starting Training")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Max Epochs: {max_epochs}")
        self.logger.info(f"Eval Every: {self.eval_every_n_steps} steps")
        self.logger.info("="*50)
        
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            val_loss, mae, mse = self.evaluate(val_loader, "val")
            
            # Log epoch metrics
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/mae": mae,
                "val/mse": mse,
                "best_val_loss": self.best_val_loss
            }, step=self.global_step)

            self.logger.info(
                f"Epoch {epoch+1}/{max_epochs} - "
                f"Train Loss: {train_loss:.4E} - "
                f"Val Loss: {val_loss:.4E} - "
                f"Best Val Loss: {self.best_val_loss:.4E}"
            )
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config["trainer"].get("save_every_n_epochs", 10) == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt", val_loss)
            
            # Early stopping
            if self.early_stopping_patience:
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        self.logger.info("Training completed!")
        return self.best_val_loss
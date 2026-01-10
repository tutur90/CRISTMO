import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, Callable, Any
from pathlib import Path
from tqdm import tqdm
import os
import json
import numpy as np
from dataclasses import dataclass, asdict
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class TrainerState:
    """Training state tracker."""
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float('inf')
    best_global_step: int = 0
    train_loss: float = 0.0

    def save(self, path: Path):
        """Save state to JSON."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path):
        """Load state from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class TrainingOutput:
    """Training output container."""
    metrics: Dict[str, float]
    global_step: int
    epoch: int


class StandaloneTrainer:
    """
    Custom standalone trainer that replaces HuggingFace Trainer.
    Implements essential training loop with evaluation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        args: Any,  # TrainingArguments from HF for compatibility
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        compute_metrics: Optional[Callable] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
    ):
        """
        Args:
            model: PyTorch model to train
            args: Training arguments (HF TrainingArguments for compatibility)
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            compute_metrics: Function to compute metrics
            optimizer: Custom optimizer (if None, uses AdamW)
            lr_scheduler: Custom learning rate scheduler
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics

        # Set device
        self.device = torch.device(args.device if hasattr(args, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Initialize optimizer
        if optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 0.0,
            )
        else:
            self.optimizer = optimizer

        # Initialize state
        self.state = TrainerState()

        # Create output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logging setup
        self.log_history = []

        # Initialize WandB if enabled
        if hasattr(args, 'report_to') and 'wandb' in args.report_to:
            try:
                import wandb
                if not wandb.run:
                    wandb.init(
                        project=os.environ.get('WANDB_PROJECT', 'default_project'),
                        config=vars(args),
                        name=args.run_name if hasattr(args, 'run_name') else None,
                    )
            except ImportError:
                logger.warning("wandb not installed, skipping wandb logging")

        # Initialize learning rate scheduler
        if lr_scheduler is None and hasattr(args, 'num_train_epochs'):
            total_steps = len(train_dataset) // args.per_device_train_batch_size * args.num_train_epochs
            self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        else:
            self.lr_scheduler = lr_scheduler

    def _create_dataloader(self, dataset: Any, batch_size: int, shuffle: bool = False) -> DataLoader:
        """Create a DataLoader from dataset."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.args.dataloader_num_workers if hasattr(self.args, 'dataloader_num_workers') else 0,
            pin_memory=True,
        )

    def _prepare_inputs(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Perform a single training step."""
        inputs = self._prepare_inputs(batch)

        # Forward pass
        outputs = self.model(**inputs)

        # Extract loss
        if isinstance(outputs, dict):
            loss = outputs.get('loss')
        elif isinstance(outputs, tuple):
            loss = outputs[0]
        else:
            loss = outputs

        return loss

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        Main training loop.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)

        # Create dataloaders
        train_dataloader = self._create_dataloader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
        )

        # Training setup
        num_epochs = int(self.args.num_train_epochs)
        gradient_accumulation_steps = self.args.gradient_accumulation_steps if hasattr(self.args, 'gradient_accumulation_steps') else 1
        max_grad_norm = self.args.max_grad_norm if hasattr(self.args, 'max_grad_norm') else 1.0

        logging_steps = self.args.logging_steps if hasattr(self.args, 'logging_steps') else 10
        eval_steps = self.args.eval_steps if hasattr(self.args, 'eval_steps') else None
        save_steps = self.args.save_steps if hasattr(self.args, 'save_steps') else None

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num epochs = {num_epochs}")
        logger.info(f"  Batch size = {self.args.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation steps = {gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {len(train_dataloader) * num_epochs // gradient_accumulation_steps}")

        # Training loop
        self.model.train()
        total_loss = 0.0
        start_epoch = self.state.epoch

        for epoch in range(start_epoch, num_epochs):
            self.state.epoch = epoch
            epoch_loss = 0.0

            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for step, batch in enumerate(progress_bar):
                # Forward and backward
                loss = self.training_step(batch)

                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()

                # Update weights
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    # Optimizer step
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    self.state.global_step += 1

                # Track loss
                total_loss += loss.item() * gradient_accumulation_steps
                epoch_loss += loss.item() * gradient_accumulation_steps

                # Update progress bar
                progress_bar.set_postfix({'loss': f"{loss.item() * gradient_accumulation_steps:.4f}"})

                # Logging
                if self.state.global_step % logging_steps == 0:
                    avg_loss = total_loss / logging_steps
                    self.log({
                        'loss': avg_loss,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'step': self.state.global_step,
                    })
                    total_loss = 0.0

                # Evaluation
                if eval_steps is not None and self.state.global_step % eval_steps == 0 and self.eval_dataset is not None:
                    eval_metrics = self.evaluate()
                    self.model.train()

                    # Check if best model
                    metric_key = f"eval_{self.args.metric_for_best_model}" if hasattr(self.args, 'metric_for_best_model') else 'eval_loss'
                    if metric_key in eval_metrics:
                        current_metric = eval_metrics[metric_key]
                        if current_metric < self.state.best_metric:
                            self.state.best_metric = current_metric
                            self.state.best_global_step = self.state.global_step
                            self.save_model(self.output_dir / "best_model")

                # Save checkpoint
                if save_steps is not None and self.state.global_step % save_steps == 0:
                    self.save_model(self.output_dir / f"checkpoint-{self.state.global_step}")

            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1} - Average loss: {avg_epoch_loss:.4f}")

            # Evaluate at end of epoch
            if self.eval_dataset is not None and (eval_steps is None or self.args.evaluation_strategy == 'epoch'):
                eval_metrics = self.evaluate()
                self.model.train()

                # Check if best model
                metric_key = f"eval_{self.args.metric_for_best_model}" if hasattr(self.args, 'metric_for_best_model') else 'eval_loss'
                if metric_key in eval_metrics:
                    current_metric = eval_metrics[metric_key]
                    if current_metric < self.state.best_metric:
                        self.state.best_metric = current_metric
                        self.state.best_global_step = self.state.global_step
                        self.save_model(self.output_dir / "best_model")

        # Return training output
        return TrainingOutput(
            metrics={'train_loss': avg_epoch_loss},
            global_step=self.state.global_step,
            epoch=num_epochs,
        )

    def evaluate(self, eval_dataset: Optional[Any] = None) -> Dict[str, float]:
        """
        Run evaluation loop.

        Args:
            eval_dataset: Dataset to evaluate on (uses self.eval_dataset if None)
        """
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("No evaluation dataset provided")

        dataloader = self._create_dataloader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size if hasattr(self.args, 'per_device_eval_batch_size') else self.args.per_device_train_batch_size,
            shuffle=False,
        )

        logger.info("***** Running evaluation *****")
        logger.info(f"  Num examples = {len(dataset)}")

        self.model.eval()

        all_losses = []
        all_preds = []
        all_labels = []
        all_inputs = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs = self._prepare_inputs(batch)

                # Forward pass
                outputs = self.model(**inputs)

                # Extract components
                if isinstance(outputs, dict):
                    loss = outputs.get('loss')
                    logits = outputs.get('logits')
                elif isinstance(outputs, tuple):
                    loss = outputs[0]
                    logits = outputs[1] if len(outputs) > 1 else None
                else:
                    loss = outputs
                    logits = None

                # Collect for metrics
                if loss is not None:
                    all_losses.append(loss.detach().cpu())
                if logits is not None:
                    all_preds.append(logits.detach().cpu())
                if 'labels' in inputs:
                    all_labels.append(inputs['labels'].detach().cpu())
                if 'sources' in inputs:
                    all_inputs.append(inputs['sources'].detach().cpu())

        # Compute metrics
        metrics = {}

        if all_losses:
            metrics['eval_loss'] = torch.stack(all_losses).mean().item()

        # Custom metrics
        if self.compute_metrics is not None and all_preds and all_labels:
            # Stack all predictions and labels
            predictions = torch.cat(all_preds, dim=0).numpy()
            labels = torch.cat(all_labels, dim=0).numpy()
            inputs_array = torch.cat(all_inputs, dim=0).numpy() if all_inputs else None

            # Create evaluation prediction object
            from collections import namedtuple
            EvalPrediction = namedtuple('EvalPrediction', ['predictions', 'label_ids', 'inputs'])
            eval_pred = EvalPrediction(
                predictions=predictions,
                label_ids=labels,
                inputs=inputs_array,
            )

            custom_metrics = self.compute_metrics(eval_pred)
            metrics.update({f"eval_{k}": v for k, v in custom_metrics.items()})

        self.log(metrics)

        return metrics

    def predict(self, test_dataset: Any):
        """
        Run prediction on test dataset.

        Args:
            test_dataset: Dataset to predict on
        """
        dataloader = self._create_dataloader(
            test_dataset,
            batch_size=self.args.per_device_eval_batch_size if hasattr(self.args, 'per_device_eval_batch_size') else self.args.per_device_train_batch_size,
            shuffle=False,
        )

        logger.info("***** Running prediction *****")
        logger.info(f"  Num examples = {len(test_dataset)}")

        self.model.eval()

        all_preds = []
        all_labels = []
        all_inputs = []
        all_losses = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                inputs = self._prepare_inputs(batch)

                # Forward pass
                outputs = self.model(**inputs)

                # Extract components
                if isinstance(outputs, dict):
                    loss = outputs.get('loss')
                    logits = outputs.get('logits')
                elif isinstance(outputs, tuple):
                    loss = outputs[0]
                    logits = outputs[1] if len(outputs) > 1 else None
                else:
                    loss = outputs
                    logits = None

                if loss is not None:
                    all_losses.append(loss.detach().cpu())
                if logits is not None:
                    all_preds.append(logits.detach().cpu())
                if 'labels' in inputs:
                    all_labels.append(inputs['labels'].detach().cpu())
                if 'sources' in inputs:
                    all_inputs.append(inputs['sources'].detach().cpu())

        # Stack results
        predictions = torch.cat(all_preds, dim=0).numpy() if all_preds else None
        labels = torch.cat(all_labels, dim=0).numpy() if all_labels else None
        inputs_array = torch.cat(all_inputs, dim=0).numpy() if all_inputs else None

        # Compute metrics
        metrics = {}
        if all_losses:
            metrics['test_loss'] = torch.stack(all_losses).mean().item()

        if self.compute_metrics is not None and predictions is not None and labels is not None:
            from collections import namedtuple
            EvalPrediction = namedtuple('EvalPrediction', ['predictions', 'label_ids', 'inputs'])
            eval_pred = EvalPrediction(
                predictions=predictions,
                label_ids=labels,
                inputs=inputs_array,
            )
            custom_metrics = self.compute_metrics(eval_pred)
            metrics.update({f"test_{k}": v for k, v in custom_metrics.items()})

        # Return prediction output
        from collections import namedtuple
        PredictionOutput = namedtuple('PredictionOutput', ['predictions', 'label_ids', 'metrics'])
        return PredictionOutput(
            predictions=predictions,
            label_ids=labels,
            metrics=metrics,
        )

    def save_model(self, output_dir: Optional[Path] = None):
        """Save model checkpoint."""
        save_dir = Path(output_dir) if output_dir else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(self.model.state_dict(), save_dir / "pytorch_model.bin")

        # Save optimizer and scheduler
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
        }, save_dir / "optimizer.pt")

        # Save trainer state
        self.state.save(save_dir / "trainer_state.json")

        logger.info(f"Model saved to {save_dir}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint_dir = Path(checkpoint_path)

        # Load model
        if (checkpoint_dir / "pytorch_model.bin").exists():
            self.model.load_state_dict(torch.load(checkpoint_dir / "pytorch_model.bin"))
            logger.info(f"Model loaded from {checkpoint_path}")

        # Load optimizer and scheduler
        if (checkpoint_dir / "optimizer.pt").exists():
            checkpoint = torch.load(checkpoint_dir / "optimizer.pt")
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.lr_scheduler and checkpoint['lr_scheduler']:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info("Optimizer and scheduler loaded")

        # Load trainer state
        if (checkpoint_dir / "trainer_state.json").exists():
            self.state = TrainerState.load(checkpoint_dir / "trainer_state.json")
            logger.info(f"Training state loaded: epoch={self.state.epoch}, step={self.state.global_step}")

    def log(self, logs: Dict[str, Any]):
        """Log metrics."""
        logs['timestamp'] = time.time()
        self.log_history.append(logs)

        # Log to console
        if hasattr(self.args, 'logging_steps'):
            log_str = " - ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in logs.items()])
            logger.info(log_str)

        # WandB logging if enabled
        if hasattr(self.args, 'report_to') and 'wandb' in self.args.report_to:
            try:
                import wandb
                wandb.log(logs)
            except ImportError:
                pass

    def log_metrics(self, split: str, metrics: Dict[str, float]):
        """Log metrics for a specific split."""
        logger.info(f"***** {split} metrics *****")
        for key, value in metrics.items():
            logger.info(f"  {key} = {value}")

    def save_metrics(self, split: str, metrics: Dict[str, float]):
        """Save metrics to file."""
        metrics_file = self.output_dir / f"{split}_results.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")

    def save_state(self):
        """Save trainer state."""
        self.state.save(self.output_dir / "trainer_state.json")

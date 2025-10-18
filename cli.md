# PyTorch Lightning CLI Training

This is a PyTorch Lightning implementation of the cryptocurrency prediction model with CLI support.

## Installation

```bash
pip install pytorch-lightning wandb
```

## Usage

### Basic Training

Train with the default LSTM configuration:

```bash
python train.py --config config.yaml
```

Train with the Transformer model:

```bash
python train.py --config config_transformer.yaml
```

### Resume Training

Resume from a checkpoint:

```bash
python train.py --config config.yaml --ckpt_path logs/checkpoints/last.ckpt
```

### Override Configuration from CLI

You can override any configuration parameter from the command line:

```bash
# Change learning rate
python train.py --config config.yaml --model.learning_rate 0.0001

# Change batch size
python train.py --config config.yaml --data.batch_size 64

# Change number of epochs
python train.py --config config.yaml --trainer.max_epochs 200

# Multiple overrides
python train.py --config config.yaml \
    --model.learning_rate 0.0005 \
    --data.batch_size 128 \
    --trainer.max_epochs 150 \
    --seed 123
```

### Test Only

Run testing on a trained model:

```bash
python train.py --config config.yaml \
    --ckpt_path logs/checkpoints/best.ckpt \
    --trainer.limit_train_batches 0 \
    --trainer.limit_val_batches 0
```

### Print Configuration

See the full configuration without training:

```bash
python train.py --config config.yaml --print_config
```

### Generate Configuration Template

Generate a configuration file with all available options:

```bash
python train.py --print_config > my_config.yaml
```

## Configuration Structure

The configuration file has three main sections:

### 1. Trainer Configuration
Controls PyTorch Lightning Trainer settings:
- `max_epochs`: Number of training epochs
- `accelerator`: Hardware accelerator (auto, gpu, cpu, tpu)
- `devices`: Number of devices to use
- `precision`: Training precision (32, 16-mixed, bf16-mixed)
- `callbacks`: List of callbacks (checkpointing, early stopping, etc.)
- `logger`: Logging configuration (WandB, TensorBoard, etc.)

### 2. Model Configuration
Model architecture and training hyperparameters:
- `model_type`: Type of model (lstm, transformer)
- `learning_rate`: Learning rate for optimizer
- `optimizer`: Optimizer type (adam, adamw, sgd)
- `scheduler`: Learning rate scheduler
- Model-specific parameters (hidden_size, num_layers, etc.)

### 3. Data Configuration
Dataset and DataLoader settings:
- `batch_size`: Batch size for training
- `num_workers`: Number of data loading workers
- Dataset-specific parameters (data_dir, sequence_length, etc.)

## Advanced Features

### Custom Callbacks

Add custom callbacks in the config:

```yaml
trainer:
  callbacks:
    - class_path: pytorch_lightning.callbacks.ModelCheckpoint
      init_args:
        monitor: val/loss
        save_top_k: 5
```

### Multiple GPUs

Train on multiple GPUs:

```bash
python train.py --config config.yaml --trainer.devices 4 --trainer.strategy ddp
```

### Mixed Precision Training

Use automatic mixed precision:

```yaml
trainer:
  precision: 16-mixed  # or bf16-mixed for better stability
```

### Gradient Accumulation

Simulate larger batch sizes:

```yaml
trainer:
  accumulate_grad_batches: 4  # Effective batch size = batch_size * 4
```

## Monitoring

### WandB Integration

The training automatically logs to Weights & Biases. Configure in the YAML:

```yaml
trainer:
  logger:
    class_path: pytorch_lightning.loggers.WandbLogger
    init_args:
      project: my-project
      name: my-experiment
      tags: [lstm, crypto]
```

### TensorBoard (Alternative)

Use TensorBoard instead of WandB:

```yaml
trainer:
  logger:
    class_path: pytorch_lightning.loggers.TensorBoardLogger
    init_args:
      save_dir: logs
      name: my_experiment
```

View with:
```bash
tensorboard --logdir logs
```

## Key Benefits of Lightning CLI

1. **Reproducibility**: All hyperparameters are in version-controlled YAML files
2. **Flexibility**: Override any parameter from command line
3. **Best Practices**: Automatic mixed precision, gradient clipping, checkpointing
4. **Scalability**: Easy multi-GPU training without code changes
5. **Monitoring**: Built-in logging to WandB, TensorBoard, etc.
6. **Debugging**: Rich progress bars, model summaries, and profiling tools

## Tips

- Start with a small subset using `--trainer.limit_train_batches 10`
- Use `--trainer.fast_dev_run true` to test the full training loop quickly
- Enable profiling with `--trainer.profiler simple`
- Use `--trainer.overfit_batches 10` to debug model on a small set
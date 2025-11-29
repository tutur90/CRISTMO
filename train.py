import logging
import os
import sys

import transformers
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import Trainer, TrainingArguments, HfArgumentParser, set_seed
from safetensors.torch import load_file as safe_open

from cerebro.args import DataTrainingArguments, ModelArguments
from cerebro.loss import RelativeMSELoss, BasicInvLoss, InvLoss
from cerebro.models import model
from cerebro.models.lstm import LSTMModel
from cerebro.models.transformer import TransformerModel
from cerebro.models.tcn import TCNModel
from cerebro.models.tcn2 import TCN2Model
from cerebro.dataset import CryptoDataset

logger = logging.getLogger(__name__)

class CustomTrainer(Trainer):
    """
    Custom Trainer to handle our specific model outputs.
    """
    

def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif sys.argv[-1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))

        
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    os.environ["WANDB_PROJECT"]=data_args.wandb_project_name

    if data_args.total_batch_size is not None:
        training_args.gradient_accumulation_steps = data_args.total_batch_size // (training_args.per_device_train_batch_size * int(os.environ.get("WORLD_SIZE", 1))) + 1
        logger.info(f"Setting gradient_accumulation_steps to {training_args.gradient_accumulation_steps} to match total_batch_size {data_args.total_batch_size}")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load datasets
    
    if training_args.do_train:

        train_dataset = CryptoDataset(**data_args.__dict__, **model_args.__dict__, split="train")

    if training_args.do_eval:
        eval_dataset = CryptoDataset(**data_args.__dict__, **model_args.__dict__, split="val")

    if model_args.loss_function["type"] == "relative_mse":

        loss_fn = RelativeMSELoss()

    elif model_args.loss_function["type"] == "basic_inv":
        loss_fn = BasicInvLoss()
    elif model_args.loss_function["type"] == "inv":

        loss_fn = InvLoss(**model_args.loss_function)
        
    else:
        raise ValueError(f"Unknown loss function: {model_args.loss_function}")

    if model_args.type == "lstm":
        model = LSTMModel(**model_args.__dict__, loss_fn=loss_fn)
    elif model_args.type == "transformer":
        model = TransformerModel(**model_args.__dict__, loss_fn=RelativeMSELoss())
    elif model_args.type == "tcn":
        model = TCNModel(**model_args.__dict__, loss_fn=loss_fn)
    elif model_args.type == "tcn2":
        model = TCN2Model(**model_args.__dict__, loss_fn=loss_fn)
    elif model_args.type == "static":
        from cerebro.models.static import StaticModel
        model = StaticModel(**model_args.__dict__, loss_fn=loss_fn)
    else:
        raise ValueError(f"Unknown model type: {model_args.type}")
    

    if model_args.pretrained_model is not None:
        logger.info(f"Loading pretrained model from {model_args.pretrained_model}")
        state_dict = safe_open(model_args.pretrained_model)

        del state_dict["fc.weight"]
        del state_dict["fc.bias"]

        model.load_state_dict(state_dict, strict=False)
        logger.info("Pretrained model loaded successfully.")
        
        
    def compute_metrics(p):
        
        import numpy as np
        import matplotlib.pyplot as plt


        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(p.losses)), p.losses, label="Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.savefig(os.path.join(training_args.output_dir, "training_loss.png"))

        plt.close()

        avg_repartition = np.mean(p.predictions, axis=0).squeeze()
        
        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(avg_repartition)), avg_repartition)
        plt.xlabel("Grid Index")
        plt.ylabel("Average Repartition")
        plt.title("Average Repartition over Grid")
        plt.savefig(os.path.join(training_args.output_dir, "avg_repartition.png"))
        plt.close()
        

        return {
            "loss": p.losses.mean(),
        }

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
    )
    
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
            
        try:
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
        except KeyboardInterrupt:
            logger.info("Training interrupted. Saving model...")
            trainer.save_model()  # Saves the tokenizer too for easy upload
            sys.exit(0)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    
    if training_args.do_eval:

        logger.info("*** Evaluate ***")
        
        
        results = trainer.evaluate()


        trainer.log_metrics("eval", results)
        trainer.save_metrics("eval", results)

    if training_args.do_predict:
        

        logger.info("*** Predict ***")

        test_dataset = CryptoDataset(**data_args.__dict__, split="test")


        results = trainer.predict(test_dataset,  metric_key_prefix="predict"

        )


        trainer.log_metrics("predict", results)
        trainer.save_metrics("predict", results)


    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
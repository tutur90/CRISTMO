from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    
    type: str = field(
        default="lstm", metadata={"help": "The type of model to use."}
    )
    
    hidden_dim: int = field(
        default=128, metadata={"help": "The hidden dimension of the model."}
    )   
    
    src_length: int = field(
        default=24, metadata={"help": "The input sequence length."}
    )
    tgt_length: int = field(
        default=1, metadata={"help": "The target sequence length."}
    )
    seg_length: int = field(
        default=60, metadata={"help": "The segment length for the model."}
    )
    dropout: float = field(     
        default=0.1, metadata={"help": "The dropout rate for the model."}
    )       
    
    input_features: Optional[str] = field(
        default="open,high,low,close", metadata={"help": "Comma-separated list of input features."}
    )   
    output_dim: int = field(
        default=3, metadata={"help": "The number of output features."}
    )   
    
    num_layers: int = field(
        default=2, metadata={"help": "The number of layers in the model."}
    )    
    bidirectional: bool = field(
        default=False, metadata={"help": "Whether to use bidirectional LSTM."}
    )
    conv_kernel: int = field(
        default=5, metadata={"help": "The convolution kernel size."}
    )
    pool_kernel: Optional[int] = field(
        default=1, metadata={"help": "The pooling kernel size."}
    )
    loss_function: str = field(
        default="basic_inv", metadata={"help": "The loss function to use."} 
    )
    
    pretrained_model: Optional[str] = field(
        default=None, metadata={"help": "Path to a pretrained model to load."}
    )
    
    kernels: Optional[str] = field(
        default="2,2,2,2", metadata={"help": "Comma-separated list of kernel sizes for TCN layers."}
    )
    dilations: Optional[str] = field(
        default="1,2,4,8", metadata={"help": "Comma-separated list of dilation factors for TCN layers."}
    )
    
    
    epsilon: float = field(
        default=1e-6, metadata={"help": "Epsilon value for numerical stability in loss functions."}
    )
    
    only_last: bool = field(
        default=True, metadata={"help": "Whether to use only the last output for prediction."}
    )
    
    n_head : int = field(
        default=4, metadata={"help": "Number of attention heads for Transformer models."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    
    total_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Total batch size for training. If provided, will override per_device_train_batch_size and "
                "gradient_accumulation_steps to match this total batch size."
            )
        },
    )
    
    tgt_mode: str = field(
        default="ohlc", metadata={"help": "Target mode for prediction (e.g., 'close', 'open')."}
    )

    symbols: Optional[str] = field(
        default=None, metadata={"help": "List of symbols to include in the dataset."}
    )
    tgt_symbol: Optional[str] = field(
        default=None, metadata={"help": "Target symbol for prediction."}
    )
    start_date: Optional[str] = field(
        default=None, metadata={"help": "Start date for filtering the dataset (YYYY-MM-DD)."}
    )


    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    
    wandb_project_name: str = field(
        default="CRISTMO", metadata={"help": "Weights & Biases project name for logging."}
    )

    def __post_init__(self):
        if self.data_path is None:
            raise ValueError("You must specify a data_path for the dataset.")
        
        if self.symbols is not None:
            self.symbols = [s.strip() for s in self.symbols.split(",")]

        self.tgt_symbol = self.tgt_symbol.strip() if self.tgt_symbol is not None else None

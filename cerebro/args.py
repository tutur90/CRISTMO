from dataclasses import dataclass, field
from typing import Optional

from sympy import N



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
    
    input_dim: int = field(
        default=4, metadata={"help": "The number of input features."}
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
    
    # src_length: Optional[int] = field(
    #     default=24, metadata={"help": "The input sequence length."}
    # )
    # tgt_length: Optional[int] = field(
    #     default=1, metadata={"help": "The target sequence length."}
    # )
    # seg_length: Optional[int] = field(
    #     default=60, metadata={"help": "The segment length for the model."}
    # )


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

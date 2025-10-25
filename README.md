# CRISTMO


CRISTMO (CRypto InveSTment MOdel) is a Python package designed to facilitate the analysis and modeling of cryptocurrency investments. It provides tools for data retrieval, preprocessing, feature engineering, and model evaluation specifically tailored for the unique characteristics of the cryptocurrency market.


## Features
- **Data Retrieval**: Fetch historical cryptocurrency data from various sources.
- **Preprocessing**: Clean and preprocess data for analysis.
- **Feature Engineering**: Create features relevant to cryptocurrency investments.
- **Model Evaluation**: Evaluate investment models using various metrics.
- **Visualization**: Visualize data and model performance.


## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
    cd CRISTMO
    ```
2. Install the required dependencies, download data and process it:

```bash
./startup.sh
```
3. Run the training script:
```bash
accelerate launch --mixed-precision=fp16 --dynamo-backend=no --num_processes=1 --num_machines=1 train.py --config configs/training/lstm.yaml
```

## Data

Quickstart to prepare the data:

```
./scripts/data_pipeline.sh
```

### Download Data

To download historical cryptocurrency data from binance, use the following script:

```bash
./scripts/download_data.sh

```

You can modify the `SYMBOLS` and `INTERVALS` variables in the script to specify which cryptocurrencies and time intervals you want to download.

Choosse between `spot` and `futures` data by changing the `type` variable in the script.

### Prepare Data

To prepare the data for analysis, run the following script:

```bash
python data/prepare_data.py --input data/futures/raw --output data/futures/dataset
```

By default, the script applies max scaling, log scaling, and bias removal. You can disable any of these preprocessing steps by setting the corresponding flags to `False`:  

```bash
python data/prepare_data.py --input data/futures/raw --output data/futures/dataset --max_scaling False --log_scaling False --bias_removal False
``` 


## Modelisation

1. Pretrain the model on high low close price: Not big improvement yet

2. Train/Finetune the model on investment return

### Notes

Log scale -> Normalisation on last price (dividing by last price and subtracting by the last price or (the std)) -> Augmentation (adding noise, scale wrapping) 
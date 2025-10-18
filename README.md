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
python train.py configs/vanilla_lstm.yaml
```

## Modelisation

1. Pretrain the model on high low close price

2. Finetune the model on investment return

### Notes

Log scale -> Normalisation on last price (dividing by last price and subtracting by the last price or (the std)) -> Augmentation (adding noise, scale wrapping) 
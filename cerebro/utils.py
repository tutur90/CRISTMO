import yaml
import numpy as np

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
    


def rmspe_loss(y_pred, y_true, epsilon=1e-8, use_close=True, use_last=True, **kwargs):
    """
    Root Mean Squared Percentage Error Loss
    Works with log-transformed data (applies exp to get original scale)
    """
    if use_close:
        y_pred = y_pred[:, :, -1]  # (B,) - last timestep only
        y_true = y_true[:, :, -1]  # (B,)
        
    if use_last:
        y_pred = y_pred[:, -1]  # (B,)
        y_true = y_true[:, -1]  # (B,)
        
    y_pred, y_true = np.exp(y_pred), np.exp(y_true)
    
    return np.sqrt(np.mean(((y_true - y_pred) / (y_true + epsilon)) ** 2)) * 100
    

def mape_loss(y_pred, y_true, epsilon=1e-8, use_close=True, use_last=True, **kwargs):
    """
    Mean Absolute Percentage Error Loss
    Works with log-transformed data (applies exp to get original scale)
    """
    if use_close:
        y_pred = y_pred[:, :, -1]  # (B,) - last timestep only
        y_true = y_true[:, :, -1]  # (B,)
        
    if use_last:
        y_pred = y_pred[:, -1]  # (B,)
        y_true = y_true[:, -1]  # (B,)
        
    y_pred, y_true = np.exp(y_pred), np.exp(y_true)
    
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def rmse_loss(y_pred, y_true, use_close=True, use_last=True, **kwargs):
    """
    Root Mean Squared Percentage Error Loss
    Works with log-transformed data (applies exp to get original scale)
    """
    if use_close:
        y_pred = y_pred[:, :, -1]  # (B,) - last timestep only
        y_true = y_true[:, :, -1]  # (B,)
        
    if use_last:
        y_pred = y_pred[:, -1]  # (B,)
        y_true = y_true[:, -1]  # (B,)
        
    y_pred, y_true = np.exp(y_pred), np.exp(y_true)
    
    return np.sqrt(np.mean(((y_true - y_pred)) ** 2)) 
    

def mae_loss(y_pred, y_true, use_close=True, use_last=True, **kwargs):
    """
    Mean Absolute Percentage Error Loss
    Works with log-transformed data (applies exp to get original scale)
    """
    if use_close:
        y_pred = y_pred[:, :, -1]  # (B,) - last timestep only
        y_true = y_true[:, :, -1]  # (B,)
        
    if use_last:
        y_pred = y_pred[:, -1]  # (B,)
        y_true = y_true[:, -1]  # (B,)
        
    y_pred, y_true = np.exp(y_pred), np.exp(y_true)
    
    return np.mean(np.abs((y_true - y_pred) ))
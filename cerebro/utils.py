import yaml
import numpy as np

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class BaseMetric:
    """Base class for loss functions."""
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true, **kwargs):
        return self.forward(y_pred, y_true, **kwargs)

    def forward(self, y_pred, y_true, **kwargs):
        raise NotImplementedError("Forward method not implemented.")
    
class BaseForecast(BaseMetric):
    """Base class for forecasting loss functions."""
    def __init__(self, epsilon=1e-8, use_close=True, use_last=True, **kwargs):
        super().__init__()
        
        self.epsilon = epsilon
        self.use_close = use_close
        self.use_last = use_last   
        
    def forward(self, y_pred, y_true, **kwargs):
        
        if self.use_close:
            y_pred = y_pred[:, :, -1]  # (B,) - last timestep only
        y_true = y_true[:, :, -1]  # (B,)
        
        if self.use_last:
            y_pred = y_pred[:, -1]  # (B,)
            y_true = y_true[:, -1]  # (B,)
        
        y_pred, y_true = np.exp(y_pred), np.exp(y_true)
        
        raise self._forward(y_pred, y_true)
    
    def _forward(self, y_pred, y_true):
        raise NotImplementedError("Forward method not implemented in subclass.")
    
class MultiForecastMetric(BaseForecast):
    """Base class for multi-step forecasting loss functions."""
    def __init__(self, epsilon=1e-8, use_close=True, **kwargs):
        super().__init__(epsilon=epsilon, use_close=use_close, use_last=False, **kwargs)
    
    def _forward(self, y_pred, y_true):
        results = {}
        results["rmse"] = rmse_loss(y_pred, y_true)
        results["mae"] = mae_loss(y_pred, y_true)
        results["mape"] = mape_loss(y_pred, y_true, epsilon=self.epsilon)
        results["rmspe"] = rmspe_loss(y_pred, y_true, epsilon=self.epsilon)
        return results
        

def rmspe_loss(y_pred, y_true, epsilon=1e-8):
    """
    Root Mean Squared Percentage Error Loss
    Works with log-transformed data (applies exp to get original scale)
    """
    return np.sqrt(np.mean(((y_true - y_pred) / (y_true + epsilon)) ** 2)) * 100
    

def mape_loss(y_pred, y_true, epsilon=1e-8):
    """
    Mean Absolute Percentage Error Loss
    """
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def rmse_loss(y_pred, y_true, **kwargs):
    """
    Root Mean Squared Percentage Error Loss
    Works with log-transformed data (applies exp to get original scale)
    """
    
    return np.sqrt(np.mean(((y_true - y_pred)) ** 2)) 
    

def mae_loss(y_pred, y_true, **kwargs):
    """
    Mean Absolute Percentage Error Loss
    Works with log-transformed data (applies exp to get original scale)
    """
    
    return np.mean(np.abs((y_true - y_pred) ))


class InvMetric(BaseMetric):
    """
    Inverse Transformation Metric for PnL calculation
    """
    def __init__(self, epsilon=1e-8, leverage=1.0, fee=0.01, **kwargs):
        super().__init__()
        self.epsilon = epsilon
        self.leverage = leverage
        self.fee = fee/100
        
    def __call__(self, inputs, inv, target, **kwargs):
        return self.forward(inputs, inv, target, **kwargs)
        
    def forward(self, inputs, inv, target,  num_items_in_batch: int= None):
        
        """
        Args:
            y_pred: predicted values (tensor)
            y_true: actual values (tensor)

        Returns:
            PnL metric
        """
        
        B, T, C = target.shape
        
        
        inv = inv[0].reshape(B, T) * self.leverage  # (B, T)
        
        last = np.exp(inputs["sources"][:, -1:, -1])  # (B, 1)
        
        ret = np.exp(target.reshape(B, T)) / last.reshape(B, 1)  # (B, T)
        
        pnl = inv * (ret - 1) + 1 - self.fee * np.abs(inv)  # (B, T)
        
        log_pnl = np.log(np.clip(pnl, a_min=1e-8, a_max=None)) 
        
        
        return {'pnl': pnl, 'log_pnl': log_pnl, 'mean_log_pnl': log_pnl.mean() * 24 * 364}
        
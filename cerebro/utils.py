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
    def __init__(self, epsilon=1e-8, leverage=1.0, fee=0.01, output_dim=None, **kwargs):
        super().__init__()
        self.epsilon = epsilon
        self.leverage = leverage
        self.fee = fee/100
        self.cumsum = False
        
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
        
        target = np.exp(target).squeeze()
        inv = inv[0].squeeze()
        
        B, T = target.shape
        
        w = np.zeros((B + T, ))  # (B + T,)

        inv = np.cumsum(inv[:,  ::-1], axis=-1)[:, ::-1]  # (B + C,)
        
        
        # for i in range(0, B):
            
        #     w[i:i + T] += inv[i , 0] / min(T, i + 1)  # distribute weight over available timesteps
        
        R =  target[1:, 0] / target[:-1, 0]  # (B - 1,)
        
        
        fees = self.fee * np.abs(np.diff(w[:-T]))
        
        pnl = w[1:-T] * (R - 1) + 1 - fees
        
        log_pnl = np.log(np.clip(pnl, a_min=1e-12, a_max=None))
        
        return {'pnl': pnl, 'log_pnl': log_pnl, 'weights': w[1:-T]}
        
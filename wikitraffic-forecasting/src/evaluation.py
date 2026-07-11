"""
Evaluation module for time series forecasting models.
Implements SMAPE metric and plotting functions for model comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd


class TimeSeriesEvaluator:
    """Evaluation class for time series forecasting models."""
    
    def __init__(self):
        self.results = {}
        
    def smape(self, y_true, y_pred):
        """
        Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            
        Returns:
            float: SMAPE score
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Handle division by zero
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        # Avoid division by zero
        mask = denominator != 0
        if not np.any(mask):
            return 0.0
            
        smape_values = np.abs(y_true - y_pred) / denominator
        smape_values = smape_values[mask]
        
        return np.mean(smape_values) * 100
    
    def calculate_metrics(self, y_true, y_pred, model_name="Model"):
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            model_name (str): Name of the model
            
        Returns:
            dict: Dictionary containing all metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Convert back from log scale if needed
        y_true_exp = np.expm1(y_true)
        y_pred_exp = np.expm1(y_pred)
        
        metrics = {
            'model': model_name,
            'smape': self.smape(y_true_exp, y_pred_exp),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mae_exp': mean_absolute_error(y_true_exp, y_pred_exp),
            'mse_exp': mean_squared_error(y_true_exp, y_pred_exp),
            'rmse_exp': np.sqrt(mean_squared_error(y_true_exp, y_pred_exp))
        }
        
        self.results[model_name] = metrics
        return metrics


if __name__ == "__main__":
    # Example usage
    evaluator = TimeSeriesEvaluator()
    
    # Generate sample data for demonstration
    np.random.seed(42)
    y_true = np.random.lognormal(5, 2, 1000)
    y_pred = y_true + np.random.normal(0, 0.5, 1000)
    
    # Convert to log scale
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(y_pred)
    
    # Evaluate
    metrics = evaluator.calculate_metrics(y_true_log, y_pred_log, "Sample Model")
    
    print("Sample Model Evaluation Results:")
    print(f"SMAPE: {metrics['smape']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")

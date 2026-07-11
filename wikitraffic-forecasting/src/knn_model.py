"""
K-Nearest Neighbors (KNN) model for time series forecasting and anomaly detection.
Implements baseline regression model with Canberra distance metric.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')


class KNNTimeSeriesModel:
    """KNN-based time series forecasting model with anomaly detection."""
    
    def __init__(self, k=10, distance_metric='canberra', weights='distance'):
        """
        Initialize KNN model.
        
        Args:
            k (int): Number of neighbors
            distance_metric (str): Distance metric for KNN
            weights (str): Weight function for prediction
        """
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_importance_ = None
        
    def fit(self, X_train, y_train, optimize_hyperparameters=True):
        """
        Train the KNN model.
        
        Args:
            X_train (array-like): Training features
            y_train (array-like): Training targets
            optimize_hyperparameters (bool): Whether to optimize hyperparameters
        """
        print("Training KNN model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if optimize_hyperparameters:
            # Hyperparameter optimization
            param_grid = {
                'n_neighbors': [5, 10, 15, 20, 25],
                'weights': ['uniform', 'distance'],
                'metric': ['canberra', 'euclidean', 'manhattan']
            }
            
            knn = KNeighborsRegressor()
            grid_search = GridSearchCV(
                knn, param_grid, cv=5, scoring='neg_mean_squared_error', 
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            self.k = grid_search.best_params_['n_neighbors']
            self.weights = grid_search.best_params_['weights']
            self.distance_metric = grid_search.best_params_['metric']
            
            print(f"Best parameters: k={self.k}, weights={self.weights}, metric={self.distance_metric}")
        
        # Train final model with best parameters
        self.model = KNeighborsRegressor(
            n_neighbors=self.k,
            metric=self.distance_metric,
            weights=self.weights
        )
        
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        print("KNN model training completed!")
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (array-like): Features for prediction
            
        Returns:
            array: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def detect_anomalies(self, X, y_true=None, threshold_multiplier=2.0):
        """
        Detect anomalies using KNN distance-based approach.
        
        Args:
            X (array-like): Features
            y_true (array-like): True values (optional)
            threshold_multiplier (float): Multiplier for anomaly threshold
            
        Returns:
            dict: Anomaly detection results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before anomaly detection")
        
        X_scaled = self.scaler.transform(X)
        
        # Get distances to k nearest neighbors
        distances, indices = self.model.kneighbors(X_scaled)
        
        # Calculate mean distance to neighbors for each point
        mean_distances = np.mean(distances, axis=1)
        
        # Set anomaly threshold
        threshold = np.mean(mean_distances) + threshold_multiplier * np.std(mean_distances)
        
        # Identify anomalies
        anomalies = mean_distances > threshold
        anomaly_indices = np.where(anomalies)[0]
        
        # If true values provided, calculate prediction errors for anomalies
        prediction_errors = None
        if y_true is not None:
            predictions = self.predict(X)
            prediction_errors = np.abs(y_true - predictions)
            anomaly_errors = prediction_errors[anomalies] if np.any(anomalies) else np.array([])
        else:
            anomaly_errors = mean_distances[anomalies] if np.any(anomalies) else np.array([])
        
        return {
            'anomaly_indices': anomaly_indices,
            'num_anomalies': len(anomaly_indices),
            'anomaly_rate': len(anomaly_indices) / len(X) * 100,
            'mean_distances': mean_distances,
            'threshold': threshold,
            'anomaly_errors': anomaly_errors,
            'anomalies': anomalies
        }


def train_knn_model(X_train, y_train, X_test, y_test, feature_names=None):
    """
    Complete training and evaluation pipeline for KNN model.
    
    Args:
        X_train (array-like): Training features
        y_train (array-like): Training targets
        X_test (array-like): Test features
        y_test (array-like): Test targets
        feature_names (list): Names of features
        
    Returns:
        tuple: (trained_model, predictions, metrics)
    """
    # Initialize and train model
    knn_model = KNNTimeSeriesModel(k=10, distance_metric='canberra')
    knn_model.fit(X_train, y_train, optimize_hyperparameters=False)
    
    # Make predictions
    y_pred = knn_model.predict(X_test)
    
    # Calculate metrics
    from evaluation import TimeSeriesEvaluator
    evaluator = TimeSeriesEvaluator()
    metrics = evaluator.calculate_metrics(y_test, y_pred, "KNN")
    
    # Detect anomalies
    anomaly_info = knn_model.detect_anomalies(X_test, y_test)
    
    print(f"\nKNN Model Results:")
    print(f"SMAPE: {metrics['smape']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"Anomalies detected: {anomaly_info['num_anomalies']} ({anomaly_info['anomaly_rate']:.2f}%)")
    
    return knn_model, y_pred, metrics


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import WikiTrafficPreprocessor
    
    # Create sample data
    preprocessor = WikiTrafficPreprocessor()
    result = preprocessor.preprocess_pipeline(sample_size=500)
    
    # Prepare data
    X_train, X_test, y_train, y_test = preprocessor.split_data(result['data'])
    
    # Train KNN model
    knn_model, predictions, metrics = train_knn_model(
        X_train, y_train, X_test, y_test, result['feature_columns']
    )
    
    print("KNN model training and evaluation completed!")

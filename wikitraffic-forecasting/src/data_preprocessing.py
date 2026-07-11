"""
Data preprocessing module for Wikipedia web traffic time series forecasting.
Handles loading, cleaning, feature engineering, and data splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class WikiTrafficPreprocessor:
    """Preprocesses Wikipedia web traffic data for time series forecasting."""
    
    def __init__(self):
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.feature_columns = []
        self.is_fitted = False
        
    def load_data(self, file_path=None, sample_size=10000):
        """
        Load Wikipedia web traffic data.
        
        Args:
            file_path (str): Path to the dataset CSV file
            sample_size (int): Number of time series to sample for demo purposes
            
        Returns:
            pd.DataFrame: Loaded data
        """
        # Check for Kaggle dataset files in data directory
        if file_path is None:
            import os
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            
            # Check for common Kaggle dataset filenames
            possible_files = ['validation_score.csv', 'train_1.csv', 'train_2.csv', 'train.csv']
            
            for filename in possible_files:
                file_path = os.path.join(data_dir, filename)
                if os.path.exists(file_path):
                    print(f"Found dataset file: {filename}")
                    break
                else:
                    file_path = None
        
        if file_path is None:
            # Create synthetic data for demonstration if no file provided
            print("No data file provided. Creating synthetic Wikipedia traffic data...")
            return self._create_synthetic_data(sample_size)
        
        try:
            print(f"Loading dataset from: {file_path}")
            data = pd.read_csv(file_path)
            print(f"✅ Real dataset loaded successfully")
            print(f"Dataset shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            
            # Check if this is the validation_score.csv format
            if 'validation_score.csv' in file_path:
                print("Detected validation_score.csv format - converting to time series format...")
                data = self._convert_validation_score_to_time_series(data, sample_size)
            
            return data
        except FileNotFoundError:
            print(f"File {file_path} not found. Creating synthetic data...")
            return self._create_synthetic_data(sample_size)
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            print("Falling back to synthetic data...")
            return self._create_synthetic_data(sample_size)
    
    def _convert_validation_score_to_time_series(self, data, sample_size=10000):
        """
        Convert validation_score.csv format to time series format for forecasting.
        
        Args:
            data (pd.DataFrame): Validation score data
            sample_size (int): Number of time series to create
            
        Returns:
            pd.DataFrame: Converted time series data
        """
        print("Converting validation score data to time series format...")
        
        # Sample the data if it's too large
        if len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} rows from dataset")
        
        # Create time series data from the validation scores
        time_series_data = []
        
        # Create date range for the time series
        dates = pd.date_range(start='2015-07-01', end='2017-09-10', freq='D')
        
        # Access types for Wikipedia traffic
        access_types = ['desktop', 'mobile', 'spider']
        
        for idx, row in data.iterrows():
            # Create a unique page identifier
            page_name = f"Page_{idx}"
            
            # Select an access type
            access_type = np.random.choice(access_types)
            
            # Use median values as base traffic patterns
            base_traffic = row.get('median7', 0.4)  # Use median7 as base
            
            # Create traffic pattern using the validation scores
            traffic_values = []
            
            # Generate traffic for each date using different median values as patterns
            for i, date in enumerate(dates):
                # Use different median columns for different time periods
                if i < len(dates) // 7:  # First 1/7 of dates
                    pattern_value = row.get('median7', 0.4)
                elif i < 2 * len(dates) // 7:  # Second 1/7
                    pattern_value = row.get('median14', 0.4)
                elif i < 3 * len(dates) // 7:  # Third 1/7
                    pattern_value = row.get('median21', 0.4)
                elif i < 4 * len(dates) // 7:  # Fourth 1/7
                    pattern_value = row.get('median28', 0.4)
                elif i < 5 * len(dates) // 7:  # Fifth 1/7
                    pattern_value = row.get('median35', 0.4)
                elif i < 6 * len(dates) // 7:  # Sixth 1/7
                    pattern_value = row.get('median42', 0.4)
                else:  # Last 1/7
                    pattern_value = row.get('median49', 0.4)
                
                # Convert validation score to traffic count
                # Scale the score to a reasonable traffic range
                traffic_count = max(1, int(pattern_value * 1000 + np.random.poisson(50)))
                
                # Add seasonal variation
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 365.25)
                traffic_count = int(traffic_count * seasonal_factor)
                
                # Add access type multiplier
                access_multiplier = {'desktop': 1.0, 'mobile': 0.7, 'spider': 0.1}[access_type]
                traffic_count = int(traffic_count * access_multiplier)
                
                traffic_values.append(traffic_count)
            
            # Add some missing values (5% of the data)
            missing_indices = np.random.choice(len(traffic_values), 
                                             size=int(0.05 * len(traffic_values)), 
                                             replace=False)
            
            for i, date in enumerate(dates):
                traffic_value = traffic_values[i] if i not in missing_indices else np.nan
                
                time_series_data.append({
                    'Page': page_name,
                    'Date': date,
                    'Access': access_type,
                    'Traffic': traffic_value
                })
        
        df = pd.DataFrame(time_series_data)
        print(f"Converted validation score data to time series format: {df.shape}")
        return df
    
    def _create_synthetic_data(self, sample_size=10000):
        """Create synthetic Wikipedia traffic data for demonstration."""
        np.random.seed(42)
        
        # Generate synthetic page names
        page_names = [f"Page_{i}" for i in range(sample_size)]
        
        # Generate synthetic access types
        access_types = ['desktop', 'mobile', 'spider']
        
        # Generate synthetic data
        data = []
        dates = pd.date_range(start='2015-07-01', end='2017-09-10', freq='D')
        
        for i, page in enumerate(page_names):
            access_type = np.random.choice(access_types)
            
            # Generate realistic traffic patterns
            base_traffic = np.random.lognormal(mean=5, sigma=2)
            
            # Add seasonal patterns
            seasonal = 1 + 0.3 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
            
            # Add weekly patterns
            weekly = 1 + 0.2 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
            
            # Add access type effects
            access_multiplier = {'desktop': 1.0, 'mobile': 0.7, 'spider': 0.1}[access_type]
            
            # Generate traffic with noise
            traffic = base_traffic * seasonal * weekly * access_multiplier
            traffic += np.random.poisson(lam=5, size=len(traffic))
            
            # Add some zeros and missing values
            missing_indices = np.random.choice(len(traffic), size=int(0.05 * len(traffic)), replace=False)
            traffic[missing_indices] = np.nan
            
            for j, date in enumerate(dates):
                data.append({
                    'Page': page,
                    'Date': date,
                    'Access': access_type,
                    'Traffic': traffic[j] if not np.isnan(traffic[j]) else np.nan
                })
        
        df = pd.DataFrame(data)
        print(f"Created synthetic data with shape: {df.shape}")
        return df
    
    def handle_missing_values(self, data, strategy='zero'):
        """
        Handle missing values in the dataset.
        
        Args:
            data (pd.DataFrame): Input data
            strategy (str): Strategy for handling missing values ('zero', 'forward_fill', 'interpolate')
            
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        data = data.copy()
        
        if strategy == 'zero':
            data['Traffic'] = data['Traffic'].fillna(0)
        elif strategy == 'forward_fill':
            data['Traffic'] = data.groupby(['Page', 'Access'])['Traffic'].fillna(method='ffill').fillna(0)
        elif strategy == 'interpolate':
            data['Traffic'] = data.groupby(['Page', 'Access'])['Traffic'].interpolate().fillna(0)
        
        print(f"Missing values handled using '{strategy}' strategy")
        return data
    
    def apply_log_transformation(self, data):
        """
        Apply log(x + 1) transformation to traffic data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with log transformation applied
        """
        data = data.copy()
        data['log_traffic'] = np.log1p(data['Traffic'])
        print("Applied log(x + 1) transformation")
        return data
    
    def extract_temporal_features(self, data):
        """
        Extract temporal features from the data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with temporal features
        """
        data = data.copy()
        
        # Convert Date to datetime if it's not already
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Extract temporal features
        data['year'] = data['Date'].dt.year
        data['month'] = data['Date'].dt.month
        data['quarter'] = data['Date'].dt.quarter
        data['day_of_week'] = data['Date'].dt.dayofweek
        data['day_of_year'] = data['Date'].dt.dayofyear
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # One-hot encode day of week
        day_dummies = pd.get_dummies(data['day_of_week'], prefix='day')
        data = pd.concat([data, day_dummies], axis=1)
        
        # One-hot encode access type
        access_dummies = pd.get_dummies(data['Access'], prefix='access')
        data = pd.concat([data, access_dummies], axis=1)
        
        print("Extracted temporal features: year, month, quarter, day_of_week, is_weekend, access_type")
        return data
    
    def create_lag_features(self, data, lags=[1, 7, 14, 30]):
        """
        Create lag features for time series modeling.
        
        Args:
            data (pd.DataFrame): Input data
            lags (list): List of lag periods to create
            
        Returns:
            pd.DataFrame: Data with lag features
        """
        data = data.copy()
        data = data.sort_values(['Page', 'Access', 'Date'])
        
        for lag in lags:
            data[f'log_traffic_lag_{lag}'] = data.groupby(['Page', 'Access'])['log_traffic'].shift(lag)
        
        # Create rolling statistics
        for window in [7, 14, 30]:
            rolling_mean = (
                data.groupby(['Page', 'Access'])['log_traffic']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
            
            rolling_std = (
                data.groupby(['Page', 'Access'])['log_traffic']
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=[0, 1], drop=True)
            )
            
            # Ensure the rolling statistics have the same index as the main dataframe
            data = data.reset_index(drop=True)
            rolling_mean = rolling_mean.reset_index(drop=True)
            rolling_std = rolling_std.reset_index(drop=True)
            
            data[f'log_traffic_rolling_mean_{window}'] = rolling_mean
            data[f'log_traffic_rolling_std_{window}'] = rolling_std
        
        print(f"Created lag features for lags: {lags}")
        print("Created rolling statistics (mean, std) for windows: 7, 14, 30")
        return data
    
    def prepare_features(self, data):
        """
        Prepare feature matrix for modeling.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Feature matrix
        """
        # Select feature columns
        feature_cols = [
            'log_traffic', 'year', 'month', 'quarter', 'day_of_year', 'is_weekend'
        ]
        
        # Add day of week dummy columns
        day_cols = [col for col in data.columns if col.startswith('day_')]
        feature_cols.extend(day_cols)
        
        # Add access type dummy columns
        access_cols = [col for col in data.columns if col.startswith('access_')]
        feature_cols.extend(access_cols)
        
        # Add lag features
        lag_cols = [col for col in data.columns if col.startswith('log_traffic_lag_')]
        feature_cols.extend(lag_cols)
        
        # Add rolling statistics
        rolling_cols = [col for col in data.columns if col.startswith('log_traffic_rolling_')]
        feature_cols.extend(rolling_cols)
        
        self.feature_columns = feature_cols
        
        # Create feature matrix
        features = data[feature_cols].copy()
        
        # Handle any remaining NaN values
        features = features.fillna(0)
        
        print(f"Prepared feature matrix with {len(feature_cols)} features")
        return features
    
    def split_data(self, data, test_size=0.2, time_series_split=True):
        """
        Split data into train and test sets.
        
        Args:
            data (pd.DataFrame): Input data
            test_size (float): Proportion of data for testing
            time_series_split (bool): Whether to use time-based split
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if time_series_split:
            # Time-based split: use last portion for testing
            split_date = data['Date'].quantile(1 - test_size)
            train_mask = data['Date'] < split_date
            test_mask = data['Date'] >= split_date
            
            train_data = data[train_mask]
            test_data = data[test_mask]
        else:
            # Random split
            train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        
        # Prepare features and targets
        X_train = self.prepare_features(train_data)
        X_test = self.prepare_features(test_data)
        
        y_train = train_data['log_traffic'].values
        y_test = test_data['log_traffic'].values
        
        print(f"Data split: Train {len(X_train)}, Test {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def prepare_sequence_data(self, data, lookback=30, target_col='log_traffic'):
        """
        Prepare data for sequence-based models (LSTM, CNN).
        
        Args:
            data (pd.DataFrame): Input data
            lookback (int): Number of past days to use for prediction
            target_col (str): Target column name
            
        Returns:
            tuple: (X_seq, y_seq) - sequence data and targets
        """
        sequences = []
        targets = []
        
        # Group by page and access type
        for (page, access), group in data.groupby(['Page', 'Access']):
            group = group.sort_values('Date')
            
            # Create sequences
            for i in range(lookback, len(group)):
                seq = group.iloc[i-lookback:i][target_col].values
                target = group.iloc[i][target_col]
                
                sequences.append(seq)
                targets.append(target)
        
        X_seq = np.array(sequences)
        y_seq = np.array(targets)
        
        print(f"Created {len(sequences)} sequences with lookback={lookback}")
        return X_seq, y_seq
    
    def preprocess_pipeline(self, file_path=None, sample_size=10000):
        """
        Complete preprocessing pipeline.
        
        Args:
            file_path (str): Path to data file
            sample_size (int): Sample size for synthetic data
            
        Returns:
            dict: Preprocessed data and metadata
        """
        print("Starting data preprocessing pipeline...")
        
        # Load data
        data = self.load_data(file_path, sample_size)
        
        # Handle missing values
        data = self.handle_missing_values(data, strategy='zero')
        
        # Apply log transformation
        data = self.apply_log_transformation(data)
        
        # Extract temporal features
        data = self.extract_temporal_features(data)
        
        # Create lag features
        data = self.create_lag_features(data)
        
        # Remove rows with NaN values (from lag features)
        data = data.dropna()
        
        print(f"Final dataset shape: {data.shape}")
        
        return {
            'data': data,
            'feature_columns': self.feature_columns,
            'preprocessor': self
        }


if __name__ == "__main__":
    # Example usage
    preprocessor = WikiTrafficPreprocessor()
    result = preprocessor.preprocess_pipeline(sample_size=1000)
    
    print("\nPreprocessing completed successfully!")
    print(f"Final data shape: {result['data'].shape}")
    print(f"Feature columns: {len(result['feature_columns'])}")

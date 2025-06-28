"""
Data Loading Module for Tabular Regression
Handles loading data from various sources for regression tasks
"""

import pandas as pd
import numpy as np
from sklearn.datasets import (
    load_boston, load_diabetes, load_california_housing,
    make_regression, fetch_california_housing
)
from typing import Tuple, Optional
import os


class RegressionDataLoader:
    """Class to handle data loading for regression tasks"""
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.data = None
        self.target = None
        
    def load_sample_dataset(self, dataset_name: str = "california_housing") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load sample datasets for regression demonstration
        
        Args:
            dataset_name: Name of the dataset ('california_housing', 'diabetes', 'synthetic')
            
        Returns:
            Tuple of (features_dataframe, target_series)
        """
        print(f"Loading {dataset_name} dataset...")
        
        if dataset_name == "california_housing":
            dataset = fetch_california_housing()
            feature_names = dataset.feature_names
            
        elif dataset_name == "diabetes":
            dataset = load_diabetes()
            feature_names = dataset.feature_names
            
        elif dataset_name == "synthetic":
            # Create synthetic regression dataset
            X, y = make_regression(
                n_samples=1000, n_features=10, n_informative=5,
                noise=0.1, random_state=42
            )
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
            self.data = pd.DataFrame(X, columns=feature_names)
            self.target = pd.Series(y, name='target')
            
            print(f"Generated synthetic dataset:")
            print(f"Shape: {self.data.shape}")
            print(f"Features: {list(self.data.columns)}")
            print(f"Target range: [{self.target.min():.2f}, {self.target.max():.2f}]")
            
            return self.data.copy(), self.target.copy()
            
        else:
            raise ValueError(f"Dataset '{dataset_name}' not supported. Choose from ['california_housing', 'diabetes', 'synthetic']")
        
        # Create DataFrame with feature names
        self.data = pd.DataFrame(dataset.data, columns=feature_names)
        self.target = pd.Series(dataset.target, name='target')
        
        print(f"Loaded {dataset_name} dataset:")
        print(f"Shape: {self.data.shape}")
        print(f"Features: {list(self.data.columns)}")
        print(f"Target range: [{self.target.min():.2f}, {self.target.max():.2f}]")
        print(f"Target mean: {self.target.mean():.2f}")
        print(f"Target std: {self.target.std():.2f}")
        
        return self.data.copy(), self.target.copy()
    
    def load_csv_data(self, file_path: str, target_column: str, 
                     separator: str = ',', encoding: str = 'utf-8') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load regression data from CSV file
        
        Args:
            file_path: Path to CSV file
            target_column: Name of target column
            separator: CSV separator
            encoding: File encoding
            
        Returns:
            Tuple of (features_dataframe, target_series)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load data
        df = pd.read_csv(file_path, sep=separator, encoding=encoding)
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Separate features and target
        self.target = df[target_column].copy()
        self.data = df.drop(columns=[target_column])
        
        # Validate target is numeric
        if not pd.api.types.is_numeric_dtype(self.target):
            print(f"Warning: Target column '{target_column}' is not numeric. Attempting conversion...")
            try:
                self.target = pd.to_numeric(self.target, errors='coerce')
                if self.target.isnull().any():
                    print(f"Warning: {self.target.isnull().sum()} target values could not be converted to numeric")
            except:
                raise ValueError(f"Target column '{target_column}' cannot be converted to numeric")
        
        print(f"Loaded data from {file_path}:")
        print(f"Shape: {self.data.shape}")
        print(f"Target range: [{self.target.min():.2f}, {self.target.max():.2f}]")
        print(f"Target mean: {self.target.mean():.2f}")
        print(f"Target std: {self.target.std():.2f}")
        
        return self.data.copy(), self.target.copy()
    
    def create_time_series_features(self, df: pd.DataFrame, 
                                  date_column: str,
                                  lag_features: int = 5) -> pd.DataFrame:
        """
        Create time series features for regression
        
        Args:
            df: DataFrame with time series data
            date_column: Name of the date column
            lag_features: Number of lag features to create
            
        Returns:
            DataFrame with time series features
        """
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found")
        
        df_ts = df.copy()
        
        # Convert date column to datetime
        df_ts[date_column] = pd.to_datetime(df_ts[date_column])
        
        # Sort by date
        df_ts = df_ts.sort_values(date_column)
        
        # Extract date features
        df_ts['year'] = df_ts[date_column].dt.year
        df_ts['month'] = df_ts[date_column].dt.month
        df_ts['day'] = df_ts[date_column].dt.day
        df_ts['dayofweek'] = df_ts[date_column].dt.dayofweek
        df_ts['quarter'] = df_ts[date_column].dt.quarter
        df_ts['is_weekend'] = (df_ts[date_column].dt.dayofweek >= 5).astype(int)
        
        # Create lag features for numeric columns
        numeric_cols = df_ts.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend']]
        
        for col in numeric_cols:
            for lag in range(1, lag_features + 1):
                df_ts[f'{col}_lag_{lag}'] = df_ts[col].shift(lag)
        
        # Create rolling features
        for col in numeric_cols:
            df_ts[f'{col}_rolling_mean_3'] = df_ts[col].rolling(window=3).mean()
            df_ts[f'{col}_rolling_std_3'] = df_ts[col].rolling(window=3).std()
        
        # Drop rows with NaN values created by lag and rolling features
        df_ts = df_ts.dropna()
        
        print(f"Created time series features:")
        print(f"Original shape: {df.shape}")
        print(f"New shape: {df_ts.shape}")
        print(f"Created {df_ts.shape[1] - df.shape[1]} new features")
        
        return df_ts
    
    def get_data_info(self) -> dict:
        """Get basic information about the loaded data"""
        if self.data is None:
            return {"message": "No data loaded"}
        
        return {
            "shape": self.data.shape,
            "features": list(self.data.columns),
            "feature_types": dict(self.data.dtypes),
            "missing_values": dict(self.data.isnull().sum()),
            "target_stats": {
                "min": float(self.target.min()),
                "max": float(self.target.max()),
                "mean": float(self.target.mean()),
                "std": float(self.target.std()),
                "missing": int(self.target.isnull().sum())
            }
        }


def main():
    """Example usage of RegressionDataLoader"""
    # Example 1: Load sample dataset
    loader = RegressionDataLoader()
    
    print("=== LOADING CALIFORNIA HOUSING DATASET ===")
    X, y = loader.load_sample_dataset("california_housing")
    
    print("\nData Info:")
    info = loader.get_data_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\n=== LOADING DIABETES DATASET ===")
    X_diabetes, y_diabetes = loader.load_sample_dataset("diabetes")
    
    print("\n=== LOADING SYNTHETIC DATASET ===")
    X_synthetic, y_synthetic = loader.load_sample_dataset("synthetic")
    
    # Example 2: Time series features (using synthetic data with dates)
    print("\n=== CREATING TIME SERIES FEATURES ===")
    # Create sample time series data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    ts_data = pd.DataFrame({
        'date': dates,
        'value1': np.random.randn(100).cumsum(),
        'value2': np.random.randn(100),
        'target': np.random.randn(100)
    })
    
    ts_features = loader.create_time_series_features(ts_data, 'date')
    print(f"Time series features created: {list(ts_features.columns)}")


if __name__ == "__main__":
    main()

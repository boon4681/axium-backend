"""
Data Loading Module for Tabular Classification
Handles loading data from various sources and initial data validation
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from typing import Tuple, Optional
import os


class ClassificationDataLoader:
    """Class to handle data loading for classification tasks"""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.data = None
        self.target = None

    def load_sample_dataset(self, dataset_name: str = "iris") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load sample datasets for demonstration

        Args:
            dataset_name: Name of the dataset ('iris', 'breast_cancer', 'wine')

        Returns:
            Tuple of (features_dataframe, target_series)
        """
        dataset_loaders = {
            'iris': load_iris,
            'breast_cancer': load_breast_cancer,
            'wine': load_wine
        }

        if dataset_name not in dataset_loaders:
            raise ValueError(
                f"Dataset '{dataset_name}' not supported. Choose from {list(dataset_loaders.keys())}")

        dataset = dataset_loaders[dataset_name]()

        # Create DataFrame with feature names
        self.data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        self.target = pd.Series(dataset.target, name='target')

        print(f"Loaded {dataset_name} dataset:")
        print(f"Shape: {self.data.shape}")
        print(f"Features: {list(self.data.columns)}")
        print(f"Target classes: {np.unique(self.target)}")

        return self.data.copy(), self.target.copy()

    def load_csv_data(self, file_path: str, target_column: str,
                      separator: str = ',', encoding: str = 'utf-8') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load data from CSV file

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
            raise ValueError(
                f"Target column '{target_column}' not found in data")

        # Separate features and target
        self.target = df[target_column].copy()
        self.data = df.drop(columns=[target_column])

        print(f"Loaded data from {file_path}:")
        print(f"Shape: {self.data.shape}")
        print(f"Target distribution:")
        print(self.target.value_counts())

        return self.data.copy(), self.target.copy()

    def get_data_info(self) -> dict:
        """Get basic information about the loaded data"""
        if self.data is None:
            return {"message": "No data loaded"}

        return {
            "shape": self.data.shape,
            "features": list(self.data.columns),
            "feature_types": dict(self.data.dtypes),
            "missing_values": dict(self.data.isnull().sum()),
            "target_classes": list(np.unique(self.target)) if self.target is not None else None,
            "target_distribution": dict(self.target.value_counts()) if self.target is not None else None
        }


def main():
    """Example usage of DataLoader"""
    # Example 1: Load sample dataset
    loader = DataLoader()
    X, y = loader.load_sample_dataset("iris")

    print("\nData Info:")
    info = loader.get_data_info()
    for key, value in info.items():
        print(f"{key}: {value}")

    # Example 2: Load CSV data (uncomment when you have a CSV file)
    # X_csv, y_csv = loader.load_csv_data("your_data.csv", "target_column")


if __name__ == "__main__":
    main()

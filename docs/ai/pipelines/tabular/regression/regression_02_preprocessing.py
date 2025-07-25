"""
Data Preprocessing Module for Tabular Regression
Handles data preprocessing, feature engineering, and preparation for regression tasks
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures,
    PowerTransformer, QuantileTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats
from typing import Tuple, Optional, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class RegressionPreprocessor:
    """Class to handle preprocessing for regression tasks"""

    def __init__(self):
        self.numeric_transformer = None
        self.categorical_transformer = None
        self.feature_selector = None
        self.target_transformer = None
        self.preprocessor = None
        self.feature_names = None
        self.polynomial_features = None

    def analyze_data(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Analyze the dataset and provide insights

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'shape': X.shape,
            'numeric_features': list(X.select_dtypes(include=[np.number]).columns),
            'categorical_features': list(X.select_dtypes(include=['object', 'category']).columns),
            'missing_values': dict(X.isnull().sum()),
            'target_stats': {
                'mean': y.mean(),
                'std': y.std(),
                'min': y.min(),
                'max': y.max(),
                'median': y.median(),
                'skewness': y.skew(),
                'kurtosis': y.kurtosis()
            },
            'feature_correlations': X.select_dtypes(include=[np.number]).corrwith(y).abs().sort_values(ascending=False).to_dict()
        }

        print("=== DATA ANALYSIS ===")
        print(f"Dataset shape: {analysis['shape']}")
        print(f"Numeric features: {len(analysis['numeric_features'])}")
        print(f"Categorical features: {len(analysis['categorical_features'])}")
        print(f"Missing values: {sum(analysis['missing_values'].values())}")
        print(f"Target statistics:")
        for key, value in analysis['target_stats'].items():
            print(f"  {key}: {value:.4f}")

        return analysis

    def handle_missing_values(self, X: pd.DataFrame,
                              numeric_strategy: str = 'mean',
                              categorical_strategy: str = 'most_frequent') -> pd.DataFrame:
        """
        Handle missing values in the dataset

        Args:
            X: Feature DataFrame
            numeric_strategy: Strategy for numeric features ('mean', 'median', 'mode', 'knn')
            categorical_strategy: Strategy for categorical features ('most_frequent', 'constant')

        Returns:
            DataFrame with missing values handled
        """
        X_filled = X.copy()

        # Get numeric and categorical columns
        numeric_cols = X_filled.select_dtypes(include=[np.number]).columns
        categorical_cols = X_filled.select_dtypes(
            include=['object', 'category']).columns

        # Handle numeric missing values
        if len(numeric_cols) > 0:
            if numeric_strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
                X_filled[numeric_cols] = imputer.fit_transform(
                    X_filled[numeric_cols])
            else:
                imputer = SimpleImputer(strategy=numeric_strategy)
                X_filled[numeric_cols] = imputer.fit_transform(
                    X_filled[numeric_cols])

        # Handle categorical missing values
        if len(categorical_cols) > 0:
            if categorical_strategy == 'constant':
                imputer = SimpleImputer(
                    strategy='constant', fill_value='missing')
            else:
                imputer = SimpleImputer(strategy=categorical_strategy)
            X_filled[categorical_cols] = imputer.fit_transform(
                X_filled[categorical_cols])

        print(
            f"Missing values handled: {X.isnull().sum().sum()} -> {X_filled.isnull().sum().sum()}")
        return X_filled

    def encode_categorical_features(self, X: pd.DataFrame,
                                    encoding_type: str = 'onehot',
                                    max_categories: int = 10) -> pd.DataFrame:
        """
        Encode categorical features

        Args:
            X: Feature DataFrame
            encoding_type: Type of encoding ('onehot', 'ordinal', 'target')
            max_categories: Maximum categories for one-hot encoding

        Returns:
            DataFrame with encoded categorical features
        """
        X_encoded = X.copy()
        categorical_cols = X_encoded.select_dtypes(
            include=['object', 'category']).columns

        if len(categorical_cols) == 0:
            return X_encoded

        for col in categorical_cols:
            unique_values = X_encoded[col].nunique()

            if encoding_type == 'onehot' and unique_values <= max_categories:
                # One-hot encoding
                dummies = pd.get_dummies(
                    X_encoded[col], prefix=col, drop_first=True)
                X_encoded = pd.concat(
                    [X_encoded.drop(col, axis=1), dummies], axis=1)

            else:
                # Ordinal encoding for high cardinality features
                from sklearn.preprocessing import OrdinalEncoder
                encoder = OrdinalEncoder(
                    handle_unknown='use_encoded_value', unknown_value=-1)
                X_encoded[col] = encoder.fit_transform(X_encoded[[col]])

        print(
            f"Categorical encoding completed. Shape: {X.shape} -> {X_encoded.shape}")
        return X_encoded

    def detect_outliers(self, X: pd.DataFrame, method: str = 'iqr',
                        threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers in numerical features

        Args:
            X: Feature DataFrame
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outlier information
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        outlier_info = pd.DataFrame()

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = (X[col] < lower_bound) | (X[col] > upper_bound)

            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(X[col]))
                outliers = z_scores > threshold

            outlier_info[col] = outliers

        outlier_counts = outlier_info.sum()
        print(f"Outliers detected (method: {method}):")
        for col, count in outlier_counts.items():
            if count > 0:
                print(f"  {col}: {count} outliers ({count/len(X)*100:.1f}%)")

        return outlier_info

    def remove_outliers(self, X: pd.DataFrame, y: pd.Series,
                        method: str = 'iqr', threshold: float = 1.5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Remove outliers from the dataset

        Args:
            X: Feature DataFrame
            y: Target Series
            method: Outlier detection method
            threshold: Threshold for outlier detection

        Returns:
            Tuple of (X_clean, y_clean) with outliers removed
        """
        outlier_info = self.detect_outliers(X, method, threshold)

        # Remove rows with outliers in any column
        outlier_mask = outlier_info.any(axis=1)
        X_clean = X[~outlier_mask].copy()
        y_clean = y[~outlier_mask].copy()

        print(
            f"Outliers removed: {outlier_mask.sum()} rows ({outlier_mask.sum()/len(X)*100:.1f}%)")
        print(f"Dataset shape: {X.shape} -> {X_clean.shape}")

        return X_clean, y_clean

    def transform_target(self, y: pd.Series,
                         method: str = 'log') -> Tuple[pd.Series, Any]:
        """
        Transform target variable for better distribution

        Args:
            y: Target Series
            method: Transformation method ('log', 'sqrt', 'box-cox', 'yeo-johnson')

        Returns:
            Tuple of (transformed_target, transformer)
        """
        if method == 'log':
            if (y <= 0).any():
                # Add constant to make all values positive
                y_shifted = y - y.min() + 1
                y_transformed = np.log(y_shifted)
                transformer = {'method': 'log', 'shift': y.min() - 1}
            else:
                y_transformed = np.log(y)
                transformer = {'method': 'log', 'shift': 0}

        elif method == 'sqrt':
            if (y < 0).any():
                y_shifted = y - y.min()
                y_transformed = np.sqrt(y_shifted)
                transformer = {'method': 'sqrt', 'shift': y.min()}
            else:
                y_transformed = np.sqrt(y)
                transformer = {'method': 'sqrt', 'shift': 0}

        elif method == 'box-cox':
            if (y <= 0).any():
                y_shifted = y - y.min() + 1
            else:
                y_shifted = y
            y_transformed, lambda_param = stats.boxcox(y_shifted)
            transformer = {'method': 'box-cox', 'lambda': lambda_param,
                           'shift': y.min() - 1 if (y <= 0).any() else 0}

        elif method == 'yeo-johnson':
            transformer_obj = PowerTransformer(method='yeo-johnson')
            y_transformed = transformer_obj.fit_transform(
                y.values.reshape(-1, 1)).flatten()
            transformer = {'method': 'yeo-johnson',
                           'transformer': transformer_obj}

        else:
            raise ValueError(f"Unknown transformation method: {method}")

        y_transformed = pd.Series(y_transformed, index=y.index, name=y.name)
        self.target_transformer = transformer

        print(f"Target transformation completed ({method}):")
        print(f"  Original skewness: {y.skew():.4f}")
        print(f"  Transformed skewness: {y_transformed.skew():.4f}")

        return y_transformed, transformer

    def scale_features(self, X: pd.DataFrame,
                       scaling_type: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features

        Args:
            X: Feature DataFrame
            scaling_type: Type of scaling ('standard', 'minmax', 'robust', 'quantile')

        Returns:
            DataFrame with scaled features
        """
        X_scaled = X.copy()
        numeric_cols = X_scaled.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return X_scaled

        if scaling_type == 'standard':
            scaler = StandardScaler()
        elif scaling_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_type == 'robust':
            scaler = RobustScaler()
        elif scaling_type == 'quantile':
            scaler = QuantileTransformer(output_distribution='normal')
        else:
            raise ValueError(f"Unknown scaling type: {scaling_type}")

        X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])
        print(f"Feature scaling completed using {scaling_type} scaler")
        return X_scaled

    def create_polynomial_features(self, X: pd.DataFrame,
                                   degree: int = 2,
                                   include_bias: bool = False) -> pd.DataFrame:
        """
        Create polynomial features

        Args:
            X: Feature DataFrame
            degree: Polynomial degree
            include_bias: Whether to include bias term

        Returns:
            DataFrame with polynomial features
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return X

        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X_poly = poly.fit_transform(X[numeric_cols])

        # Create feature names
        feature_names = poly.get_feature_names_out(numeric_cols)
        X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)

        # Add categorical features back
        categorical_cols = X.select_dtypes(
            include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            X_poly_df = pd.concat([X_poly_df, X[categorical_cols]], axis=1)

        self.polynomial_features = poly
        print(
            f"Polynomial features created (degree {degree}): {X.shape[1]} -> {X_poly_df.shape[1]} features")
        return X_poly_df

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                        method: str = 'f_regression', k: int = 10) -> pd.DataFrame:
        """
        Select best features for regression

        Args:
            X: Feature DataFrame
            y: Target Series
            method: Feature selection method ('f_regression', 'mutual_info')
            k: Number of features to select

        Returns:
            DataFrame with selected features
        """
        # Ensure all features are numeric for feature selection
        X_numeric = X.select_dtypes(include=[np.number])

        if len(X_numeric.columns) == 0:
            print("No numeric features available for selection")
            return X

        if method == 'f_regression':
            selector = SelectKBest(
                score_func=f_regression, k=min(k, len(X_numeric.columns)))
        elif method == 'mutual_info':
            selector = SelectKBest(
                score_func=mutual_info_regression, k=min(k, len(X_numeric.columns)))
        else:
            raise ValueError(f"Unknown feature selection method: {method}")

        X_selected = selector.fit_transform(X_numeric, y)
        selected_features = X_numeric.columns[selector.get_support()]

        # Get feature scores
        feature_scores = selector.scores_[selector.get_support()]
        score_df = pd.DataFrame({
            'feature': selected_features,
            'score': feature_scores
        }).sort_values('score', ascending=False)

        # Combine selected numeric features with categorical features
        categorical_cols = X.select_dtypes(
            include=['object', 'category']).columns
        X_final = pd.concat([
            pd.DataFrame(X_selected, columns=selected_features, index=X.index),
            X[categorical_cols]
        ], axis=1)

        print(
            f"Feature selection completed: {len(X_numeric.columns)} -> {len(selected_features)} numeric features")
        print("Top selected features:")
        for _, row in score_df.head().iterrows():
            print(f"  {row['feature']}: {row['score']:.4f}")

        return X_final

    def prepare_data(self, X: pd.DataFrame, y: pd.Series,
                     test_size: float = 0.2,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training (split)

        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of test set
            random_state: Random state for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"Data split completed:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(
            f"Train target stats: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
        print(
            f"Test target stats: mean={y_test.mean():.4f}, std={y_test.std():.4f}")

        return X_train, X_test, y_train, y_test

    def full_preprocessing_pipeline(self, X: pd.DataFrame, y: pd.Series,
                                    test_size: float = 0.2,
                                    handle_missing: bool = True,
                                    encode_categorical: bool = True,
                                    remove_outliers: bool = False,
                                    transform_target: bool = False,
                                    target_transform_method: str = 'log',
                                    scale_features: bool = True,
                                    scaling_method: str = 'standard',
                                    create_polynomial: bool = False,
                                    poly_degree: int = 2,
                                    select_features: bool = False,
                                    feature_selection_k: int = 10,
                                    # New kwargs for customization
                                    missing_value_strategy: str = 'mean',
                                    categorical_missing_strategy: str = 'most_frequent',
                                    categorical_encoding: str = 'onehot',
                                    max_categories: int = 10,
                                    outlier_method: str = 'iqr',
                                    outlier_threshold: float = 1.5,
                                    feature_selection_method: str = 'f_regression',
                                    random_state: int = 42) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline for regression

        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of test set
            handle_missing: Whether to handle missing values
            encode_categorical: Whether to encode categorical features
            remove_outliers: Whether to remove outliers
            transform_target: Whether to transform target variable
            target_transform_method: Method for target transformation
            scale_features: Whether to scale features
            scaling_method: Method for feature scaling
            create_polynomial: Whether to create polynomial features
            poly_degree: Degree for polynomial features
            select_features: Whether to perform feature selection
            feature_selection_k: Number of features to select
            missing_value_strategy: Strategy for numeric missing values ('mean', 'median', 'mode', 'knn')
            categorical_missing_strategy: Strategy for categorical missing values ('most_frequent', 'constant')
            categorical_encoding: Encoding method ('onehot', 'ordinal')
            max_categories: Maximum categories for one-hot encoding
            outlier_method: Outlier detection method ('iqr', 'zscore')
            outlier_threshold: Threshold for outlier detection
            feature_selection_method: Feature selection method ('f_regression', 'mutual_info')
            random_state: Random state for reproducibility

        Returns:
            Dictionary containing processed data and metadata
        """
        print("=== STARTING FULL PREPROCESSING PIPELINE ===")

        # Analyze data
        analysis = self.analyze_data(X, y)

        # Handle missing values
        if handle_missing:
            X = self.handle_missing_values(X,
                                           numeric_strategy=missing_value_strategy,
                                           categorical_strategy=categorical_missing_strategy)

        # Remove outliers
        if remove_outliers:
            X, y = self.remove_outliers(
                X, y, method=outlier_method, threshold=outlier_threshold)

        # Transform target
        y_transformed = y
        target_transformer = None
        if transform_target:
            y_transformed, target_transformer = self.transform_target(
                y, target_transform_method)

        # Encode categorical features
        if encode_categorical:
            X = self.encode_categorical_features(X,
                                                 encoding_type=categorical_encoding,
                                                 max_categories=max_categories)

        # Create polynomial features
        if create_polynomial:
            X = self.create_polynomial_features(X, degree=poly_degree)

        # Feature selection (before scaling to avoid issues)
        if select_features:
            X = self.select_features(
                X, y_transformed, method=feature_selection_method, k=feature_selection_k)

        # Scale features
        if scale_features:
            X = self.scale_features(X, scaling_type=scaling_method)

        # Split data
        X_train, X_test, y_train, y_test = self.prepare_data(
            X, y_transformed, test_size=test_size, random_state=random_state)

        result = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_original': y,
            'feature_names': list(X.columns),
            'analysis': analysis,
            'target_transformer': target_transformer,
            'polynomial_features': self.polynomial_features,
            'preprocessing_config': {
                'missing_value_strategy': missing_value_strategy,
                'categorical_missing_strategy': categorical_missing_strategy,
                'categorical_encoding': categorical_encoding,
                'max_categories': max_categories,
                'outlier_method': outlier_method,
                'outlier_threshold': outlier_threshold,
                'target_transform_method': target_transform_method,
                'scaling_method': scaling_method,
                'poly_degree': poly_degree,
                'feature_selection_method': feature_selection_method,
                'feature_selection_k': feature_selection_k,
                'test_size': test_size,
                'random_state': random_state
            }
        }

        print("=== PREPROCESSING PIPELINE COMPLETED ===")
        return result


def main():
    """Example usage of RegressionPreprocessor"""
    from sklearn.datasets import load_california_housing, load_diabetes

    # Load sample dataset
    print("Loading California housing dataset...")
    data = load_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')

    # Create preprocessor
    preprocessor = RegressionPreprocessor()

    # Run full preprocessing pipeline
    result = preprocessor.full_preprocessing_pipeline(
        X, y,
        test_size=0.2,
        handle_missing=True,
        encode_categorical=True,
        remove_outliers=True,
        transform_target=True,
        target_transform_method='log',
        scale_features=True,
        scaling_method='standard',
        create_polynomial=False,
        select_features=True,
        feature_selection_k=6
    )

    print(f"\nFinal training set shape: {result['X_train'].shape}")
    print(f"Final test set shape: {result['X_test'].shape}")
    print(f"Selected features: {len(result['feature_names'])}")

    # Show target transformation effect
    if result['target_transformer']:
        print(f"\nTarget transformation:")
        print(
            f"Original target range: [{result['y_original'].min():.4f}, {result['y_original'].max():.4f}]")
        print(
            f"Transformed target range: [{result['y_train'].min():.4f}, {result['y_train'].max():.4f}]")


if __name__ == "__main__":
    main()

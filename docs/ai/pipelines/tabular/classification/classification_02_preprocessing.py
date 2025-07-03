"""
Data Preprocessing Module for Tabular Classification
Handles data preprocessing, feature engineering, and preparation for classification tasks
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder,
    OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Optional, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class ClassificationPreprocessor:
    """Class to handle preprocessing for classification tasks"""

    def __init__(self):
        self.numeric_transformer = None
        self.categorical_transformer = None
        self.feature_selector = None
        self.label_encoder = None
        self.preprocessor = None
        self.feature_names = None

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
            'target_classes': list(y.unique()),
            'target_distribution': dict(y.value_counts()),
            'class_balance': y.value_counts(normalize=True).to_dict()
        }

        print("=== DATA ANALYSIS ===")
        print(f"Dataset shape: {analysis['shape']}")
        print(f"Numeric features: {len(analysis['numeric_features'])}")
        print(f"Categorical features: {len(analysis['categorical_features'])}")
        print(f"Missing values: {sum(analysis['missing_values'].values())}")
        print(f"Target classes: {analysis['target_classes']}")
        print(f"Class distribution: {analysis['target_distribution']}")

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
            encoding_type: Type of encoding ('onehot', 'ordinal', 'label')
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

            elif encoding_type == 'ordinal' or unique_values > max_categories:
                # Ordinal encoding for high cardinality features
                encoder = OrdinalEncoder(
                    handle_unknown='use_encoded_value', unknown_value=-1)
                X_encoded[col] = encoder.fit_transform(X_encoded[[col]])

            elif encoding_type == 'label':
                # Label encoding
                encoder = LabelEncoder()
                X_encoded[col] = encoder.fit_transform(X_encoded[col])

        print(
            f"Categorical encoding completed. Shape: {X.shape} -> {X_encoded.shape}")
        return X_encoded

    def scale_features(self, X: pd.DataFrame,
                       scaling_type: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features

        Args:
            X: Feature DataFrame
            scaling_type: Type of scaling ('standard', 'minmax', 'robust')

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
        else:
            raise ValueError(f"Unknown scaling type: {scaling_type}")

        X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])
        print(f"Feature scaling completed using {scaling_type} scaler")
        return X_scaled

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                        method: str = 'chi2', k: int = 10) -> pd.DataFrame:
        """
        Select best features for classification

        Args:
            X: Feature DataFrame
            y: Target Series
            method: Feature selection method ('chi2', 'f_classif', 'mutual_info')
            k: Number of features to select

        Returns:
            DataFrame with selected features
        """
        # Ensure all features are numeric for feature selection
        X_numeric = X.select_dtypes(include=[np.number])

        if len(X_numeric.columns) == 0:
            print("No numeric features available for selection")
            return X

        if method == 'chi2':
            # Chi-square test requires non-negative features
            X_numeric = X_numeric - X_numeric.min() + 1e-6
            selector = SelectKBest(
                score_func=chi2, k=min(k, len(X_numeric.columns)))
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif,
                                   k=min(k, len(X_numeric.columns)))
        else:
            raise ValueError(f"Unknown feature selection method: {method}")

        X_selected = selector.fit_transform(X_numeric, y)
        selected_features = X_numeric.columns[selector.get_support()]

        # Combine selected numeric features with categorical features
        categorical_cols = X.select_dtypes(
            include=['object', 'category']).columns
        X_final = pd.concat([
            pd.DataFrame(X_selected, columns=selected_features, index=X.index),
            X[categorical_cols]
        ], axis=1)

        print(
            f"Feature selection completed: {len(X_numeric.columns)} -> {len(selected_features)} numeric features")
        return X_final

    def create_preprocessing_pipeline(self, X: pd.DataFrame,
                                      numeric_strategy: str = 'standard',
                                      categorical_strategy: str = 'onehot') -> ColumnTransformer:
        """
        Create a preprocessing pipeline

        Args:
            X: Feature DataFrame
            numeric_strategy: Strategy for numeric features
            categorical_strategy: Strategy for categorical features

        Returns:
            ColumnTransformer pipeline
        """
        numeric_features = X.select_dtypes(
            include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(
            include=['object', 'category']).columns.tolist()

        # Numeric preprocessing
        if numeric_strategy == 'standard':
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
        elif numeric_strategy == 'minmax':
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', MinMaxScaler())
            ])
        elif numeric_strategy == 'robust':
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', RobustScaler())
            ])

        # Categorical preprocessing
        if categorical_strategy == 'onehot':
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
            ])
        elif categorical_strategy == 'ordinal':
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('ordinal', OrdinalEncoder(
                    handle_unknown='use_encoded_value', unknown_value=-1))
            ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        self.preprocessor = preprocessor
        print(
            f"Preprocessing pipeline created for {len(numeric_features)} numeric and {len(categorical_features)} categorical features")
        return preprocessor

    def prepare_data(self, X: pd.DataFrame, y: pd.Series,
                     test_size: float = 0.2,
                     random_state: int = 42,
                     stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training (split and basic preprocessing)

        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of test set
            random_state: Random state for reproducibility
            stratify: Whether to stratify the split

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Encode target labels if they are strings
        if y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            y_encoded = pd.Series(y_encoded, index=y.index, name=y.name)
        else:
            y_encoded = y

        # Split data
        stratify_param = y_encoded if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state,
            stratify=stratify_param
        )

        print(f"Data split completed:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Train class distribution: {dict(y_train.value_counts())}")
        print(f"Test class distribution: {dict(y_test.value_counts())}")

        return X_train, X_test, y_train, y_test

    def full_preprocessing_pipeline(self, X: pd.DataFrame, y: pd.Series,
                                    test_size: float = 0.2,
                                    handle_missing: bool = True,
                                    encode_categorical: bool = True,
                                    scale_features: bool = True,
                                    select_features: bool = False,
                                    feature_selection_k: int = 10,
                                    # New kwargs for customization
                                    missing_value_strategy: str = 'mean',
                                    categorical_missing_strategy: str = 'most_frequent',
                                    categorical_encoding: str = 'onehot',
                                    max_categories: int = 10,
                                    scaling_method: str = 'standard',
                                    feature_selection_method: str = 'f_classif',
                                    random_state: int = 42,
                                    stratify: bool = True) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline

        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of test set
            handle_missing: Whether to handle missing values
            encode_categorical: Whether to encode categorical features
            scale_features: Whether to scale features
            select_features: Whether to perform feature selection
            feature_selection_k: Number of features to select
            missing_value_strategy: Strategy for numeric missing values ('mean', 'median', 'mode', 'knn')
            categorical_missing_strategy: Strategy for categorical missing values ('most_frequent', 'constant')
            categorical_encoding: Encoding method ('onehot', 'ordinal', 'label')
            max_categories: Maximum categories for one-hot encoding
            scaling_method: Scaling method ('standard', 'minmax', 'robust')
            feature_selection_method: Feature selection method ('chi2', 'f_classif')
            random_state: Random state for reproducibility
            stratify: Whether to stratify the split

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

        # Encode categorical features
        if encode_categorical:
            X = self.encode_categorical_features(X,
                                                 encoding_type=categorical_encoding,
                                                 max_categories=max_categories)

        # Scale features
        if scale_features:
            X = self.scale_features(X, scaling_type=scaling_method)

        # Feature selection
        if select_features:
            X = self.select_features(
                X, y, method=feature_selection_method, k=feature_selection_k)

        # Split data
        X_train, X_test, y_train, y_test = self.prepare_data(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify)

        result = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': list(X.columns),
            'analysis': analysis,
            'label_encoder': self.label_encoder,
            'preprocessing_config': {
                'missing_value_strategy': missing_value_strategy,
                'categorical_missing_strategy': categorical_missing_strategy,
                'categorical_encoding': categorical_encoding,
                'max_categories': max_categories,
                'scaling_method': scaling_method,
                'feature_selection_method': feature_selection_method,
                'feature_selection_k': feature_selection_k,
                'test_size': test_size,
                'random_state': random_state,
                'stratify': stratify
            }
        }

        print("=== PREPROCESSING PIPELINE COMPLETED ===")
        return result


def main():
    """Example usage of ClassificationPreprocessor"""
    from sklearn.datasets import load_breast_cancer, load_wine

    # Load sample dataset
    print("Loading breast cancer dataset...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')

    # Create preprocessor
    preprocessor = ClassificationPreprocessor()

    # Run full preprocessing pipeline
    result = preprocessor.full_preprocessing_pipeline(
        X, y,
        test_size=0.2,
        handle_missing=True,
        encode_categorical=True,
        scale_features=True,
        select_features=True,
        feature_selection_k=15
    )

    print(f"\nFinal training set shape: {result['X_train'].shape}")
    print(f"Final test set shape: {result['X_test'].shape}")
    print(f"Selected features: {len(result['feature_names'])}")


if __name__ == "__main__":
    main()

"""
Example: Using Updated Preprocessing with Kwargs
Shows how to use the improved preprocessing methods with direct parameter passing
"""

from pipelines.tabular.regression.regression_02_preprocessing import RegressionPreprocessor
from pipelines.tabular.regression.regression_01_load_data import RegressionDataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def demo_regression_with_kwargs():
    """Demo regression preprocessing with custom parameters"""
    print("\n" + "="*70)
    print("REGRESSION PREPROCESSING WITH KWARGS")
    print("="*70)

    # Load data
    loader = RegressionDataLoader()
    X, y = loader.load_sample_dataset("california_housing")

    # Create preprocessor
    preprocessor = RegressionPreprocessor()

    # Example 1: Conservative preprocessing
    print("\n1. CONSERVATIVE PREPROCESSING")
    print("-" * 40)
    result1 = preprocessor.full_preprocessing_pipeline(
        X, y,
        test_size=0.2,
        missing_value_strategy='mean',
        scaling_method='standard',
        remove_outliers=False,
        transform_target=False,
        select_features=False
    )
    print(f"Result shape: {result1['X_train'].shape}")
    print(f"Config: {result1['preprocessing_config']}")

    # Example 2: Aggressive preprocessing
    print("\n2. AGGRESSIVE PREPROCESSING")
    print("-" * 40)
    result2 = preprocessor.full_preprocessing_pipeline(
        X, y,
        test_size=0.25,
        missing_value_strategy='median',
        scaling_method='robust',
        remove_outliers=True,
        outlier_method='iqr',
        outlier_threshold=1.5,
        transform_target=True,
        target_transform_method='log',
        select_features=True,
        feature_selection_method='f_regression',
        feature_selection_k=6
    )
    print(f"Result shape: {result2['X_train'].shape}")
    print(f"Config: {result2['preprocessing_config']}")

    # Example 3: Custom preprocessing with polynomial features
    print("\n3. POLYNOMIAL FEATURES PREPROCESSING")
    print("-" * 40)
    result3 = preprocessor.full_preprocessing_pipeline(
        X, y,
        test_size=0.2,
        missing_value_strategy='mean',
        scaling_method='minmax',
        create_polynomial=True,
        poly_degree=2,
        select_features=True,
        feature_selection_method='f_regression',
        feature_selection_k=15
    )
    print(f"Result shape: {result3['X_train'].shape}")
    print(f"Config: {result3['preprocessing_config']}")

    return result1, result2, result3


def main():
    """Main demo function"""

    # Demo regression
    reg_results = demo_regression_with_kwargs()

    print("\n" + "="*70)
    print("PREPROCESSING WITH KWARGS COMPLETED!")
    print("="*70)

    return {
        'regression_results': reg_results
    }


if __name__ == "__main__":
    results = main()

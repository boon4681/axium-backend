"""
Example: Using Updated Preprocessing with Kwargs
Shows how to use the improved preprocessing methods with direct parameter passing
"""

from pipelines.tabular.regression.regression_02_preprocessing import RegressionPreprocessor
from pipelines.tabular.regression.regression_01_load_data import RegressionDataLoader
from pipelines.tabular.regression.regression_03_model_selection import RegressionModelSelector
from pipelines.tabular.regression.regression_04_model_training import RegressionModelTrainer
from pipelines.tabular.regression.regression_05_evaluation import RegressionEvaluator
from pipelines.tabular.regression.regression_06_deployment import RegressionModelDeployment
from pipelines.tabular.regression.regression_07_visualization import RegressionVisualizer
import pandas as pd
import sys
import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))


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


def demo_regression_pipeline():
    """Demo end-to-end regression pipeline with preprocessing, model selection, training, and evaluation"""
    print("\n" + "="*70)
    print("REGRESSION PIPELINE DEMO")
    print("="*70)

    # Load data
    loader = RegressionDataLoader()
    X, y = loader.load_sample_dataset("california_housing")

    # Preprocessing
    preprocessor = RegressionPreprocessor()
    processed_data = preprocessor.full_preprocessing_pipeline(
        X, y,
        test_size=0.2,
        missing_value_strategy='mean',
        scaling_method='standard',
        remove_outliers=False,
        transform_target=False,
        select_features=False
    )

    X_train, X_test = processed_data['X_train'], processed_data['X_test']
    y_train, y_test = processed_data['y_train'], processed_data['y_test']

    # Model selection
    selector = RegressionModelSelector()
    selector.list_models()  # List available models
    model = selector.get_model('random_forest')

    # Model training
    trainer = RegressionModelTrainer(model)
    trained_model = trainer.train_model(X_train, y_train, param_grid={
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20]
    })

    # Save the trained model
    deployment = RegressionModelDeployment()
    deployment.save_model(trained_model, "regression_model.joblib")

    # Evaluation
    evaluator = RegressionEvaluator(trained_model)
    evaluation_results = evaluator.evaluate(
        X_test, y_test, metrics=['mse', 'mae', 'r2'])

    print(f"Evaluation results: {evaluation_results}")

    # Visualization
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    visualizer = RegressionVisualizer(trained_model)

    # Generate all visualizations
    visualization_metrics = visualizer.plot_all_metrics(
        X_train, y_train, X_test, y_test,
        feature_names=list(X_train.columns),
        save_dir='regression_visualizations',
        show_plots=False
    )

    print(f"Visualization metrics: {visualization_metrics}")

    return evaluation_results


def main():
    """Main demo function"""

    # Demo regression pipeline with visualization
    pipeline_results = demo_regression_pipeline()

    return {
        'pipeline_results': pipeline_results
    }


if __name__ == "__main__":
    results = main()

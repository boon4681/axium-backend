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
        feature_names=list(X_train.columns)
    )

    print(f"Visualization metrics: {visualization_metrics}")

    return evaluation_results


def demo_regression_visualization():
    """Demo individual visualization capabilities for regression"""
    print("\n" + "="*70)
    print("REGRESSION VISUALIZATION DEMO")
    print("="*70)

    # Load and prepare data
    loader = RegressionDataLoader()
    X, y = loader.load_sample_dataset("california_housing")

    # Preprocessing
    preprocessor = RegressionPreprocessor()
    processed_data = preprocessor.full_preprocessing_pipeline(
        X, y,
        test_size=0.3,
        missing_value_strategy='mean',
        scaling_method='standard',
        remove_outliers=False,
        select_features=False
    )

    X_train, X_test = processed_data['X_train'], processed_data['X_test']
    y_train, y_test = processed_data['y_train'], processed_data['y_test']

    # Train a Random Forest model
    selector = RegressionModelSelector()
    model = selector.get_model('random_forest')
    trainer = RegressionModelTrainer(model)
    trained_model = trainer.train_model(X_train, y_train)

    # Create visualizer
    visualizer = RegressionVisualizer(trained_model)

    # Individual visualization demos
    print("\n1. Predictions vs Actual")
    print("-" * 40)
    visualizer.plot_predictions_vs_actual(X_test, y_test)

    print("\n2. Residual Analysis")
    print("-" * 40)
    visualizer.plot_residuals(X_test, y_test)

    print("\n3. Learning Curve")
    print("-" * 40)
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    visualizer.plot_learning_curve(X_full, y_full)

    print("\n4. Feature Importance")
    print("-" * 40)
    visualizer.plot_feature_importance(list(X_train.columns))

    print("\n5. Error Distribution")
    print("-" * 40)
    error_metrics = visualizer.plot_error_distribution(X_test, y_test)
    print(f"Error metrics: {error_metrics}")

    print("\n6. Prediction Intervals")
    print("-" * 40)
    visualizer.plot_prediction_intervals(X_test, y_test)

    return trained_model


def main():
    """Main demo function"""

    # Demo regression pipeline with visualization
    pipeline_results = demo_regression_pipeline()

    # Demo individual visualization capabilities
    print("\n" + "="*70)
    print("RUNNING INDIVIDUAL VISUALIZATION DEMOS")
    print("="*70)
    reg_visualization_model = demo_regression_visualization()

    return {
        'pipeline_results': pipeline_results,
        'visualization_model': reg_visualization_model
    }


if __name__ == "__main__":
    results = main()

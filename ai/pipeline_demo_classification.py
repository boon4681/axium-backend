"""
Example: Using Updated Preprocessing with Kwargs
Shows how to use the improved preprocessing methods with direct parameter passing
"""

from pipelines.tabular.regression.regression_02_preprocessing import RegressionPreprocessor
from pipelines.tabular.regression.regression_01_load_data import RegressionDataLoader
from pipelines.tabular.classification.classification_02_preprocessing import ClassificationPreprocessor
from pipelines.tabular.classification.classification_01_load_data import ClassificationDataLoader
from pipelines.tabular.classification.classification_03_model_selection import ClassificationModelSelector
from pipelines.tabular.classification.classification_04_model_training import ClassificationModelTrainer
from pipelines.tabular.classification.classification_05_evaluation import ClassificationEvaluator
from pipelines.tabular.classification.classification_06_deployment import ClassificationModelDeployment
from pipelines.tabular.classification.classification_07_visualization import ClassificationVisualizer
import pandas as pd
import sys
import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def demo_classification_with_kwargs():
    """Demo classification preprocessing with custom parameters"""
    print("="*70)
    print("CLASSIFICATION PREPROCESSING WITH KWARGS")
    print("="*70)

    # Load data
    loader = ClassificationDataLoader()
    X, y = loader.load_sample_dataset("breast_cancer")

    # Create preprocessor
    preprocessor = ClassificationPreprocessor()

    # Example 1: Standard preprocessing
    print("\n1. STANDARD PREPROCESSING")
    print("-" * 40)
    result1 = preprocessor.full_preprocessing_pipeline(
        X, y,
        test_size=0.2,
        missing_value_strategy='mean',
        scaling_method='standard',
        select_features=True,
        feature_selection_method='f_classif',
        feature_selection_k=10
    )
    print(f"Result shape: {result1['X_train'].shape}")
    print(f"Config: {result1['preprocessing_config']}")

    # Example 2: Robust preprocessing
    print("\n2. ROBUST PREPROCESSING")
    print("-" * 40)
    result2 = preprocessor.full_preprocessing_pipeline(
        X, y,
        test_size=0.25,
        missing_value_strategy='median',
        scaling_method='robust',
        select_features=True,
        feature_selection_method='chi2',
        feature_selection_k=15
    )
    print(f"Result shape: {result2['X_train'].shape}")
    print(f"Config: {result2['preprocessing_config']}")

    # Example 3: Minimal preprocessing
    print("\n3. MINIMAL PREPROCESSING")
    print("-" * 40)
    result3 = preprocessor.full_preprocessing_pipeline(
        X, y,
        test_size=0.3,
        handle_missing=True,
        encode_categorical=False,
        scale_features=False,
        select_features=False
    )
    print(f"Result shape: {result3['X_train'].shape}")
    print(f"Config: {result3['preprocessing_config']}")

    return result1, result2, result3


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


def demo_classification_pipeline():
    """Demo end-to-end classification pipeline"""
    print("\n" + "="*70)
    print("CLASSIFICATION PIPELINE DEMO")
    print("="*70)

    # Load data
    loader = ClassificationDataLoader()
    X, y = loader.load_sample_dataset("breast_cancer")

    # Preprocessing
    preprocessor = ClassificationPreprocessor()
    processed_data = preprocessor.full_preprocessing_pipeline(
        X, y,
        test_size=0.2,
        missing_value_strategy='mean',
        scaling_method='standard',
        select_features=True,
        feature_selection_method='f_classif',
        feature_selection_k=10
    )

    X_train, X_test = processed_data['X_train'], processed_data['X_test']
    y_train, y_test = processed_data['y_train'], processed_data['y_test']

    # Model selection
    selector = ClassificationModelSelector()
    selector.list_models()  # List available models
    model = selector.get_model('random_forest')

    # Model training
    trainer = ClassificationModelTrainer(model)
    trained_model = trainer.train_model(X_train, y_train, param_grid={
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20]
    })

    # Save the trained model
    deployment = ClassificationModelDeployment()
    deployment.save_model(trained_model, "classification_model.joblib")

    # Evaluation
    evaluator = ClassificationEvaluator(trained_model)
    evaluation_results = evaluator.evaluate(
        X_test, y_test, metrics=['accuracy', 'precision', 'recall', 'f1'])

    print(f"Evaluation results: {evaluation_results}")

    # Visualization
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    visualizer = ClassificationVisualizer(trained_model)

    # Generate all visualizations
    visualizer.plot_all_metrics(
        X_train, y_train, X_test, y_test,
        feature_names=list(X_train.columns),
        class_names=['Malignant', 'Benign']
    )

    return evaluation_results


def demo_classification_visualization():
    """Demo individual visualization capabilities for classification"""
    print("\n" + "="*70)
    print("CLASSIFICATION VISUALIZATION DEMO")
    print("="*70)

    # Load and prepare data
    loader = ClassificationDataLoader()
    X, y = loader.load_sample_dataset("iris")  # Using iris for multiclass demo

    # Preprocessing
    preprocessor = ClassificationPreprocessor()
    processed_data = preprocessor.full_preprocessing_pipeline(
        X, y,
        test_size=0.3,
        missing_value_strategy='mean',
        scaling_method='standard',
        select_features=False
    )

    X_train, X_test = processed_data['X_train'], processed_data['X_test']
    y_train, y_test = processed_data['y_train'], processed_data['y_test']

    # Train a Random Forest model
    selector = ClassificationModelSelector()
    model = selector.get_model('random_forest')
    trainer = ClassificationModelTrainer(model)
    trained_model = trainer.train_model(X_train, y_train)

    # Create visualizer
    visualizer = ClassificationVisualizer(trained_model)

    # Individual visualization demos
    print("\n1. Confusion Matrix")
    print("-" * 40)
    visualizer.plot_confusion_matrix(X_test, y_test,
                                     class_names=['Setosa', 'Versicolor', 'Virginica'])

    print("\n2. ROC Curve")
    print("-" * 40)
    visualizer.plot_roc_curve(X_test, y_test)

    print("\n3. Learning Curve")
    print("-" * 40)
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    visualizer.plot_learning_curve(X_full, y_full)

    print("\n4. Feature Importance")
    print("-" * 40)
    visualizer.plot_feature_importance(list(X_train.columns))

    print("\n5. Classification Report")
    print("-" * 40)
    visualizer.plot_classification_report(X_test, y_test,
                                          class_names=['Setosa', 'Versicolor', 'Virginica'])

    return trained_model


def main():
    """Main demo function"""

    # Demo classification pipeline with visualization
    clf_pipeline_results = demo_classification_pipeline()

    # Demo individual visualization capabilities
    print("\n" + "="*70)
    print("RUNNING INDIVIDUAL VISUALIZATION DEMOS")
    print("="*70)
    clf_visualization_model = demo_classification_visualization()

    return {
        'classification_pipeline_results': clf_pipeline_results,
        'visualization_model': clf_visualization_model
    }


if __name__ == "__main__":
    results = main()

from nodes.classification import *
from nodes.regression import *
import pandas as pd
import numpy as np

print("=== TESTING AUTO SAVE FUNCTIONALITY ===\n")

# Load sample data
print("Loading sample datasets...")
classification_data = ClassificationLoadSampleDataset.run({'dataset_name': 'iris'}, {})
regression_data = RegressionLoadSampleDataset.run({'dataset_name': 'synthetic'}, {})

# Prepare data
clf_df = classification_data['features'].copy()
clf_df['target'] = classification_data['target']

reg_df = regression_data['features'].copy()
reg_df['target'] = regression_data['target']

# Train models
print("Training models...")
clf_model = ClassificationGetModel.run({'model_name': 'logistic_regression'}, {})['model']
trained_clf_model = ClassificationTrainModel.run({'target': 'target'}, {'data': clf_df, 'model': clf_model})['model']

reg_model = RegressionGetModel.run({'model_name': 'linear_regression'}, {})['model']
trained_reg_model = RegressionTrainModel.run({'target': 'target'}, {'data': reg_df, 'model': reg_model})['model']

print("\n=== Testing Auto Save (saves to organized directories) ===")

# Test auto_save=True for classification plots
print("1. Classification plots with auto_save=True:")
result = ClassificationPlotConfusionMatrix.run(
    {'target': 'target', 'show_plot': False, 'auto_save': True}, 
    {'data': clf_df, 'model': trained_clf_model}
)
print(f"   Confusion matrix saved to: plots/classification/confusion_matrix/confusion_matrix.png")

result = ClassificationPlotROCCurve.run(
    {'target': 'target', 'show_plot': False, 'auto_save': True}, 
    {'data': clf_df, 'model': trained_clf_model}
)
print(f"   ROC curve saved to: plots/classification/roc_curves/roc_curve.png")

# Test auto_save=True for regression plots
print("\n2. Regression plots with auto_save=True:")
result = RegressionPlotPredictionsVsActual.run(
    {'target': 'target', 'show_plot': False, 'auto_save': True}, 
    {'data': reg_df, 'model': trained_reg_model}
)
print(f"   Predictions vs actual saved to: plots/regression/predictions/predictions_vs_actual.png")

result = RegressionPlotResiduals.run(
    {'target': 'target', 'show_plot': False, 'auto_save': True}, 
    {'data': reg_df, 'model': trained_reg_model}
)
print(f"   Residuals plot saved to: plots/regression/residuals/residuals.png")

result = RegressionPlotFeatureImportance.run(
    {'target': 'target', 'show_plot': False, 'auto_save': True}, 
    {'data': reg_df, 'model': trained_reg_model}
)
print(f"   Feature importance saved to: plots/regression/feature_importance/feature_importance.png")

print("\n=== Testing Custom Save Paths ===")

# Test custom filename (still organized into directories)
result = ClassificationPlotConfusionMatrix.run(
    {'target': 'target', 'show_plot': False, 'save_path': 'iris_confusion_matrix.png'}, 
    {'data': clf_df, 'model': trained_clf_model}
)
print(f"   Custom confusion matrix saved to: plots/classification/confusion_matrix/iris_confusion_matrix.png")

# Test full custom path (respects the full path)
result = RegressionPlotPredictionsVsActual.run(
    {'target': 'target', 'show_plot': False, 'save_path': 'custom_plots/my_predictions.png'}, 
    {'data': reg_df, 'model': trained_reg_model}
)
print(f"   Custom path predictions saved to: custom_plots/my_predictions.png")

print("\n=== Auto Save Features ===")
print("• auto_save=True: Automatically saves with default filename to organized directories")
print("• save_path='filename.png': Saves to organized directory with custom filename")
print("• save_path='full/path/file.png': Saves to the exact path specified")
print("• No parameters: Only displays the plot (doesn't save)")

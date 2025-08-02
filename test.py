from nodes.classification import *
from nodes.regression import *
import pandas as pd
import numpy as np

print("=== TESTING ALL NODES ===\n")

# Test 1: Load Sample Datasets
print("1. Loading Sample Datasets...")
classification_data = ClassificationLoadSampleDataset.run({'dataset_name': 'iris'}, {})
regression_data = RegressionLoadSampleDataset.run({'dataset_name': 'synthetic'}, {})
print(f"Classification dataset loaded: {type(classification_data.get('features'))}")
print(f"Regression dataset loaded: {type(regression_data.get('features'))}")

# Combine features and target into single dataframe for easier testing
if classification_data.get('features') is not None and classification_data.get('target') is not None:
    clf_df = classification_data['features'].copy()
    clf_df['target'] = classification_data['target']
else:
    clf_df = None

if regression_data.get('features') is not None and regression_data.get('target') is not None:
    reg_df = regression_data['features'].copy()
    reg_df['target'] = regression_data['target']
else:
    reg_df = None

print("\n2. Testing Classification Nodes...")

# Test ClassificationAnalyzeData
print("2.1 ClassificationAnalyzeData:")
result = ClassificationAnalyzeData.run({'target': 'target'}, {'data': clf_df})
print(f"   Result keys: {list(result.keys()) if result else 'None'}")

# Test ClassificationGetDataInfo
print("2.2 ClassificationGetDataInfo:")
result = ClassificationGetDataInfo.run({'target': 'target'}, {'data': clf_df})
print(f"   Result keys: {list(result.keys()) if result else 'None'}")

# Test ClassificationGetModel
print("2.3 ClassificationGetModel:")
model_result = ClassificationGetModel.run({'model_name': 'logistic_regression'}, {})
model = model_result.get('model')
print(f"   Model type: {type(model)}")

# Test ClassificationTrainModel
print("2.4 ClassificationTrainModel:")
if model and clf_df is not None:
    trained_model_result = ClassificationTrainModel.run({'target': 'target'}, {'data': clf_df, 'model': model})
    trained_model = trained_model_result.get('model')
    print(f"   Trained model type: {type(trained_model)}")
else:
    trained_model = None
    print("   Skipped - missing model or data")

# Test ClassificationEvaluateModel
print("2.5 ClassificationEvaluateModel:")
if trained_model and clf_df is not None:
    eval_result = ClassificationEvaluateModel.run({'target': 'target'}, {'data': clf_df, 'model': trained_model})
    print(f"   Evaluation metrics: {eval_result.get('metrics')}")
else:
    print("   Skipped - missing trained model or data")
    
# Test ClassificationSaveModel
print("2.6 ClassificationSaveModel:")
if trained_model:
    result = ClassificationSaveModel.run({'file': 'test_model.pkl'}, {'model': trained_model})
    print(f"   Save result: {result}")
else:
    print("   Skipped - no trained model")

# # Test ClassificationLoadCSVData
# print("2.7 ClassificationLoadCSVData:")
# result = ClassificationLoadCSVData.run({'file': 'xd.csv'}, {})
# print(f"   Result: {result}")

# Test ClassificationLoadModel
print("2.8 ClassificationLoadModel:")
result = ClassificationLoadModel.run({'file': 'test_model.pkl'}, {})
print(f"   Result: {result}")

# Test ClassificationPlotConfusionMatrix
print("2.9 ClassificationPlotConfusionMatrix:")
if trained_model and clf_df is not None:
    result = ClassificationPlotConfusionMatrix.run({'target': 'target', 'show_plot': False}, 
                                                   {'data': clf_df, 'model': trained_model})
    print(f"   Plot result: {type(result.get('figure'))}")
else:
    print("   Skipped - missing trained model or data")

# Test ClassificationPlotROCCurve
print("2.10 ClassificationPlotROCCurve:")
if trained_model and clf_df is not None:
    result = ClassificationPlotROCCurve.run({'target': 'target', 'show_plot': False}, 
                                           {'data': clf_df, 'model': trained_model})
    print(f"   ROC result: {result.get('figure')}")
else:
    print("   Skipped - missing trained model or data")

# Test ClassificationCreateTimeSeriesFeatures
print("2.11 ClassificationCreateTimeSeriesFeatures:")
# Create sample data with date column
if clf_df is not None:
    ts_df = clf_df.copy()
    ts_df['date'] = pd.date_range('2021-01-01', periods=len(ts_df))
    result = ClassificationCreateTimeSeriesFeatures.run({'date_column': 'date', 'lag_features': 2}, 
                                                        {'data': ts_df})
    print(f"   Time series features shape: {result.get('features').shape if result.get('features') is not None else 'None'}")
else:
    print("   Skipped - no data")

print("\n3. Testing Regression Nodes...")

# Test RegressionAnalyzeData
print("3.1 RegressionAnalyzeData:")
result = RegressionAnalyzeData.run({'target': 'target'}, {'data': reg_df})
print(f"   Result keys: {list(result.keys()) if result else 'None'}")

# Test RegressionGetDataInfo
print("3.2 RegressionGetDataInfo:")
result = RegressionGetDataInfo.run({'target': 'target'}, {'data': reg_df})
print(f"   Result keys: {list(result.keys()) if result else 'None'}")

# Test RegressionGetModel
print("3.3 RegressionGetModel:")
reg_model_result = RegressionGetModel.run({'model_name': 'linear_regression'}, {})
reg_model = reg_model_result.get('model')
print(f"   Model type: {type(reg_model)}")

# Test RegressionTrainModel
print("3.4 RegressionTrainModel:")
if reg_model and reg_df is not None:
    trained_reg_model_result = RegressionTrainModel.run({'target': 'target'}, {'data': reg_df, 'model': reg_model})
    trained_reg_model = trained_reg_model_result.get('model')
    print(f"   Trained model type: {type(trained_reg_model)}")
else:
    trained_reg_model = None
    print("   Skipped - missing model or data")

# Test RegressionEvaluateModel
print("3.5 RegressionEvaluateModel:")
if trained_reg_model and reg_df is not None:
    eval_result = RegressionEvaluateModel.run({'target': 'target'}, {'data': reg_df, 'model': trained_reg_model})
    print(f"   Evaluation metrics: {eval_result.get('metrics')}")
else:
    print("   Skipped - missing trained model or data")

# # Test RegressionLoadCSVData
# print("3.6 RegressionLoadCSVData:")
# result = RegressionLoadCSVData.run({'file': 'xd.csv'}, {})
# print(f"   Result: {result}")

# Test RegressionSaveModel
print("3.7 RegressionSaveModel:")
if trained_reg_model:
    result = RegressionSaveModel.run({'file': 'test_reg_model.pkl'}, {'model': trained_reg_model})
    print(f"   Save result: {result}")
else:
    print("   Skipped - no trained model")

# Test RegressionLoadModel
print("3.8 RegressionLoadModel:")
result = RegressionLoadModel.run({'file': 'test_reg_model.pkl'}, {})
print(f"   Result: {result}")



# Test RegressionPlotErrorDistribution
print("3.9 RegressionPlotErrorDistribution:")
if trained_reg_model and reg_df is not None:
    result = RegressionPlotErrorDistribution.run({'target': 'target', 'show_plot': False}, 
                                                 {'data': reg_df, 'model': trained_reg_model})
    print(f"   Error plot result: {type(result.get('figure'))}")
else:
    print("   Skipped - missing trained model or data")

# Test RegressionPlotFeatureImportance
print("3.10 RegressionPlotFeatureImportance:")
if trained_reg_model and reg_df is not None:
    result = RegressionPlotFeatureImportance.run({'target': 'target', 'show_plot': False}, 
                                                 {'data': reg_df, 'model': trained_reg_model})
    print(f"   Feature importance result: {type(result.get('figure'))}")
else:
    print("   Skipped - missing trained model or data")

# Test RegressionPlotLearningCurve
print("3.11 RegressionPlotLearningCurve:")
if trained_reg_model and reg_df is not None:
    result = RegressionPlotLearningCurve.run({'target': 'target', 'cv': 3, 'show_plot': False}, 
                                           {'data': reg_df, 'model': trained_reg_model})
    print(f"   Learning curve result: {type(result.get('figure'))}")
else:
    print("   Skipped - missing trained model or data")

# Test RegressionPlotPredictionIntervals
print("3.12 RegressionPlotPredictionIntervals:")
if trained_reg_model and reg_df is not None:
    result = RegressionPlotPredictionIntervals.run({'target': 'target', 'show_plot': False}, 
                                                   {'data': reg_df, 'model': trained_reg_model})
    print(f"   Prediction intervals result: {type(result.get('figure'))}")
else:
    print("   Skipped - missing trained model or data")

# Test RegressionPlotPredictionsVsActual
print("3.13 RegressionPlotPredictionsVsActual:")
if trained_reg_model and reg_df is not None:
    result = RegressionPlotPredictionsVsActual.run({'target': 'target', 'show_plot': False}, 
                                                   {'data': reg_df, 'model': trained_reg_model})
    print(f"   Predictions vs actual result: {result.get('figure')}")
else:
    print("   Skipped - missing trained model or data")

# Test RegressionPlotResiduals
print("3.14 RegressionPlotResiduals:")
if trained_reg_model and reg_df is not None:
    result = RegressionPlotResiduals.run({'target': 'target', 'show_plot': False}, 
                                        {'data': reg_df, 'model': trained_reg_model})
    print(f"   Residuals plot result: {type(result.get('figure'))}")
else:
    print("   Skipped - missing trained model or data")

# Test RegressionCreateTimeSeriesFeatures
print("3.15 RegressionCreateTimeSeriesFeatures:")
if reg_df is not None:
    ts_reg_df = reg_df.copy()
    ts_reg_df['date'] = pd.date_range('2021-01-01', periods=len(ts_reg_df))
    result = RegressionCreateTimeSeriesFeatures.run({'date_column': 'date', 'lag_features': 2}, 
                                                    {'data': ts_reg_df})
    print(f"   Time series features shape: {result.get('features').shape if result.get('features') is not None else 'None'}")
else:
    print("   Skipped - no data")

print("\n=== ALL TESTS COMPLETED ===")
print("Note: Some plotting functions may show import warnings for matplotlib/seaborn if not installed.")

print("\n=== TESTING AUTO SAVE FUNCTIONALITY ===")

# Test auto_save=True for classification plots
print("Classification plots with auto_save=True:")
if trained_model and clf_df is not None:
    result = ClassificationPlotConfusionMatrix.run(
        {'target': 'target', 'show_plot': False, 'auto_save': True}, 
        {'data': clf_df, 'model': trained_model}
    )
    print(f"   Confusion matrix auto-saved to: plots/classification/confusion_matrix/confusion_matrix.png")

    result = ClassificationPlotROCCurve.run(
        {'target': 'target', 'show_plot': False, 'auto_save': True}, 
        {'data': clf_df, 'model': trained_model}
    )
    print(f"   ROC curve auto-saved to: plots/classification/roc_curves/roc_curve.png")

# Test auto_save=True for regression plots
print("\nRegression plots with auto_save=True:")
if trained_reg_model and reg_df is not None:
    result = RegressionPlotPredictionsVsActual.run(
        {'target': 'target', 'show_plot': False, 'auto_save': True}, 
        {'data': reg_df, 'model': trained_reg_model}
    )
    print(f"   Predictions vs actual auto-saved to: plots/regression/predictions/predictions_vs_actual.png")

    result = RegressionPlotResiduals.run(
        {'target': 'target', 'show_plot': False, 'auto_save': True}, 
        {'data': reg_df, 'model': trained_reg_model}
    )
    print(f"   Residuals plot auto-saved to: plots/regression/residuals/residuals.png")

    result = RegressionPlotFeatureImportance.run(
        {'target': 'target', 'show_plot': False, 'auto_save': True}, 
        {'data': reg_df, 'model': trained_reg_model}
    )
    print(f"   Feature importance auto-saved to: plots/regression/feature_importance/feature_importance.png")

print("\nAuto Save Features:")
print("• auto_save=True: Automatically saves with default filename to organized directories")
print("• save_path='filename.png': Saves to organized directory with custom filename")
print("• save_path='full/path/file.png': Saves to the exact path specified")

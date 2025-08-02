from axium.node_typing import AxiumNode

class RegressionPlotPredictionIntervals(AxiumNode):
    id = "regression.plot-prediction-intervals"
    category = "regression"
    name = "Plot Prediction Intervals"

    inputs = {
        "model": ("sklearn.model", {}),
        "data": ("pandas.df", {})
    }
    outputs = {
        "figure": ("matplotlib.figure", {})
    }
    parameters = None

    @classmethod
    def validate_inputs(cls, inputs: dict):
        model = inputs.get("model")
        data = inputs.get("data")
        if model is None:
            return {"error": "Model is required"}
        if data is None:
            return {"error": "Data is required"}
        return {}

    @classmethod
    def run(cls, parameters: dict, inputs: dict):
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        model = inputs.get("model")
        data = inputs.get("data")
        target_col = parameters.get("target") if parameters else None
        save_path = parameters.get("save_path") if parameters else None
        show_plot = parameters.get("show_plot", True) if parameters else True
        auto_save = parameters.get("auto_save", False) if parameters else False
        
        if model is None or data is None or target_col is None or target_col not in data.columns:
            return {"figure": None}
        
        # Set default save path if auto_save is enabled or save_path is provided
        if auto_save and not save_path:
            save_path = "prediction_intervals.png"  # Save to current directory
        
        # Organize save path into plots/regression/prediction_intervals/ if it's just a filename
        if save_path:
            if not os.path.dirname(save_path):  # If save_path is just a filename
                plots_dir = os.path.join("plots", "regression", "prediction_intervals")
                os.makedirs(plots_dir, exist_ok=True)
                save_path = os.path.join(plots_dir, save_path)
        
        features = data.drop(columns=[target_col])
        target = data[target_col]
        y_pred = model.predict(features)
        sorted_idx = np.argsort(y_pred)
        target_sorted = target.iloc[sorted_idx]
        y_pred_sorted = y_pred[sorted_idx]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(target_sorted)), target_sorted, alpha=0.6, label='Actual', color='blue')
        plt.plot(range(len(y_pred_sorted)), y_pred_sorted, color='red', label='Predicted', linewidth=2)
        residuals = target_sorted - y_pred_sorted
        std_residuals = np.std(residuals)
        plt.fill_between(range(len(y_pred_sorted)),
                         y_pred_sorted - 1.96 * std_residuals,
                         y_pred_sorted + 1.96 * std_residuals,
                         alpha=0.2, color='red', label='95% Prediction Interval')
        plt.xlabel('Sample Index (sorted by prediction)')
        plt.ylabel('Target Value')
        plt.title('Predictions with Confidence Intervals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return {"figure": plt.gcf()}

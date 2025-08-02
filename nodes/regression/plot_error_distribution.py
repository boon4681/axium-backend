from axium.node_typing import AxiumNode

class RegressionPlotErrorDistribution(AxiumNode):
    id = "regression.plot-error-distribution"
    category = "regression"
    name = "Plot Error Distribution"

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
        from sklearn.metrics import mean_squared_error, mean_absolute_error
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
            save_path = "error_distribution.png"  # Save to current directory
        
        # Organize save path into plots/regression/error_distribution/ if it's just a filename
        if save_path:
            if not os.path.dirname(save_path):  # If save_path is just a filename
                plots_dir = os.path.join("plots", "regression", "error_distribution")
                os.makedirs(plots_dir, exist_ok=True)
                save_path = os.path.join(plots_dir, save_path)
        
        features = data.drop(columns=[target_col])
        target = data[target_col]
        y_pred = model.predict(features)
        
        mse = mean_squared_error(target, y_pred)
        mae = mean_absolute_error(target, y_pred)
        rmse = np.sqrt(mse)
        absolute_errors = np.abs(target - y_pred)
        percentage_errors = (absolute_errors / np.abs(target)) * 100
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].hist(absolute_errors, bins=30, alpha=0.7, color='lightcoral')
        axes[0].set_xlabel('Absolute Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Absolute Error Distribution\nMAE = {mae:.3f}')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(percentage_errors, bins=30, alpha=0.7, color='lightgreen')
        axes[1].set_xlabel('Percentage Error (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Percentage Error Distribution')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].scatter(y_pred, absolute_errors, alpha=0.6, color='orange')
        axes[2].set_xlabel('Predicted Values')
        axes[2].set_ylabel('Absolute Error')
        axes[2].set_title('Error vs Predicted Values')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        
        figure = plt.gcf()
        if not show_plot:
            plt.close()
        
        return {"figure": figure}

from axium.template import AxiumTemplate
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class RegressionPlotErrorDistribution(AxiumTemplate):
    name = "RegressionPlotErrorDistribution"
    id = "RegressionPlotErrorDistribution"
    category = "regression"

    input = {
        "model": "axium.model",
        "features": "axium.dataframe",
        "target": "axium.series",
        "save_path": "axium.str",
        "show_plot": "axium.bool"
    }
    output = {
        "metrics": "axium.dict"
    }

    @classmethod
    def run(cls, model, features, target, save_path=None, show_plot=True):
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
        else:
            plt.close()
        return {"metrics": {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mean_percentage_error': np.mean(percentage_errors)
        }}

from axium.template import AxiumTemplate
import matplotlib.pyplot as plt
import numpy as np

class RegressionPlotPredictionIntervals(AxiumTemplate):
    name = "RegressionPlotPredictionIntervals"
    id = "RegressionPlotPredictionIntervals"
    category = "regression"

    input = {
        "model": "axium.model",
        "features": "axium.dataframe",
        "target": "axium.series",
        "save_path": "axium.str",
        "show_plot": "axium.bool"
    }
    output = {}

    @classmethod
    def run(cls, model, features, target, save_path=None, show_plot=True):
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
        return {}

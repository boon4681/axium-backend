from axium.template import AxiumTemplate
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class RegressionPlotPredictionsVsActual(AxiumTemplate):
    name = "RegressionPlotPredictionsVsActual"
    id = "RegressionPlotPredictionsVsActual"
    category = "regression"

    input = {
        "model": "axium.model",
        "features": "axium.dataframe",
        "target": "axium.series",
        "save_path": "axium.str",
        "show_plot": "axium.bool"
    }
    output = {
        "r2": "axium.float"
    }

    @classmethod
    def run(cls, model, features, target, save_path=None, show_plot=True):
        y_pred = model.predict(features)
        plt.figure(figsize=(8, 6))
        plt.scatter(target, y_pred, alpha=0.6)
        min_val = min(target.min(), y_pred.min())
        max_val = max(target.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual Values')
        plt.grid(True, alpha=0.3)
        r2 = r2_score(target, y_pred)
        plt.text(0.05, 0.95, f'R2 = {r2:.3f}', transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        return {"r2": float(r2)}

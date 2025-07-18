from axium.template import AxiumTemplate
import matplotlib.pyplot as plt
from scipy import stats

class RegressionPlotResiduals(AxiumTemplate):
    name = "RegressionPlotResiduals"
    id = "RegressionPlotResiduals"
    category = "regression"

    input = {
        "model": "axium.model",
        "features": "axium.dataframe",
        "target": "axium.series",
        "save_path": "axium.str",
        "show_plot": "axium.bool"
    }
    output = {
        "residuals": "axium.series"
    }

    @classmethod
    def run(cls, model, features, target, save_path=None, show_plot=True):
        y_pred = model.predict(features)
        residuals = target - y_pred
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        axes[1].hist(residuals, bins=30, alpha=0.7, color='skyblue')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residuals Distribution')
        axes[1].grid(True, alpha=0.3)
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot')
        axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        return {"residuals": residuals}

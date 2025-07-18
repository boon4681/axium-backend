from axium.template import AxiumTemplate
import matplotlib.pyplot as plt
import numpy as np

class RegressionPlotFeatureImportance(AxiumTemplate):
    name = "RegressionPlotFeatureImportance"
    id = "RegressionPlotFeatureImportance"
    category = "regression"

    input = {
        "model": "axium.model",
        "feature_names": "axium.list",
        "save_path": "axium.str",
        "show_plot": "axium.bool"
    }
    output = {}

    @classmethod
    def run(cls, model, feature_names, save_path=None, show_plot=True):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.title('Feature Importance')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if show_plot:
                plt.show()
            else:
                plt.close()
        return {}

from axium.template import AxiumTemplate
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

class ClassificationPlotROCCurve(AxiumTemplate):
    name = "ClassificationPlotROCCurve"
    id = "ClassificationPlotROCCurve"
    category = "classification"

    input = {
        "model": "axium.model",
        "features": "axium.dataframe",
        "target": "axium.series",
        "save_path": "axium.str",
        "show_plot": "axium.bool"
    }
    output = {
        "auc": "axium.float"
    }

    @classmethod
    def run(cls, model, features, target, save_path=None, show_plot=True):
        plt.figure(figsize=(8, 6))
        unique_classes = np.unique(target)
        if len(unique_classes) == 2:
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(features)[:, 1]
            else:
                y_score = model.decision_function(features)
            fpr, tpr, _ = roc_curve(target, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if show_plot:
                plt.show()
            else:
                plt.close()
            return {"auc": float(roc_auc)}
        else:
            return {"auc": None}

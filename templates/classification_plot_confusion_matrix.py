from axium.template import AxiumTemplate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class ClassificationPlotConfusionMatrix(AxiumTemplate):
    name = "ClassificationPlotConfusionMatrix"
    id = "ClassificationPlotConfusionMatrix"
    category = "classification"

    input = {
        "model": "axium.model",
        "features": "axium.dataframe",
        "target": "axium.series",
        "class_names": "axium.list",
        "save_path": "axium.str",
        "show_plot": "axium.bool"
    }
    output = {
        "confusion_matrix": "axium.dataframe"
    }

    @classmethod
    def run(cls, model, features, target, class_names=None, save_path=None, show_plot=True):
        y_pred = model.predict(features)
        cm = confusion_matrix(target, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        return {"confusion_matrix": cm}

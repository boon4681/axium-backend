from axium.template import AxiumTemplate
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

class RegressionPlotLearningCurve(AxiumTemplate):
    name = "RegressionPlotLearningCurve"
    id = "RegressionPlotLearningCurve"
    category = "regression"

    input = {
        "model": "axium.model",
        "features": "axium.dataframe",
        "target": "axium.series",
        "cv": "axium.int",
        "save_path": "axium.str",
        "show_plot": "axium.bool"
    }
    output = {}

    @classmethod
    def run(cls, model, features, target, cv=5, save_path=None, show_plot=True):
        train_sizes, train_scores, val_scores = learning_curve(
            model, features, target, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error')
        train_scores = -train_scores
        val_scores = -val_scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training MSE')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation MSE')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        plt.xlabel('Training Set Size')
        plt.ylabel('Mean Squared Error')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        return {}

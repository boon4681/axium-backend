from axium.node_typing import AxiumNode

class RegressionPlotFeatureImportance(AxiumNode):
    id = "regression.plot-feature-importance"
    category = "regression"
    name = "Plot Feature Importance"

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
        
        if model is None or data is None:
            return {"figure": None}
        
        # Set default save path if auto_save is enabled or save_path is provided
        if auto_save and not save_path:
            save_path = "feature_importance.png"  # Save to current directory
        
        # Organize save path into plots/regression/feature_importance/ if it's just a filename
        if save_path:
            if not os.path.dirname(save_path):  # If save_path is just a filename
                plots_dir = os.path.join("plots", "regression", "feature_importance")
                os.makedirs(plots_dir, exist_ok=True)
                save_path = os.path.join(plots_dir, save_path)
        
        features = data.drop(columns=[target_col]) if target_col and target_col in data.columns else data
        feature_names = list(features.columns)
        
        plt.figure(figsize=(10, 6))
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models have feature importances
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.title('Feature Importance')
            plt.ylabel('Importance')
        elif hasattr(model, 'coef_'):
            # Linear models have coefficients
            coefs = np.abs(model.coef_)  # Use absolute values for importance
            indices = np.argsort(coefs)[::-1]
            plt.bar(range(len(coefs)), coefs[indices])
            plt.xticks(range(len(coefs)), [feature_names[i] for i in indices], rotation=45)
            plt.title('Feature Coefficients (Absolute Values)')
            plt.ylabel('Coefficient Magnitude')
        else:
            # Model doesn't have interpretable feature importance
            plt.text(0.5, 0.5, 'Model does not support feature importance', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Importance Not Available')
        
        plt.xlabel('Features')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        
        fig = plt.gcf()
        if not show_plot:
            plt.close()
        return {"figure": fig}

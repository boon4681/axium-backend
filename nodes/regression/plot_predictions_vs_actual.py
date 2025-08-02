from axium.node_typing import AxiumNode

class RegressionPlotPredictionsVsActual(AxiumNode):
    id = "regression.plot-predictions-vs-actual"
    category = "regression"
    name = "Plot Predictions vs Actual"

    inputs = {
        "y_true": ("pandas.series", {}),
        "y_pred": ("pandas.series", {})
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
        from sklearn.metrics import r2_score
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
            save_path = "predictions_vs_actual.png"  # Save to current directory
        
        # Organize save path into plots/regression/predictions/ if it's just a filename
        if save_path:
            if not os.path.dirname(save_path):  # If save_path is just a filename
                plots_dir = os.path.join("plots", "regression", "predictions")
                os.makedirs(plots_dir, exist_ok=True)
                save_path = os.path.join(plots_dir, save_path)
        
        features = data.drop(columns=[target_col])
        target = data[target_col]
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
        
        fig = plt.gcf()
        if not show_plot:
            plt.close()
        
        return {"figure": fig}

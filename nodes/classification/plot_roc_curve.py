from axium.node_typing import AxiumNode

class ClassificationPlotROCCurve(AxiumNode):
    id = "classification.plot-roc-curve"
    category = "classification"
    name = "Plot ROC Curve"

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
        from sklearn.metrics import roc_curve, auc
        import numpy as np
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
            save_path = "roc_curve.png"  # Save to current directory
        
        # Organize save path into plots/classification/roc_curves/ if it's just a filename
        if save_path:
            if not os.path.dirname(save_path):  # If save_path is just a filename
                plots_dir = os.path.join("plots", "classification", "roc_curves")
                os.makedirs(plots_dir, exist_ok=True)
                save_path = os.path.join(plots_dir, save_path)
        
        features = data.drop(columns=[target_col])
        target = data[target_col]
        
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
            return {"figure": float(roc_auc)}
        else:
            # For multiclass, create one-vs-rest ROC curves
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc
            from itertools import cycle
            
            y_test_bin = label_binarize(target, classes=unique_classes)
            n_classes = y_test_bin.shape[1]
            
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(features)
            else:
                # For models without predict_proba, use decision_function if available
                if hasattr(model, 'decision_function'):
                    y_score = model.decision_function(features)
                else:
                    return {"figure": "Model does not support probability prediction for multiclass ROC"}
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot all ROC curves
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'ROC curve of class {unique_classes[i]} (AUC = {roc_auc[i]:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Multi-class ROC Curves')
            plt.legend(loc="lower right")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return {"figure": dict(roc_auc)}

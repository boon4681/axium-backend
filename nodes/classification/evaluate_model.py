from axium.node_typing import AxiumNode

class ClassificationEvaluateModel(AxiumNode):
    id = "classification.evaluate-model"
    category = "classification"
    name = "Evaluate Model"

    inputs = {
        "model": ("sklearn.model", {}),
        "data": ("pandas.df", {})
    }
    outputs = {
        "metrics": ("dict", {})
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
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        model = inputs.get("model")
        data = inputs.get("data")
        target_col = parameters.get("target") if parameters else None
        metrics = parameters.get("metrics", ["accuracy", "precision", "recall", "f1"]) if parameters else ["accuracy", "precision", "recall", "f1"]
        if model is None or data is None or target_col is None or target_col not in data.columns:
            return {"metrics": None}
        features = data.drop(columns=[target_col])
        target = data[target_col]
        y_pred = model.predict(features)
        results = {}
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(target, y_pred)
        if 'precision' in metrics:
            results['precision'] = precision_score(target, y_pred, average='weighted')
        if 'recall' in metrics:
            results['recall'] = recall_score(target, y_pred, average='weighted')
        if 'f1' in metrics:
            results['f1'] = f1_score(target, y_pred, average='weighted')
        return {"metrics": results}

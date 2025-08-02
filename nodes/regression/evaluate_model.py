from axium.node_typing import AxiumNode

class RegressionEvaluateModel(AxiumNode):
    id = "regression.evaluate-model"
    category = "regression"
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
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        model = inputs.get("model")
        data = inputs.get("data")
        target_col = parameters.get("target") if parameters else None
        metrics = parameters.get("metrics", ["mse", "mae", "r2"]) if parameters else ["mse", "mae", "r2"]
        
        if model is None or data is None or target_col is None or target_col not in data.columns:
            return {"metrics": None}
        
        features = data.drop(columns=[target_col])
        target = data[target_col]
        y_pred = model.predict(features)
        results = {}
        
        if 'mse' in metrics:
            results['mse'] = mean_squared_error(target, y_pred)
        if 'mae' in metrics:
            results['mae'] = mean_absolute_error(target, y_pred)
        if 'r2' in metrics:
            results['r2'] = r2_score(target, y_pred)
        
        return {"metrics": results}

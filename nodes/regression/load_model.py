from axium.node_typing import AxiumNode

class RegressionLoadModel(AxiumNode):
    id = "regression.load-model"
    category = "regression"
    name = "Load Model"

    outputs = {
        "model": ("sklearn.model", {})
    }
    parameters = {
        "file": ("axium.file", {
            "file_type": "pkl",
            "placeholder": "Select model file .pkl",
            "label": "Model File",
            "inline": False
        })
    }

    @classmethod
    def validate_inputs(cls, inputs: dict):
        return {}

    @classmethod
    def run(cls, parameters: dict, inputs: dict):
        import joblib
        file_path = parameters.get("file") if parameters else None
        
        if not file_path:
            return {"model": None}
        
        model = joblib.load(file_path)
        return {"model": model}

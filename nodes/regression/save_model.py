from axium.node_typing import AxiumNode

class RegressionSaveModel(AxiumNode):
    id = "regression.save-model"
    category = "regression"
    name = "Save Model"

    inputs = {
        "model": ("sklearn.model", {})
    }
    outputs = {
        "file": ("axium.file", {})
    }
    parameters = {
        "file": ("axium.file", {
            "file_type": "pkl",
            "placeholder": "Save as .pkl",
            "label": "Save File",
            "inline": False
        })
    }

    @classmethod
    def validate_inputs(cls, inputs: dict):
        model = inputs.get("model")
        if model is None:
            return {"error": "Model is required"}
        return {}

    @classmethod
    def run(cls, parameters: dict, inputs: dict):
        import joblib
        model = inputs.get("model")
        file_path = parameters.get("file") if parameters else None
        
        if model is None or file_path is None:
            return {"file": None}
        
        joblib.dump(model, file_path)
        return {"file": file_path}

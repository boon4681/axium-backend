from axium.template import AxiumTemplate
import joblib

class RegressionSaveModel(AxiumTemplate):
    name = "RegressionSaveModel"
    id = "RegressionSaveModel"
    category = "regression"

    input = {
        "model": "axium.model",
        "file_path": "axium.str"
    }
    output = {
        "result": "axium.str"
    }

    @classmethod
    def run(cls, model, file_path):
        joblib.dump(model, file_path)
        return {"result": f"Model saved to {file_path}"}

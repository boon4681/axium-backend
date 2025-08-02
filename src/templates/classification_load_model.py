from axium.template import AxiumTemplate
import joblib

class ClassificationLoadModel(AxiumTemplate):
    name = "ClassificationLoadModel"
    id = "ClassificationLoadModel"
    category = "classification"

    input = {
        "file_path": "axium.str"
    }
    output = {
        "model": "axium.model"
    }

    @classmethod
    def run(cls, file_path):
        model = joblib.load(file_path)
        return {"model": model}

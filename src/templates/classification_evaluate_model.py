from axium.template import AxiumTemplate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ClassificationEvaluateModel(AxiumTemplate):
    name = "ClassificationEvaluateModel"
    id = "ClassificationEvaluateModel"
    category = "classification"

    input = {
        "model": "axium.model",
        "features": "axium.dataframe",
        "target": "axium.series",
        "metrics": "axium.list"
    }
    output = {
        "results": "axium.dict"
    }

    @classmethod
    def run(cls, model, features, target, metrics):
        y_pred = model.predict(features)
        results = {}
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(target, y_pred)
        if 'precision' in metrics:
            results['precision'] = precision_score(target, y_pred, average='binary')
        if 'recall' in metrics:
            results['recall'] = recall_score(target, y_pred, average='binary')
        if 'f1' in metrics:
            results['f1'] = f1_score(target, y_pred, average='binary')
        return {"results": results}

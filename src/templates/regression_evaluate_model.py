from axium.template import AxiumTemplate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class RegressionEvaluateModel(AxiumTemplate):
    name = "RegressionEvaluateModel"
    id = "RegressionEvaluateModel"
    category = "regression"

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
        if 'mse' in metrics:
            results['mse'] = mean_squared_error(target, y_pred)
        if 'mae' in metrics:
            results['mae'] = mean_absolute_error(target, y_pred)
        if 'r2' in metrics:
            results['r2'] = r2_score(target, y_pred)
        return {"results": results}

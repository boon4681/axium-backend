from axium.template import AxiumTemplate
from sklearn.model_selection import GridSearchCV

class RegressionTrainModel(AxiumTemplate):
    name = "RegressionTrainModel"
    id = "RegressionTrainModel"
    category = "regression"

    input = {
        "model": "axium.model",
        "features": "axium.dataframe",
        "target": "axium.series",
        "param_grid": "axium.dict"
    }
    output = {
        "trained_model": "axium.model"
    }

    @classmethod
    def run(cls, model, features, target, param_grid):
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(features, target)
        return {"trained_model": grid_search.best_estimator_}

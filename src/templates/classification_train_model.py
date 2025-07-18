from axium.template import AxiumTemplate
from sklearn.model_selection import GridSearchCV

class ClassificationTrainModel(AxiumTemplate):
    name = "ClassificationTrainModel"
    id = "ClassificationTrainModel"
    category = "classification"

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
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(features, target)
        return {"trained_model": grid_search.best_estimator_}

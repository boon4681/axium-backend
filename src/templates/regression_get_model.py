from axium.template import AxiumTemplate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

class RegressionGetModel(AxiumTemplate):
    name = "RegressionGetModel"
    id = "RegressionGetModel"
    category = "regression"

    input = {
        "model_name": "axium.str"
    }
    output = {
        "model": "axium.model"
    }

    @classmethod
    def run(cls, model_name):
        models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(),
            'svm': SVR(),
            'decision_tree': DecisionTreeRegressor(),
            'gradient_boosting': GradientBoostingRegressor(),
            'ada_boost': AdaBoostRegressor(),
            'extra_trees': ExtraTreesRegressor(),
            'neural_network': MLPRegressor()
        }
        if model_name not in models:
            raise ValueError(f"Model {model_name} is not available. Choose from {list(models.keys())}")
        return {"model": models[model_name]}

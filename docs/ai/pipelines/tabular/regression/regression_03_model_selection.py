import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor


class RegressionModelSelector:
    def __init__(self):
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(),
            'svm': SVR(),
            'decision_tree': DecisionTreeRegressor(),
            'gradient_boosting': GradientBoostingRegressor(),
            'ada_boost': AdaBoostRegressor(),
            'extra_trees': ExtraTreesRegressor(),
            'neural_network': MLPRegressor()
        }

    def get_model(self, model_name: str):
        if model_name not in self.models:
            raise ValueError(
                f"Model {model_name} is not available. Choose from {list(self.models.keys())}")
        return self.models[model_name]

    def list_models(self):
        """List all available regression models."""
        return print(list(self.models.keys()))

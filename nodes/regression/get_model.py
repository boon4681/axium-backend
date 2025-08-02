from axium.node_typing import AxiumNode

class RegressionGetModel(AxiumNode):
    id = "regression.get-model"
    category = "regression"
    name = "Get Model"

    inputs = {
        "model_path": ("axium.file", {})
    }
    outputs = {
        "model": ("sklearn.model", {})
    }
    parameters = None

    @classmethod
    def validate_inputs(cls, inputs: dict):
        return {}

    @classmethod
    def run(cls, parameters: dict, inputs: dict):
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.neural_network import MLPRegressor
        
        model_name = parameters.get("model_name") if parameters else None
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

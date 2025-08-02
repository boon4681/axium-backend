from axium.node_typing import AxiumNode

class ClassificationTrainModel(AxiumNode):
    id = "classification.train-model"
    category = "classification"
    name = "Train Model"

    inputs = {
        "data": ("pandas.df", {})
    }
    outputs = {
        "model": ("sklearn.model", {})
    }
    parameters = None

    @classmethod
    def validate_inputs(cls, inputs: dict):
        data = inputs.get("data")
        model = inputs.get("model")
        if data is None:
            return {"error": "Data is required"}
        if model is None:
            return {"error": "Model is required"}
        return {}

    @classmethod
    def run(cls, parameters: dict, inputs: dict):
        from sklearn.model_selection import GridSearchCV
        
        data = inputs.get("data")
        model = inputs.get("model")
        target_col = parameters.get("target") if parameters else None
        param_grid = parameters.get("param_grid", {}) if parameters else {}
        
        if data is None or model is None or target_col is None or target_col not in data.columns:
            return {"model": None}
        
        features = data.drop(columns=[target_col])
        target = data[target_col]
        
        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(features, target)
            trained_model = grid_search.best_estimator_
        else:
            model.fit(features, target)
            trained_model = model
            
        return {"model": trained_model}

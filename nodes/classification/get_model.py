from axium.node_typing import AxiumNode

class ClassificationGetModel(AxiumNode):
    id = "classification.get-model"
    category = "classification"
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
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neural_network import MLPClassifier
        model_name = parameters.get("model_name") if parameters else None
        models = {
            'logistic_regression': LogisticRegression(),
            'random_forest': RandomForestClassifier(),
            'svm': SVC(),
            'decision_tree': DecisionTreeClassifier(),
            'gradient_boosting': GradientBoostingClassifier(),
            'ada_boost': AdaBoostClassifier(),
            'extra_trees': ExtraTreesClassifier(),
            'neural_network': MLPClassifier()
        }
        if model_name not in models:
            raise ValueError(f"Model {model_name} is not available. Choose from {list(models.keys())}")
        return {"model": models[model_name]}

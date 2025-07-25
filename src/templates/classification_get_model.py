from axium.template import AxiumTemplate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

class ClassificationGetModel(AxiumTemplate):
    name = "ClassificationGetModel"
    id = "ClassificationGetModel"
    category = "classification"

    input = {
        "model_name": "axium.str"
    }
    output = {
        "model": "axium.model"
    }

    @classmethod
    def run(cls, model_name):
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

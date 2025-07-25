import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


class ClassificationModelSelector:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(),
            'random_forest': RandomForestClassifier(),
            'svm': SVC(),
            'decision_tree': DecisionTreeClassifier(),
            'gradient_boosting': GradientBoostingClassifier(),
            'ada_boost': AdaBoostClassifier(),
            'extra_trees': ExtraTreesClassifier(),
            'neural_network': MLPClassifier()
        }

    def get_model(self, model_name: str):
        if model_name not in self.models:
            raise ValueError(
                f"Model {model_name} is not available. Choose from {list(self.models.keys())}")
        return self.models[model_name]

    def list_models(self):
        """List all available classification models."""
        return print(list(self.models.keys()))

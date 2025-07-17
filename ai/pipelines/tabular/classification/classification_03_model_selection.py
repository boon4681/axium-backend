import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class ClassificationModelSelector:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(),
            'random_forest': RandomForestClassifier(),
            'svm': SVC(),
            'decision_tree': DecisionTreeClassifier()
        }

    def get_model(self, model_name: str):
        if model_name not in self.models:
            raise ValueError(
                f"Model {model_name} is not available. Choose from {list(self.models.keys())}")
        return self.models[model_name]

    def list_models(self):
        """List all available classification models."""
        return print(list(self.models.keys()))

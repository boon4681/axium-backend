import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ClassificationEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, metrics: list):
        y_pred = self.model.predict(X_test)
        results = {}

        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y_test, y_pred)
        if 'precision' in metrics:
            results['precision'] = precision_score(
                y_test, y_pred, average='binary')
        if 'recall' in metrics:
            results['recall'] = recall_score(y_test, y_pred, average='binary')
        if 'f1' in metrics:
            results['f1'] = f1_score(y_test, y_pred, average='binary')

        print("Evaluation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

        return results

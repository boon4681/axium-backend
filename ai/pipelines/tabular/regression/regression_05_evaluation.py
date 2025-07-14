import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class RegressionEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, metrics: list):
        y_pred = self.model.predict(X_test)
        results = {}

        if 'mse' in metrics:
            results['mse'] = mean_squared_error(y_test, y_pred)
        if 'mae' in metrics:
            results['mae'] = mean_absolute_error(y_test, y_pred)
        if 'r2' in metrics:
            results['r2'] = r2_score(y_test, y_pred)

        print("Evaluation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

        return results

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV


class ClassificationModelTrainer:
    def __init__(self, model):
        self.model = model

    def train_model(self, X: pd.DataFrame, y: pd.Series, param_grid: dict, test_size: float = 0.2, random_state: int = 42):
        # X_train, X_val, y_train, y_val = train_test_split(
        #     X, y, test_size=test_size, random_state=random_state)

        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X, y)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Validation accuracy: {grid_search.best_score_:.4f}")

        self.model = grid_search.best_estimator_
        return self.model

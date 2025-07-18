"""
Module for Saving and Loading Regression Models
"""
import joblib


class RegressionModelDeployment:
    @staticmethod
    def save_model(model, file_path):
        """Save the regression model to a file."""
        joblib.dump(model, file_path)
        print(f"Model saved to {file_path}")

    @staticmethod
    def load_model(file_path):
        """Load a regression model from a file."""
        model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")
        return model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')


class RegressionVisualizer:
    def __init__(self, model):
        self.model = model

    def plot_predictions_vs_actual(self, X_test: pd.DataFrame, y_test: pd.Series,
                                   figsize=(8, 6), save_path=None, show_plot=True):
        """Plot predicted vs actual values."""
        y_pred = self.model.predict(X_test)

        plt.figure(figsize=figsize)
        plt.scatter(y_test, y_pred, alpha=0.6)

        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual Values')
        plt.grid(True, alpha=0.3)

        # Add R² score
        r2 = r2_score(y_test, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Predictions vs Actual plot saved to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return y_pred

    def plot_residuals(self, X_test: pd.DataFrame, y_test: pd.Series,
                       figsize=(12, 4), save_path=None, show_plot=True):
        """Plot residual plots to check model assumptions."""
        y_pred = self.model.predict(X_test)
        residuals = y_test - y_pred

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)

        # Residuals histogram
        axes[1].hist(residuals, bins=30, alpha=0.7, color='skyblue')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residuals Distribution')
        axes[1].grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Residuals plot saved to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return residuals

    def plot_learning_curve(self, X: pd.DataFrame, y: pd.Series,
                            cv=5, figsize=(10, 6), save_path=None, show_plot=True):
        """Plot learning curve to show training and validation scores."""
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error')

        # Convert to positive MSE
        train_scores = -train_scores
        val_scores = -val_scores

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=figsize)
        plt.plot(train_sizes, train_mean, 'o-', color='blue',
                 label='Training MSE')
        plt.fill_between(train_sizes, train_mean - train_std,
                         train_mean + train_std, alpha=0.1, color='blue')

        plt.plot(train_sizes, val_mean, 'o-', color='red',
                 label='Validation MSE')
        plt.fill_between(train_sizes, val_mean - val_std,
                         val_mean + val_std, alpha=0.1, color='red')

        plt.xlabel('Training Set Size')
        plt.ylabel('Mean Squared Error')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curve saved to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_feature_importance(self, feature_names: list, figsize=(10, 6), save_path=None, show_plot=True):
        """Plot feature importance if model supports it."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=figsize)
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)),
                       [feature_names[i] for i in indices], rotation=45)
            plt.title('Feature Importance')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Feature importance plot saved to: {save_path}")

            if show_plot:
                plt.show()
            else:
                plt.close()
        else:
            print("Model does not support feature importance.")

    def plot_error_distribution(self, X_test: pd.DataFrame, y_test: pd.Series,
                                figsize=(12, 4), save_path=None, show_plot=True):
        """Plot error distribution analysis."""
        y_pred = self.model.predict(X_test)

        # Calculate different error metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        absolute_errors = np.abs(y_test - y_pred)
        percentage_errors = (absolute_errors / np.abs(y_test)) * 100

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Absolute errors
        axes[0].hist(absolute_errors, bins=30, alpha=0.7, color='lightcoral')
        axes[0].set_xlabel('Absolute Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Absolute Error Distribution\nMAE = {mae:.3f}')
        axes[0].grid(True, alpha=0.3)

        # Percentage errors
        axes[1].hist(percentage_errors, bins=30, alpha=0.7, color='lightgreen')
        axes[1].set_xlabel('Percentage Error (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Percentage Error Distribution')
        axes[1].grid(True, alpha=0.3)

        # Error vs Predicted
        axes[2].scatter(y_pred, absolute_errors, alpha=0.6, color='orange')
        axes[2].set_xlabel('Predicted Values')
        axes[2].set_ylabel('Absolute Error')
        axes[2].set_title('Error vs Predicted Values')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Error distribution plot saved to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mean_percentage_error': np.mean(percentage_errors)
        }

    def plot_prediction_intervals(self, X_test: pd.DataFrame, y_test: pd.Series,
                                  figsize=(10, 6), save_path=None, show_plot=True):
        """Plot predictions with confidence intervals (for models that support it)."""
        y_pred = self.model.predict(X_test)

        # Sort by predicted values for better visualization
        sorted_idx = np.argsort(y_pred)
        y_test_sorted = y_test.iloc[sorted_idx]
        y_pred_sorted = y_pred[sorted_idx]

        plt.figure(figsize=figsize)

        # Plot actual vs predicted
        plt.scatter(range(len(y_test_sorted)), y_test_sorted,
                    alpha=0.6, label='Actual', color='blue')
        plt.plot(range(len(y_pred_sorted)), y_pred_sorted,
                 color='red', label='Predicted', linewidth=2)

        # Add error bands (approximate)
        residuals = y_test_sorted - y_pred_sorted
        std_residuals = np.std(residuals)

        plt.fill_between(range(len(y_pred_sorted)),
                         y_pred_sorted - 1.96 * std_residuals,
                         y_pred_sorted + 1.96 * std_residuals,
                         alpha=0.2, color='red', label='95% Prediction Interval')

        plt.xlabel('Sample Index (sorted by prediction)')
        plt.ylabel('Target Value')
        plt.title('Predictions with Confidence Intervals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction intervals plot saved to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_all_metrics(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series,
                         feature_names: list = None, save_dir: str = None, show_plots=True):
        """Plot all available visualizations."""
        print("Generating all visualization plots...")

        # Create save paths if save_dir is provided
        save_paths = {}
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            save_paths = {
                'predictions_vs_actual': os.path.join(save_dir, 'predictions_vs_actual.png'),
                'residuals': os.path.join(save_dir, 'residuals.png'),
                'learning_curve': os.path.join(save_dir, 'learning_curve.png'),
                'feature_importance': os.path.join(save_dir, 'feature_importance.png'),
                'error_distribution': os.path.join(save_dir, 'error_distribution.png'),
                'prediction_intervals': os.path.join(save_dir, 'prediction_intervals.png')
            }

        # Predictions vs Actual
        self.plot_predictions_vs_actual(X_test, y_test,
                                        save_path=save_paths.get(
                                            'predictions_vs_actual'),
                                        show_plot=show_plots)

        # Residual Analysis
        self.plot_residuals(X_test, y_test,
                            save_path=save_paths.get('residuals'),
                            show_plot=show_plots)

        # Learning Curve
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test])
        self.plot_learning_curve(X_full, y_full,
                                 save_path=save_paths.get('learning_curve'),
                                 show_plot=show_plots)

        # Feature Importance
        if feature_names:
            self.plot_feature_importance(feature_names,
                                         save_path=save_paths.get(
                                             'feature_importance'),
                                         show_plot=show_plots)

        # Error Distribution
        error_metrics = self.plot_error_distribution(X_test, y_test,
                                                     save_path=save_paths.get(
                                                         'error_distribution'),
                                                     show_plot=show_plots)

        # Prediction Intervals
        self.plot_prediction_intervals(X_test, y_test,
                                       save_path=save_paths.get(
                                           'prediction_intervals'),
                                       show_plot=show_plots)

        if save_dir:
            print(f"All plots saved to directory: {save_dir}")

        return error_metrics

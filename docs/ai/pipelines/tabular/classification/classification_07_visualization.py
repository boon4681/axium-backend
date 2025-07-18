import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')


class ClassificationVisualizer:
    def __init__(self, model):
        self.model = model

    def plot_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series,
                              class_names=None, figsize=(8, 6), save_path=None, show_plot=True):
        """Plot confusion matrix heatmap."""
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return cm

    def plot_roc_curve(self, X_test: pd.DataFrame, y_test: pd.Series,
                       figsize=(8, 6), save_path=None, show_plot=True):
        """Plot ROC curve for binary or multiclass classification."""
        plt.figure(figsize=figsize)

        # Check if binary or multiclass
        unique_classes = np.unique(y_test)

        if len(unique_classes) == 2:
            # Binary classification
            if hasattr(self.model, 'predict_proba'):
                y_score = self.model.predict_proba(X_test)[:, 1]
            else:
                y_score = self.model.decision_function(X_test)

            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        else:
            # Multiclass classification
            lb = LabelBinarizer()
            y_test_bin = lb.fit_transform(y_test)
            y_score = self.model.predict_proba(X_test)

            for i in range(len(unique_classes)):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2,
                         label=f'Class {unique_classes[i]} (AUC = {roc_auc:.2f})')

            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_precision_recall_curve(self, X_test: pd.DataFrame, y_test: pd.Series,
                                    figsize=(8, 6), save_path=None, show_plot=True):
        """Plot precision-recall curve."""
        if hasattr(self.model, 'predict_proba'):
            y_score = self.model.predict_proba(X_test)[:, 1]
        else:
            y_score = self.model.decision_function(X_test)

        precision, recall, _ = precision_recall_curve(y_test, y_score)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='darkorange', lw=2,
                 label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_learning_curve(self, X: pd.DataFrame, y: pd.Series,
                            cv=5, figsize=(10, 6), save_path=None, show_plot=True):
        """Plot learning curve to show training and validation scores."""
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10))

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=figsize)
        plt.plot(train_sizes, train_mean, 'o-', color='blue',
                 label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std,
                         train_mean + train_std, alpha=0.1, color='blue')

        plt.plot(train_sizes, val_mean, 'o-', color='red',
                 label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std,
                         val_mean + val_std, alpha=0.1, color='red')

        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
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

    def plot_classification_report(self, X_test: pd.DataFrame, y_test: pd.Series,
                                   class_names=None, figsize=(8, 6), save_path=None, show_plot=True):
        """Plot classification report as heatmap."""
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred,
                                       target_names=class_names,
                                       output_dict=True)

        # Remove support column and create DataFrame
        df_report = pd.DataFrame(report).transpose()
        df_report = df_report.drop('support', axis=1)

        plt.figure(figsize=figsize)
        sns.heatmap(df_report, annot=True, cmap='Blues', fmt='.2f')
        plt.title('Classification Report')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Classification report saved to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_all_metrics(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series,
                         feature_names: list = None, class_names=None,
                         save_dir: str = None, show_plots=True):
        """Plot all available visualizations."""
        print("Generating all visualization plots...")

        # Create save paths if save_dir is provided
        save_paths = {}
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            save_paths = {
                'confusion_matrix': os.path.join(save_dir, 'confusion_matrix.png'),
                'roc_curve': os.path.join(save_dir, 'roc_curve.png'),
                'precision_recall': os.path.join(save_dir, 'precision_recall_curve.png'),
                'learning_curve': os.path.join(save_dir, 'learning_curve.png'),
                'feature_importance': os.path.join(save_dir, 'feature_importance.png'),
                'classification_report': os.path.join(save_dir, 'classification_report.png')
            }

        # Confusion Matrix
        self.plot_confusion_matrix(X_test, y_test, class_names,
                                   save_path=save_paths.get(
                                       'confusion_matrix'),
                                   show_plot=show_plots)

        # ROC Curve
        self.plot_roc_curve(X_test, y_test,
                            save_path=save_paths.get('roc_curve'),
                            show_plot=show_plots)

        # Precision-Recall Curve (only for binary classification)
        if len(np.unique(y_test)) == 2:
            self.plot_precision_recall_curve(X_test, y_test,
                                             save_path=save_paths.get(
                                                 'precision_recall'),
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

        # Classification Report
        self.plot_classification_report(X_test, y_test, class_names,
                                        save_path=save_paths.get(
                                            'classification_report'),
                                        show_plot=show_plots)

        if save_dir:
            print(f"All plots saved to directory: {save_dir}")

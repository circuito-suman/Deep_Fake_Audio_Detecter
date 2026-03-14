from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import os

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.output_dir = config['outputs']['plots_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Initialize classifiers with config parameters
        models_cfg = self.config['models']
        
        self.classifiers = {
            'SVM': SVC(
                kernel=models_cfg['svm']['kernel'], 
                probability=models_cfg['svm']['probability'],
                random_state=self.config['random_seed']
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=models_cfg['random_forest']['n_estimators'],
                random_state=models_cfg['random_forest']['random_state']
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=models_cfg['logistic_regression']['max_iter'],
                random_state=models_cfg['logistic_regression']['random_state']
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=models_cfg['knn']['n_neighbors']
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=tuple(models_cfg['mlp']['hidden_layer_sizes']), 
                max_iter=models_cfg['mlp']['max_iter'], 
                alpha=models_cfg['mlp']['alpha'],
                random_state=models_cfg['mlp']['random_state']
            )
        }
        self.results = {}

    def train_classifiers(self, X_train, y_train):
        for name, clf in self.classifiers.items():
            self.logger.info(f"Training {name}...")
            clf.fit(X_train, y_train)
            self.logger.info(f"{name} trained.")

    def compute_eer(self, y_true, y_score):
        """Computes the Equal Error Rate (EER)."""
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        err = 0 
        # fnr = 1 - tpr
        # the EER is where fpr = fnr
        # so we look for where fpr - (1 - tpr) crosses 0
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return eer, eer_threshold

    def evaluate_classifiers(self, X_test, y_test):
        for name, clf in self.classifiers.items():
            self.logger.info(f"Evaluating {name}...")
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Calculate EER
            eer = 0.0
            if y_prob is not None:
                eer, _ = self.compute_eer(y_test, y_prob)

            self.results[name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'eer': eer,
                'y_prob': y_prob,
                'clf': clf
            }
            
            self.logger.info(f"{name} - Accuracy: {acc:.4f}, EER: {eer:.4f}, F1: {f1:.4f}")
            print(f"\n{name} Classification Report:\n{classification_report(y_test, y_pred)}")

    def plot_confusion_matrices(self, X_test, y_test):
        for name, res in self.results.items():
            clf = res['clf']
            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'confusion_matrix_{name.replace(" ", "_")}.png'))
            plt.close()

    def plot_roc_curves(self, y_test):
        plt.figure(figsize=(10, 8))
        for name, res in self.results.items():
            y_prob = res.get('y_prob')
            if y_prob is not None:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.output_dir, 'roc_curves.png'))
        plt.close()

"""
Actuarial Evaluation and Visualization Module.
Generates categorized industrial-grade plots for reporting.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

logger = logging.getLogger(__name__)

class ActuarialVisualizer:
    """
    Handles organized export of quantitative performance graphics.
    """
    
    def __init__(self, base_dir: str = "results"):
        self.base_dir = base_dir
        # 定义子文件夹路径映射
        self.paths = {
            "roc": os.path.join(base_dir, "roc_curves"),
            "cm": os.path.join(base_dir, "confusion_matrices"),
            "feat": os.path.join(base_dir, "feature_importance")
        }
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Creates the directory tree if it doesn't exist."""
        for path in self.paths.values():
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                logger.info(f"Created/Verified directory: {path}")

    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, model_name: str):
        """Generates and saves a high-DPI ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title(f'ROC Curve: {model_name.upper()}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        # 自动存入 roc_curves 子目录
        save_path = os.path.join(self.paths["roc"], f"roc_{model_name.lower()}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Exported ROC plot to: {save_path}")

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Generates and saves a styled Confusion Matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Healthy', 'Bankrupt'], yticklabels=['Healthy', 'Bankrupt'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix: {model_name.upper()}')
        
        # 自动存入 confusion_matrices 子目录
        save_path = os.path.join(self.paths["cm"], f"cm_{model_name.lower()}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Exported Confusion Matrix to: {save_path}")

    def plot_feature_importance(self, feature_names: list, importances: np.ndarray, model_name: str, top_n: int = 15):
        """Generates and saves a feature importance bar chart."""
        indices = np.argsort(importances)[::-1][:top_n]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
        plt.title(f'Top {top_n} Actuarial Features: {model_name.upper()}')
        plt.xlabel('Relative Importance')
        
        # 自动存入 feature_importance 子目录
        save_path = os.path.join(self.paths["feat"], f"feat_{model_name.lower()}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Exported Feature Importance to: {save_path}")
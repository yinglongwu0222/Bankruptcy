import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score


def plot_roc_curve_comparison_per_model(y_true_dict, y_score_dict, model_name, save_path=None):
    plt.figure(figsize=(8, 6))
    for method in y_true_dict:
        fpr, tpr, _ = roc_curve(y_true_dict[method], y_score_dict[method])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{method} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_curve_comparison_per_model(y_true_dict, y_score_dict, model_name, save_path=None):
    plt.figure(figsize=(8, 6))
    for method in y_true_dict:
        precision, recall, _ = precision_recall_curve(y_true_dict[method], y_score_dict[method])
        pr_auc = average_precision_score(y_true_dict[method], y_score_dict[method])
        plt.plot(recall, precision, lw=2, label=f"{method} (AP={pr_auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve - {model_name}")
    plt.legend(loc="lower left")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, labels, title="Confusion Matrix", save_path=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_feature_importance_bar(df, score_col, title, save_path=None, top_n=20):
    top_df = df.sort_values(by=score_col, ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=score_col, y="feature", data=top_df, palette="viridis", hue="feature", legend=False)
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_correlation_heatmap(X, save_path=None):
    plt.figure(figsize=(12, 10))
    sns.heatmap(X.corr(), cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

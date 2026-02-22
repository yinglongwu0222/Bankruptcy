import os
from datetime import datetime

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from processing.config import load_config
from processing.data_loader import load_data, split_data
from processing.feature_selection import combined_feature_selection
from processing.sampler import apply_sampling
from models.train_predict import get_model
from evaluation.metrics import compute_metrics
from evaluation.visualizer import plot_confusion_matrix, plot_roc_curve_comparison_per_model, plot_pr_curve_comparison_per_model
import pandas as pd

matplotlib.use('TkAgg')


def run_pipeline(config_path="config/config.yaml"):
    config = load_config(config_path)
    data_path = config.get("data_path", "data/data.csv")
    target_col = config.get("target_col", "target")
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df, target_col,
                                                  test_size=config.get("test_size", 0.2),
                                                  random_state=config.get("random_seed", 42))

    print("Starting feature selection...")
    corr_threshold = config.get("feature_selection", {}).get("corr_threshold", 0.8)
    vif_threshold = config.get("feature_selection", {}).get("vif_threshold", 10)
    top_k = config.get("feature_selection", {}).get("top_k", 50)
    selected_features = combined_feature_selection(X_train, y_train,
                                                   corr_threshold=corr_threshold,
                                                   vif_threshold=vif_threshold,
                                                   top_k=top_k)

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    results = []
    models_scores = {}

    for sampler_name in config.get("sampling_methods", []):
        print(f"Sampling method: {sampler_name}")
        X_res, y_res = apply_sampling(X_train, y_train, method=sampler_name, random_state=config.get("random_seed", 42))

        for model_name in config.get("models", []):
            print(f" Training model: {model_name}")
            model = get_model(model_name)
            model.fit(X_res, y_res)
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                y_score = model.decision_function(X_test)
            metrics = compute_metrics(y_test, y_pred, y_score)
            metrics.update({
                "model": model_name,
                "sampling": sampler_name if sampler_name else "None"
            })
            results.append(metrics)
            label = f"{model_name}_{sampler_name if sampler_name else 'None'}"
            models_scores[label] = (y_test, y_score)

    df_results = pd.DataFrame(results)
    os.makedirs("results/metric_summary", exist_ok=True)
    df_results.to_csv("results/metric_summary/results_summary.csv", index=False)
    print("Results saved to results/metric_summary/results_summary.csv")

    # Bar plot for F1 and Recall
    metrics = ["f1", "recall"]
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_results, x="model", y=metric, hue="sampling")
        plt.title(f"{metric.upper()} Comparison Across Models and Sampling Methods")
        plt.ylabel(metric.upper())
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.legend(title="Sampling", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"results/metric_summary/{metric}_bar_comparison.png")
        plt.close()


    os.makedirs("results/roc_curves", exist_ok=True)
    os.makedirs("results/pr_curves", exist_ok=True)
    os.makedirs("results/confusion_matrices", exist_ok=True)

    # Get all model names
    all_models = config.get("models", [])

    for model_name in all_models:
        y_true_dict = {}
        y_score_dict = {}
        for sampler_name in config.get("sampling_methods", []):
            label = f"{model_name}_{sampler_name if sampler_name else 'None'}"
            if label in models_scores:
                y_true_dict[sampler_name] = models_scores[label][0]
                y_score_dict[sampler_name] = models_scores[label][1]

        if y_true_dict:
            plot_roc_curve_comparison_per_model(
                y_true_dict,
                y_score_dict,
                model_name=model_name,
                save_path=f"results/roc_curves/roc_{model_name.lower().replace(' ', '_')}.png"
            )
            plot_pr_curve_comparison_per_model(
                y_true_dict,
                y_score_dict,
                model_name=model_name,
                save_path=f"results/pr_curves/pr_{model_name.lower().replace(' ', '_')}.png"
            )

    # Plot confusion matrices for all model-sampler combinations
    for _, row in df_results.iterrows():
        label = f'{row["model"]}_{row["sampling"]}'
        cm = row["confusion_matrix"]
        plot_confusion_matrix(
            cm,
            labels=[0, 1],
            title=f"Confusion Matrix {label}",
            save_path=f"results/confusion_matrices/cm_{label}.png"
        )


if __name__ == "__main__":
    print("----------------------")
    start_time = datetime.now()
    print("Start time:", start_time)
    run_pipeline()
    end_time = datetime.now()
    print("End time:", end_time)
    print("Elapsed time:", end_time - start_time)
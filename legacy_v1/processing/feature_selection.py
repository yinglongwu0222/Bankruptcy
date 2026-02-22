import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import lightgbm as lgb
import shap

from evaluation.visualizer import plot_correlation_heatmap, plot_feature_importance_bar


def remove_constant_and_invalid_columns(X):
    """
    Remove constant or invalid columns
    """
    X = X.loc[:, X.nunique() > 1]
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    return X


def remove_highly_correlated_features(X, threshold=0.7):
    """
    Remove highly correlated features
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f"Removed {len(to_drop)} highly correlated features: {to_drop}")
    return X.drop(columns=to_drop)


def calculate_vif(X, threshold=10.0):
    """
    VIF filtering
    """
    X_scaled = StandardScaler().fit_transform(X)
    vif_data = []
    for i in range(X.shape[1]):
        try:
            vif = variance_inflation_factor(X_scaled, i)
        except Exception as e:
            print(f"Skipping VIF calculation for feature {X.columns[i]} due to error: {e}")
            vif = np.inf
        vif_data.append((X.columns[i], vif))

    vif_df = pd.DataFrame(vif_data, columns=["feature", "VIF"])
    selected = vif_df[vif_df["VIF"] < threshold]["feature"].tolist()
    return selected


def shap_feature_importance(X, y, top_k=50):
    """
    Compute SHAP feature importance using LightGBM.

    Parameters:
        X (pd.DataFrame): Input features
        y (pd.Series or np.array): Labels
        top_k (int): Number of top features to return

    Returns:
        pd.DataFrame: DataFrame with 'feature' and 'shap_importance'
    """
    original_cols = X.columns.tolist()
    safe_feature_names = [f"f{i}" for i in range(X.shape[1])]
    feature_map = dict(zip(safe_feature_names, original_cols))

    # Prepare LightGBM data with safe names
    lgb_data = lgb.Dataset(X.values, label=y, feature_name=safe_feature_names)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "n_estimators": 100,
        "boost_from_average": True,
        "random_state": 42
    }
    model = lgb.train(params, lgb_data)

    # shap analysis
    X_safe = pd.DataFrame(X.values, columns=safe_feature_names)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_safe)

    # Select positive class SHAP values
    shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap_importance = np.abs(shap_vals).mean(axis=0)

    shap_df = pd.DataFrame({
        "feature": [feature_map[f] for f in safe_feature_names],
        "shap_importance": shap_importance
    })
    return shap_df.sort_values(by="shap_importance", ascending=False).head(top_k)


def rf_feature_importance(X, y, top_k=50):
    """
    Compute feature importance using RandomForest.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_df = pd.DataFrame({
        "feature": X.columns,
        "rf_importance": rf.feature_importances_
    })
    return rf_df.sort_values(by="rf_importance", ascending=False).head(top_k)


def combined_feature_selection(X, y, corr_threshold=0.8, vif_threshold=10, top_k=50, save_dir="results/feature_selection"):
    """
    Comprehensive feature selection process：
    - Remove constant or invalid columns
    - Remove highly correlated features
    - VIF filtering
    - SHAP + RandomForest ranking combination
    - Visualization output
    """
    os.makedirs(save_dir, exist_ok=True)

    # correlation heatmap
    plot_correlation_heatmap(X, save_path=os.path.join(save_dir, "feature_correlation_heatmap_raw.png"))

    # data processing
    print("Remove constant or invalid columns...")
    X_clean = remove_constant_and_invalid_columns(X)
    print("Remove highly correlated features...")
    X_clean = remove_highly_correlated_features(X_clean, threshold=corr_threshold)
    print("VIF filtering...")
    selected_vif = calculate_vif(X_clean, threshold=vif_threshold)
    X_filtered = X_clean[selected_vif]

    print("Calculating feature importance...")
    shap_df = shap_feature_importance(X_filtered, y, top_k=top_k)
    rf_df = rf_feature_importance(X_filtered, y, top_k=top_k)

    # feature importance
    plot_feature_importance_bar(shap_df, "shap_importance", "Top SHAP Feature Importance",
                                save_path=os.path.join(save_dir, "shap_feature_importance.png"))
    plot_feature_importance_bar(rf_df, "rf_importance", "Top Random Forest Feature Importance",
                                save_path=os.path.join(save_dir, "rf_feature_importance.png"))

    merged = pd.merge(shap_df, rf_df, on="feature")
    merged["mean_rank"] = (
        merged["shap_importance"].rank(ascending=False) +
        merged["rf_importance"].rank(ascending=False)
    ) / 2

    top_features = merged.sort_values("mean_rank").head(top_k)["feature"].tolist()
    return top_features

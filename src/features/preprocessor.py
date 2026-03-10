import logging
from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)

class FinancialFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, corr_threshold: float = 0.7, vif_threshold: float = 10.0):
        self.corr_threshold = corr_threshold
        self.vif_threshold = vif_threshold
        self.scaler = StandardScaler()
        self.selected_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        logger.info(f"Starting Preprocessing: Input features = {X.shape[1]}")
        
        # 1. Basic Cleaning
        X_clean = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        X_clean = X_clean.loc[:, X_clean.nunique() > 1].copy()
        
        # 2. Filter high correlation features first
        corr_matrix = X_clean.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_corr = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]
        X_clean = X_clean.drop(columns=to_drop_corr)
        logger.info(f"Dropped {len(to_drop_corr)} highly correlated features. Remaining: {X_clean.shape[1]}")

        # 3. VIF Selection
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_clean),
            columns=X_clean.columns,
            index=X_clean.index
        )

        current_cols = X_clean.columns.tolist()
        logger.info("Starting VIF selection...")
        
        iteration = 0
        while True:
            # Calculate VIF
            vif_values = [
                variance_inflation_factor(X_scaled[current_cols].values, i)
                for i in range(len(current_cols))
            ]
            max_vif = max(vif_values)

            if max_vif > self.vif_threshold:
                max_idx = vif_values.index(max_vif)
                dropped_feature = current_cols.pop(max_idx)
                iteration += 1
                # Print progress
                print(f"  > [VIF Step {iteration}] Dropped: {dropped_feature} (VIF: {max_vif:.2f})")
            else:
                break

        self.selected_features_ = current_cols
        
        # Refit scaler on final features
        self.scaler.fit(X_clean[self.selected_features_])
        
        logger.info(f"Feature Selection Complete. Final features: {len(self.selected_features_)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Ensure test data is clean (handle inf/nan) before scaling to prevent runtime crashes
        X_selected = X[self.selected_features_].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_transformed = self.scaler.transform(X_selected)
        return pd.DataFrame(X_transformed, columns=self.selected_features_, index=X.index)
"""
Industrial Machine Learning Pipeline and Evaluation Engine.
Orchestrates the preprocessing, sampling, and modeling stages into a unified DAG.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, roc_auc_score, f1_score

logger = logging.getLogger(__name__)

class ActuarialPipelineEngine:
    """
    Unified engine to compile and evaluate the end-to-end machine learning pipeline.
    Ensures strict isolation of train/validation folds to prevent data leakage.
    """

    def __init__(self, preprocessor: Any, sampler: Optional[Any], classifier: Any):
        """
        Args:
            preprocessor: Instantiated feature selection and scaling transformer.
            sampler: Instantiated imblearn resampling object (or None).
            classifier: Instantiated predictive model.
        """
        self.preprocessor = preprocessor
        self.sampler = sampler
        self.classifier = classifier
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> ImbPipeline:
        """
        Compiles the components into an imblearn Pipeline (DAG).
        This guarantees that resampling is strictly applied ONLY to the training splits.
        """
        steps = [("preprocessor", self.preprocessor)]
        
        if self.sampler is not None:
            steps.append(("sampler", self.sampler))
            
        steps.append(("model", self.classifier))
        
        logger.info("DAG Pipeline compiled successfully.")
        return ImbPipeline(steps)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, float]:
        """
        Executes rigorous Stratified K-Fold Cross-Validation.
        
        Returns:
            Dict: Aggregated out-of-fold metrics (mean and standard deviation).
        """
        logger.info(f"Initiating {n_splits}-Fold Stratified Cross-Validation...")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_recalls = []
        cv_aucs = []
        cv_f1s = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            # Strict isolation of training and validation indices
            X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
            X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

            # Fit the entire DAG pipeline strictly on the training fold
            self.pipeline.fit(X_train_fold, y_train_fold)

            # Inference on the untouched validation fold
            y_pred = self.pipeline.predict(X_val_fold)
            
            # Probability inference for AUC (fallback to hard labels if unsupported)
            if hasattr(self.pipeline.named_steps['model'], "predict_proba"):
                y_proba = self.pipeline.predict_proba(X_val_fold)[:, 1]
            else:
                y_proba = y_pred

            # Compute fold metrics
            cv_recalls.append(recall_score(y_val_fold, y_pred))
            cv_aucs.append(roc_auc_score(y_val_fold, y_proba))
            cv_f1s.append(f1_score(y_val_fold, y_pred))
            
            logger.debug(f"Fold {fold} - Recall: {cv_recalls[-1]:.3f} | AUC: {cv_aucs[-1]:.3f}")

        # Aggregate metrics
        results = {
            "mean_recall": float(np.mean(cv_recalls)),
            "std_recall": float(np.std(cv_recalls)),
            "mean_auc": float(np.mean(cv_aucs)),
            "mean_f1": float(np.mean(cv_f1s))
        }
        
        logger.info(f"CV Complete. Mean Recall: {results['mean_recall']:.3f} (+/- {results['std_recall']:.3f})")
        return results

    def fit_production(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'ActuarialPipelineEngine':
        """
        Trains the pipeline on the full training dataset for production deployment.
        """
        logger.info("Training production model on full training set...")
        self.pipeline.fit(X_train, y_train)
        return self

    def predict_production(self, X_test: pd.DataFrame) -> np.ndarray:
        """Executes inference using the production-ready pipeline."""
        return self.pipeline.predict(X_test)
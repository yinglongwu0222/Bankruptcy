"""
Pipeline wrapper for training and evaluation.
Handles the training loop and cross-validation.
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
    Wrapper to manage the imblearn pipeline and cross-validation.
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
        Builds the imblearn Pipeline.
        """
        steps = [("preprocessor", self.preprocessor)]
        
        if self.sampler is not None:
            steps.append(("sampler", self.sampler))
            
        steps.append(("model", self.classifier))
        
        logger.info("Pipeline built.")
        return ImbPipeline(steps)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, float]:
        """
        Runs Stratified K-Fold Cross-Validation.
        """
        logger.info(f"Starting {n_splits}-Fold CV...")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_recalls = []
        cv_aucs = []
        cv_f1s = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            # Split data
            X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
            X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

            # Fit pipeline
            self.pipeline.fit(X_train_fold, y_train_fold)

            # Predict
            y_pred = self.pipeline.predict(X_val_fold)
            
            # Get probabilities if available
            try:
                y_proba = self.pipeline.predict_proba(X_val_fold)[:, 1]
            except (NotImplementedError, AttributeError):
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
        Fits the model on the full training set.
        """
        logger.info("Fitting model on full data...")
        self.pipeline.fit(X_train, y_train)
        return self

    def predict_production(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predicts on new data."""
        return self.pipeline.predict(X_test)
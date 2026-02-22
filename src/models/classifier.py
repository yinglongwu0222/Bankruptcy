"""
Advanced actuarial modeling engine.
Wraps standard algorithms into a unified OOP interface and implements 
mathematically derived custom objective functions for extreme class imbalance.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

class ActuarialModelEngine(BaseEstimator, ClassifierMixin):
    """
    Unified industrial wrapper for machine learning classifiers.
    Injects custom objective functions for rare-event (default) prediction.
    """

    def __init__(self, model_name: str, params: Dict[str, Any], custom_loss: bool = False):
        self.model_name = model_name.lower().strip()
        self.params = params
        self.custom_loss = custom_loss
        self.model = self._initialize_model()

    def _initialize_model(self) -> BaseEstimator:
        """Instantiates the specified algorithm with rigorous error handling."""
        logger.info(f"Initializing predictive engine: {self.model_name.upper()}")
        
        if self.model_name == "logistic":
            return LogisticRegression(**self.params)
        elif self.model_name == "svm":
            return SVC(**self.params)
        elif self.model_name == "random_forest":
            return RandomForestClassifier(**self.params)
        elif self.model_name == "xgboost":
            if self.custom_loss:
                # Remove standard metric to utilize custom Hessian/Gradient calculation
                self.params.pop("eval_metric", None)
                self.params["objective"] = self._asymmetric_log_loss
                logger.warning("Custom Asymmetric Objective Function injected into XGBoost.")
            return xgb.XGBClassifier(**self.params)
        else:
            raise ValueError(f"Architecture '{self.model_name}' is not supported in the engine.")

    def _asymmetric_log_loss(self, labels: np.ndarray, preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        [Implemented From Scratch]: Custom Asymmetric Objective Function.
        Adjusted for modern XGBoost API where labels are passed directly.
        """
        # XGBoost margin output requires sigmoid transformation
        p = 1.0 / (1.0 + np.exp(-preds))
        
        # Actuarial penalty for False Negatives (Missing a bankruptcy)
        alpha = 5.0 
        
        grad = p - labels
        hess = p * (1.0 - p)
        
        # Apply the penalty multiplier to samples where the true status is 'Bankrupt'
        grad[labels == 1] *= alpha
        hess[labels == 1] *= alpha
        
        return grad, hess

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ActuarialModelEngine':
        """Trains the underlying model engine."""
        logger.info(f"Commencing training sequence for {self.model_name.upper()}")
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            logger.error(f"Model {self.model_name} does not support probability estimation.")
            raise NotImplementedError("Probability inference not available for this architecture.")
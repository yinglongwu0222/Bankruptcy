"""
Model wrapper for classifiers and custom loss functions.
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
    Wrapper for classifiers.
    """

    def __init__(self, model_name: str, params: Dict[str, Any], custom_loss: bool = False):
        self.model_name = model_name.lower().strip()
        self.params = params
        self.custom_loss = custom_loss
        self.model = self._initialize_model()

    def _initialize_model(self) -> BaseEstimator:
        """Initializes the model."""
        logger.info(f"Initializing model: {self.model_name.upper()}")
        
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
                logger.warning("Using custom asymmetric loss.")
            return xgb.XGBClassifier(**self.params)
        else:
            raise ValueError(f"Architecture '{self.model_name}' is not supported in the engine.")

    def _asymmetric_log_loss(self, labels: np.ndarray, preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Custom Asymmetric Log Loss for XGBoost.
        """
        # Numerically stable sigmoid mapped from log-odds margin
        p = 1.0 / (1.0 + np.exp(-np.clip(preds, -15, 15)))
        
        # Actuarial penalty for False Negatives (Missing a bankruptcy)
        alpha = 5.0 
        
        grad = p - labels
        hess = p * (1.0 - p)
        
        # Apply the penalty multiplier to samples where the true status is 'Bankrupt'
        grad[labels == 1] *= alpha
        hess[labels == 1] *= alpha
        
        return grad, hess

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ActuarialModelEngine':
        """Trains the model."""
        logger.info(f"Training {self.model_name.upper()}")
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts probabilities. Handles custom loss margin conversion.
        """
        if self.model_name == "xgboost" and self.custom_loss:
            # Fix: Get margin and manually apply sigmoid to avoid overflow
            dmatrix = xgb.DMatrix(X)
            margin = self.model.get_booster().predict(dmatrix, output_margin=True)
            
            # 使用截断避免 exp 溢出导致 NaNs
            prob_1 = 1.0 / (1.0 + np.exp(-np.clip(margin, -15, 15)))
            prob_0 = 1.0 - prob_1
            return np.vstack((prob_0, prob_1)).T
            
        elif hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            logger.error(f"Model {self.model_name} does not support probability estimation.")
            raise NotImplementedError("Probability inference not available for this architecture.")
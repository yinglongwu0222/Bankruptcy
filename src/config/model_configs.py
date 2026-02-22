"""
Model configuration management.
Provides robust hyperparameters fetching via Factory Pattern.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ModelConfigFactory:
    """
    Factory class to securely retrieve machine learning model configurations.
    Enforces standardized hyperparameter dictionaries for pipeline execution.
    """
    
    _CONFIGS: Dict[str, Dict[str, Any]] = {
        "logistic": {
            "C": 1.0, 
            "solver": "liblinear", 
            "max_iter": 1000, 
            "random_state": 42
        },
        "svm": {
            "C": 1.0, 
            "kernel": "rbf", 
            "probability": True, 
            "random_state": 42
        },
        "random_forest": {
            "n_estimators": 100, 
            "criterion": "gini", 
            "max_depth": 10, 
            "random_state": 42
        },
        "xgboost": {
            "n_estimators": 100, 
            "learning_rate": 0.1, 
            "max_depth": 10, 
            "eval_metric": "logloss", 
            "random_state": 42
        }
    }

    @classmethod
    def get_params(cls, model_name: str) -> Dict[str, Any]:
        """
        Retrieves the base hyperparameters for a specified model.
        """
        normalized_name = model_name.strip().lower()
        
        if normalized_name not in cls._CONFIGS:
            error_msg = f"Model '{normalized_name}' is not supported. Available: {list(cls._CONFIGS.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"Loaded configuration for model: {normalized_name}")
        return cls._CONFIGS[normalized_name]
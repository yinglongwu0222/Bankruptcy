from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

from .model_configs import get_model_params


def get_model(model_name):
    params = get_model_params(model_name)
    if model_name == "logistic":
        return LogisticRegression(**params)
    elif model_name == "svm":
        return SVC(**params)
    elif model_name == "random_forest":
        return RandomForestClassifier(**params)
    elif model_name == "gbdt":
        return GradientBoostingClassifier(**params)
    elif model_name == "xgboost":
        return xgb.XGBClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_model_params(model_name):
    if model_name == "logistic":
        return {"C": 1.0, "solver": "liblinear", "max_iter": 1000, "random_state": 42}
    elif model_name == "svm":
        return {"C": 1.0, "kernel": "rbf", "probability": True, "random_state": 42}
    elif model_name == "random_forest":
        return {"n_estimators": 100, "criterion": 'gini', "max_depth": 10, "random_state": 42}
    elif model_name == "gbdt":
        return {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 10, "random_state": 42}
    elif model_name == "xgboost":
        return {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 10, "eval_metric": "logloss", "random_state": 42}
    else:
        raise ValueError(f"Unknown model: {model_name}")

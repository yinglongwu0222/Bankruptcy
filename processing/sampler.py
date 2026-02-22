from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTEENN


def apply_sampling(X, y, method=None, random_state=42):
    if method is None or method == "None":
        return X, y
    sampler_map = {
        "RandomOverSampler": RandomOverSampler(random_state=random_state),
        "SMOTE": SMOTE(random_state=random_state),
        "ADASYN": ADASYN(random_state=random_state),
        "NearMiss": NearMiss(),
        "SMOTEENN": SMOTEENN(random_state=random_state),
    }
    sampler = sampler_map.get(method)
    if sampler is None:
        raise ValueError(f"Unknown sampling method: {method}")
    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res

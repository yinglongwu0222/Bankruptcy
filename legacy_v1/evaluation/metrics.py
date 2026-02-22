from sklearn.metrics import f1_score, recall_score, \
    precision_score, roc_auc_score, confusion_matrix, \
    matthews_corrcoef, precision_recall_curve, auc


def compute_metrics(y_true, y_pred, y_score):
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)
    mcc = matthews_corrcoef(y_true, y_pred)

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall_curve, precision_curve)

    cm = confusion_matrix(y_true, y_pred)

    return dict(
        f1=f1,
        recall=recall,
        precision=precision,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        mcc=mcc,
        confusion_matrix=cm
    )

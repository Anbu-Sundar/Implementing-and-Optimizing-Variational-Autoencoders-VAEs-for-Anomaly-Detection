import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

def evaluate(y_true, scores):
    threshold = scores.mean() + 2 * scores.std()
    y_pred = (scores > threshold).astype(int)

    return {
        "ROC_AUC": roc_auc_score(y_true, scores),
        "PR_AUC": average_precision_score(y_true, scores),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred)
    }

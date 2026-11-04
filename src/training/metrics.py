"""Fixed four-class validation metrics; absent classes receive F1=0."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape or not y_true.size:
        raise ValueError("Expected nonempty, equally sized 1D labels/predictions")
    if not np.isin(y_true, range(4)).all() or not np.isin(y_pred, range(4)).all():
        raise ValueError("Labels/predictions must be in [0, 3]")
    precision, recall, scores, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(4)), average=None, zero_division=0,
    )
    return {"macro_f1": float(scores.mean()), "class_f1": scores.tolist(),
            "class_precision": precision.tolist(), "class_recall": recall.tolist(),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(4))).tolist(),
            "accuracy": float(accuracy_score(y_true, y_pred))}

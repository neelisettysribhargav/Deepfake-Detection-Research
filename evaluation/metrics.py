import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def calculate_metrics(targets, predictions):
    """
    Computes all evaluation metrics defined in Section 3.2 and reported in Table 2:
    1. Accuracy (A): (TP + TN) / (TP + FP + TN + FN)
    2. Specificity (Sp): TN / (TN + FP) [Percentage of actual Fake images correctly identified]
    3. Sensitivity (Sn): TP / (TP + FN) [Percentage of actual Real images correctly identified]
    4. F1-score (F1): Harmonic mean of Precision and Recall
    5. Precision (P): TP / (TP + FP)
    6. Recall (R): TP / (TP + FN)
    """
    targets = np.array(targets)
    predictions = np.array(predictions)

    cm = confusion_matrix(targets, predictions, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    acc = accuracy_score(targets, predictions)
    prec = precision_score(targets, predictions, zero_division=0)
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(targets, predictions, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "sensitivity": float(sn),
        "specificity": float(sp),
        "sn": float(sn),
        "sp": float(sp),
        "precision": float(prec),
        "recall": float(sn),
        "f1": float(f1)
    }

    return metrics

def calculate_confusion_matrix(targets, predictions):
    """
    Returns confusion matrix with labels [0, 1] (Fake: 0, Real: 1).
    """
    return confusion_matrix(targets, predictions, labels=[0, 1])
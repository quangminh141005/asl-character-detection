# src/evaluate.py

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def evaluate_on_test(pipe, X_test, y_test):
    """Print test metrics."""
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    w = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n[TEST]")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Macro F1   : {macro:.4f}")
    print(f"Weighted F1: {w:.4f}")
    return y_pred


def print_confusion_matrix(y_test, y_pred, normalize: bool = False):
    """Print a (optionally normalized) confusion matrix."""
    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    if normalize:
        cm = np.round(cm / cm.sum(axis=1, keepdims=True), 3)
        title = "NORMALIZED CONFUSION MATRIX – TEST"
    else:
        title = "CONFUSION MATRIX – TEST"

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{l}" for l in labels],
        columns=[f"pred_{l}" for l in labels],
    )
    print(f"\n[{title}]")
    print(cm_df)
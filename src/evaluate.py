# src/evaluate.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from src.config import PLOTS_DIR


def evaluate_on_test(pipe, X_test, y_test):
    """Print and save test metrics to a text file."""
    y_pred = pipe.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    macro  = f1_score(y_test, y_pred, average="macro",     zero_division=0)
    w      = f1_score(y_test, y_pred, average="weighted",  zero_division=0)
    report = classification_report(y_test, y_pred,         zero_division=0)

    # --- Print to console ---
    lines = [
        "\n[TEST RESULTS]",
        f"Accuracy   : {acc:.4f}",
        f"Macro F1   : {macro:.4f}",
        f"Weighted F1: {w:.4f}",
        "\n[CLASSIFICATION REPORT]",
        report,
    ]
    for line in lines:
        print(line)

    # --- Save to text file ---
    save_path = os.path.join(PLOTS_DIR, "test_metrics.txt")
    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Evaluate] Metrics saved → {save_path}")

    return y_pred


def plot_confusion_matrix(
    y_test,
    y_pred,
    normalize: bool = False,
    filename: str = None,
    figsize: tuple = (16, 14),
    cmap: str = "Blues",
):
    """
    Plot and save a heatmap confusion matrix.

    Args:
        normalize:  If True, rows are normalized to [0, 1].
        filename:   Output filename inside PLOTS_DIR.
                    Defaults to 'confusion_matrix.png' or 'confusion_matrix_normalized.png'.
        figsize:    Figure size (width, height).
        cmap:       Matplotlib colormap name.
    """
    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = "Confusion Matrix (Normalized)"
        default_file = "confusion_matrix_normalized.png"
    else:
        fmt = "d"
        title = "Confusion Matrix"
        default_file = "confusion_matrix.png"

    filename = filename or default_file

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.4,
        linecolor="gray",
        ax=ax,
        cbar_kws={"shrink": 0.75},
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] Confusion matrix saved → {save_path}")
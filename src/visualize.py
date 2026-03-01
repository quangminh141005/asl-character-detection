# src/visualize.py

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.config import HAND_CONNECTIONS, NUM_SAMPLES, NUM_AUG_SAMPLES, ROT_DEG, SCALE_JITTER, NOISE_STD, PLOTS_DIR
from src.augmentation import augment_landmarks_xy


def inspect_data(df: pd.DataFrame):
    """Print a summary of the dataset."""
    print("--- DATASET SUMMARY ---")
    print(f"Total Rows: {len(df)}")
    print(f"Unique Videos: {df['video_id'].nunique()}")
    print("\n--- SAMPLES PER LABEL ---")
    print(df["label"].value_counts())
    print("\n--- FEATURE RANGE ---")
    print(df[["0_x", "0_y", "8_x", "8_y"]].describe().loc[["min", "max"]])


def plot_hand_skeleton(row: pd.Series, ax):
    """Draw a single hand skeleton from a DataFrame row onto an axis."""
    x_data = [row[f"{i}_x"] for i in range(21)]
    y_data = [row[f"{i}_y"] for i in range(21)]
    for conn in HAND_CONNECTIONS:
        ax.plot([x_data[conn[0]], x_data[conn[1]]],
                [y_data[conn[0]], y_data[conn[1]]], "b-", alpha=0.6)
    ax.scatter(x_data, y_data, c="red", s=10)
    ax.scatter(x_data[0], y_data[0], c="green", s=25)
    ax.set_title(f"ID: {row['video_id']}\nLabel: {row['label']} ({row['type']})", fontsize=9)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")


def plot_samples(df: pd.DataFrame, n: int = NUM_SAMPLES, filename: str = "samples.png"):
    """Save n random hand skeletons to PLOTS_DIR."""
    samples = df.sample(n=min(n, len(df)))
    fig, axes = plt.subplots(1, len(samples), figsize=(15, 5))
    if len(samples) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, samples.iterrows()):
        plot_hand_skeleton(row, ax)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Samples saved → {save_path}")


def draw_skeleton(data_vector: np.ndarray, ax, title: str):
    """Draw a skeleton from a flat [x0, y0, x1, y1, ...] vector onto an axis."""
    x_coords = data_vector[0::2]
    y_coords = data_vector[1::2]
    for conn in HAND_CONNECTIONS:
        ax.plot([x_coords[conn[0]], x_coords[conn[1]]],
                [y_coords[conn[0]], y_coords[conn[1]]], "b-", alpha=0.6, linewidth=1)
    ax.scatter(x_coords, y_coords, c="red", s=8)
    ax.scatter(x_coords[0], y_coords[0], c="green", s=20)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")


def plot_augmentations(df: pd.DataFrame, n_copies: int = NUM_AUG_SAMPLES, filename: str = "augmentations.png"):
    """Save original + augmented hand skeletons to PLOTS_DIR."""
    sample_row = df.sample(n=1)
    label = sample_row["label"].values[0]
    vid_id = sample_row["video_id"].values[0]

    feature_cols = []
    for i in range(21):
        feature_cols.extend([f"{i}_x", f"{i}_y"])
    X_original = sample_row[feature_cols].values  # (1, 42)

    augmented = augment_landmarks_xy(X_original, n_copies=n_copies)

    fig, axes = plt.subplots(1, n_copies + 1, figsize=(15, 4))
    draw_skeleton(X_original[0], axes[0], f"ORIGINAL\n{label}\n{vid_id}")
    for i, aug in enumerate(augmented):
        draw_skeleton(aug[0], axes[i + 1], f"Augmented {i + 1}")
    plt.suptitle(f"Augmentation: rot={ROT_DEG}°, scale={SCALE_JITTER}, noise={NOISE_STD}")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Augmentations saved → {save_path}")
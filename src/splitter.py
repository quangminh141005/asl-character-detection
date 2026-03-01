# src/splitter.py

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except ImportError:
    HAS_SGKF = False

from src.config import TEST_SIZE, RANDOM_STATE, N_SPLITS


def make_splits(df: pd.DataFrame, feature_cols: list):
    """
    Returns:
        X_train_full, y_train_full, g_train_full,
        X_test, y_test,
        split_iter (CV iterator on train)
    """
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()
    groups = df["video_id"].to_numpy()

    # Hold-out test split by video_id
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train_full = X[train_idx]
    y_train_full = y[train_idx]
    g_train_full = groups[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    # Sanity check: no video leakage
    assert set(g_train_full).isdisjoint(set(groups[test_idx])), "Video leakage!"

    print(f"[Split] Train: {len(X_train_full)} samples | "
          f"{len(np.unique(g_train_full))} videos")
    print(f"[Split] Test : {len(X_test)} samples | "
          f"{len(np.unique(groups[test_idx]))} videos")

    # CV splitter on train only
    if HAS_SGKF:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        split_iter = list(cv.split(X_train_full, y_train_full, groups=g_train_full))
        print(f"[CV] Using StratifiedGroupKFold (n_splits={N_SPLITS})")
    else:
        cv = GroupKFold(n_splits=N_SPLITS)
        split_iter = list(cv.split(X_train_full, y_train_full, groups=g_train_full))
        print(f"[CV] Using GroupKFold (n_splits={N_SPLITS})")

    return X_train_full, y_train_full, g_train_full, X_test, y_test, split_iter
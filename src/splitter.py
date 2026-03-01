# src/splitter.py

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except ImportError:
    HAS_SGKF = False

from src.config import TEST_SIZE, RANDOM_STATE, N_SPLITS


def make_splits(df: pd.DataFrame, feature_cols: list):
    """
    Group-safe train/test split that GUARANTEES every label has
    at least `min_test_videos` videos in the test set.

    Strategy (Option C — manual per-label reservation):
      1. For each label, randomly pick min_test_videos videos → force into test.
      2. From the remaining videos, fill up to TEST_SIZE using GroupShuffleSplit.
      3. Any video not yet assigned → train.
      4. Build StratifiedGroupKFold CV splits on TRAIN only.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    # ------------------------------------------------------------------ #
    # 1. Per-label minimum video reservation
    # ------------------------------------------------------------------ #
    MIN_TEST_VIDEOS = 2   # guaranteed test videos per label

    # Map: video_id → set of labels that appear in it
    video_labels = (
        df.groupby("video_id")["label"]
        .apply(set)
        .to_dict()
    )
    all_videos = list(video_labels.keys())

    forced_test_videos = set()

    for label in df["label"].unique():
        # videos that contain this label (a video may cover multiple labels)
        candidate_videos = [v for v in all_videos if label in video_labels[v]]

        if len(candidate_videos) < MIN_TEST_VIDEOS:
            raise ValueError(
                f"Label '{label}' only has {len(candidate_videos)} video(s), "
                f"need at least {MIN_TEST_VIDEOS} for test reservation."
            )

        # randomly pick min_test_videos that are not already forced into test
        # prefer videos not yet reserved so we don't over-consume test budget
        not_yet_reserved = [v for v in candidate_videos if v not in forced_test_videos]
        already_reserved  = [v for v in candidate_videos if v in forced_test_videos]

        still_needed = MIN_TEST_VIDEOS - len(already_reserved)
        if still_needed > 0:
            chosen = rng.choice(not_yet_reserved, size=min(still_needed, len(not_yet_reserved)), replace=False)
            forced_test_videos.update(chosen.tolist())

    print(f"[Split] Forced test videos (per-label reservation): {len(forced_test_videos)}")

    # ------------------------------------------------------------------ #
    # 2. Fill remaining test quota from the leftover videos
    # ------------------------------------------------------------------ #
    remaining_videos = [v for v in all_videos if v not in forced_test_videos]

    # how many MORE videos to add to roughly hit TEST_SIZE by frame count
    total_frames      = len(df)
    forced_test_frames = df[df["video_id"].isin(forced_test_videos)].shape[0]
    target_test_frames = int(total_frames * TEST_SIZE)
    extra_frames_needed = max(0, target_test_frames - forced_test_frames)

    # sort remaining by frame count so we can greedily fill
    remaining_frame_counts = (
        df[df["video_id"].isin(remaining_videos)]
        .groupby("video_id")
        .size()
        .reindex(remaining_videos)
        .fillna(0)
        .astype(int)
    )
    shuffled = remaining_frame_counts.sample(frac=1, random_state=RANDOM_STATE)

    extra_test_videos = set()
    running_total = 0
    for vid, count in shuffled.items():
        if running_total >= extra_frames_needed:
            break
        extra_test_videos.add(vid)
        running_total += count

    test_videos  = forced_test_videos | extra_test_videos
    train_videos = set(all_videos) - test_videos

    # ------------------------------------------------------------------ #
    # 3. Build index arrays
    # ------------------------------------------------------------------ #
    test_mask  = df["video_id"].isin(test_videos)
    train_mask = df["video_id"].isin(train_videos)

    train_df = df[train_mask].reset_index(drop=True)
    test_df  = df[test_mask].reset_index(drop=True)

    # Sanity checks
    assert set(train_df["video_id"]).isdisjoint(set(test_df["video_id"])), \
        "Video leakage between train and test!"

    missing_in_test = set(df["label"].unique()) - set(test_df["label"].unique())
    assert not missing_in_test, f"Labels missing from test set: {missing_in_test}"

    print(f"[Split] Train : {len(train_df):>7,} frames | {train_df['video_id'].nunique()} videos")
    print(f"[Split] Test  : {len(test_df):>7,} frames | {test_df['video_id'].nunique()} videos")
    print(f"[Split] Test  labels covered: {sorted(test_df['label'].unique())}")

    X_train_full = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train_full = train_df["label"].to_numpy()
    g_train_full = train_df["video_id"].to_numpy()

    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy()

    # ------------------------------------------------------------------ #
    # 4. CV splits on TRAIN only
    # ------------------------------------------------------------------ #
    if HAS_SGKF:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        split_iter = list(cv.split(X_train_full, y_train_full, groups=g_train_full))
        print(f"[CV] Using StratifiedGroupKFold (n_splits={N_SPLITS})")
    else:
        cv = GroupKFold(n_splits=N_SPLITS)
        split_iter = list(cv.split(X_train_full, y_train_full, groups=g_train_full))
        print(f"[CV] Using GroupKFold (n_splits={N_SPLITS})")

    for fold, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        fold_train_vids = set(g_train_full[tr_idx])
        fold_val_vids   = set(g_train_full[va_idx])
        assert fold_train_vids.isdisjoint(fold_val_vids), f"Video leakage in CV fold {fold}!"

    return X_train_full, y_train_full, g_train_full, X_test, y_test, split_iter
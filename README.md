# ASL Character Detection — RBF SVM on Hand Landmarks

Classify American Sign Language (ASL) hand signs (A–Z + `space` + `del`) from 2D MediaPipe hand landmarks using a **RBF Support Vector Machine**.

---

## Project Structure

```
asl-character-detection/
│
├── data/
│   └── hand_landmarks.csv          # Raw dataset (200,936 frames, 283 videos, 28 labels)
│
├── outputs/
│   ├── models/
│   │   └── rbf_svm_no_aug.joblib   # Saved final model
│   └── plots/
│       ├── samples.png                      # Random sample skeletons
│       ├── augmentations.png                # Augmentation preview
│       ├── test_metrics.txt                 # Accuracy, F1, classification report
│       ├── confusion_matrix.png             # Raw count heatmap
│       └── confusion_matrix_normalized.png  # Row-normalized heatmap
│
├── src/
│   ├── __init__.py
│   ├── config.py          # All constants & hyperparameters
│   ├── data_loader.py     # Load & clean CSV
│   ├── augmentation.py    # On-the-fly rotation/scale/noise
│   ├── splitter.py        # Group-safe train/test/CV splits
│   ├── visualize.py       # Skeleton plots → saved to disk
│   ├── train_svm.py       # SVM pipeline, CV loop, final training
│   └── evaluate.py        # Metrics, classification report, confusion matrix heatmaps
│
├── main.py                # Entry point — runs the full pipeline
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/quangminh141005/test_dump.git
cd asl-character-detection

pip install -r requirements.txt
```

Place your dataset at `data/hand_landmarks.csv`.

---

## Usage

Run the full pipeline with:

```bash
python main.py
```

This will:
1. Load and inspect the dataset
2. Save skeleton and augmentation preview plots
3. Split data by `video_id` (no leakage)
4. Run 5-fold cross-validation
5. Train the final model on all training data
6. Evaluate on the held-out test set
7. Save confusion matrix graphs and metrics

---

## Pipeline Overview

```
hand_landmarks.csv
        │
        ▼
 ┌─────────────────┐
 │   Data Loading  │  Drop NaNs, extract 42 landmark features (0_x/0_y … 20_x/20_y)
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │   EDA & Plots   │  Save skeleton samples + augmentation preview to outputs/plots/
 └────────┬────────┘
          │
          ▼
 ┌──────────────────────────────────────┐
 │   Group Split by video_id            │
 │                                      │
 │   GroupShuffleSplit (test_size=0.2)  │
 │   ├── TRAIN_FULL  (~80% of videos)   │
 │   └── TEST        (~20% of videos)   │  ← held out, touched ONCE
 └────────┬─────────────────────────────┘
          │
          ▼
 ┌──────────────────────────────────────┐
 │   StratifiedGroupKFold (k=5)         │  on TRAIN_FULL only
 │   Fold 1 … Fold 5                    │  groups = video_id
 └────────┬─────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────┐
 │   SVM Pipeline (per fold)           │
 │   StandardScaler → SVC(RBF)         │
 │   C=10, gamma='scale',              │
 │   class_weight='balanced'           │
 └────────┬────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────┐
 │   Final Training on TRAIN_FULL      │  saved → outputs/models/rbf_svm_no_aug.joblib
 └────────┬────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────┐
 │   Evaluation on TEST                │
 │   • test_metrics.txt                │
 │   • confusion_matrix.png            │
 │   • confusion_matrix_normalized.png │
 └─────────────────────────────────────┘
```

---

## Dataset

| Property | Value |
|---|---|
| Total frames | 200,936 |
| Unique videos | 283 |
| Labels | 28 (A–Z + `space` + `del`) |
| Features | 42 (x, y coords of 21 MediaPipe hand landmarks) |
| Augmentation (in CSV) | Each frame has an `original` and `flipped` copy |

Landmarks are normalized so the **wrist (landmark 0) is at the origin (0, 0)**.

---

## Model

**RBF Support Vector Machine** via `sklearn.svm.SVC`

| Hyperparameter | Value | Reason |
|---|---|---|
| `kernel` | `rbf` | Non-linear boundaries between similar signs |
| `C` | `10.0` | Moderate regularization |
| `gamma` | `scale` | Auto-set: 1 / (n_features × var(X)) |
| `class_weight` | `balanced` | Compensates for label imbalance |

---

## Results

### Cross-Validation (5-fold, StratifiedGroupKFold)

| Fold | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| 1 | 0.7974 | 0.6785 | 0.8071 |
| 2 | 0.7736 | 0.7491 | 0.7784 |
| 3 | 0.8230 | 0.6579 | 0.8359 |
| 4 | 0.8611 | 0.6612 | 0.8707 |
| 5 | 0.7911 | 0.6538 | 0.8019 |
| **Mean** | **0.8093 ± 0.030** | **0.6801 ± 0.036** | **0.8188 ± 0.032** |

### Test Set (held-out)

| Metric | Score |
|---|---|
| Accuracy | **0.8154** |
| Macro F1 | **0.7046** |
| Weighted F1 | **0.8286** |

### Notable Confusions

| True | Predicted | Recall | Reason |
|---|---|---|---|
| T | A | ~38% misclassified | Closed fist looks similar |
| J | I | ~39% misclassified | J is I in motion |
| K | H | ~15% misclassified | Overlapping joint positions |
| R | U | ~23% misclassified | Crossed vs. parallel fingers |

---

## Key Design Decisions

### Why group-split by `video_id`?
Consecutive frames from the same video are nearly identical. Splitting randomly would leak near-duplicate frames into both train and test, giving falsely high accuracy.

### Why `StratifiedGroupKFold`?
- **Group**: keeps all frames from a video in the same fold — no leakage within CV
- **Stratified**: ensures each fold has a balanced mix of all 28 labels

### Why SVM over a neural network?
- Features are already compact (42 floats) and normalized — no spatial or temporal structure to exploit
- SVM trains in minutes and achieves >81% accuracy
- Easily interpretable and deployable with `joblib`

---

## Configuration

All hyperparameters live in `src/config.py` — no magic numbers elsewhere:

```python
# Augmentation
NOISE_STD    = 0.01
ROT_DEG      = 8.0
SCALE_JITTER = 0.05

# Split
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
N_SPLITS       = 5

# SVM
SVM_C            = 10.0
SVM_KERNEL       = "rbf"
SVM_GAMMA        = "scale"
SVM_CLASS_WEIGHT = "balanced"
```

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

This model is used in the following project:  
👉 **[ASL Detection Website](https://github.com/Tuan-Nguyen-Minhh/web-american-sign-language)**  




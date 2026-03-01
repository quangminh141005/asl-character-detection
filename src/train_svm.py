# src/train_svm.py

import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from src.config import SVM_C, SVM_KERNEL, SVM_GAMMA, SVM_CLASS_WEIGHT


def build_svm_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=SVM_KERNEL,
            C=SVM_C,
            gamma=SVM_GAMMA,
            class_weight=SVM_CLASS_WEIGHT,
        )),
    ])


def cross_validate(X_train_full, y_train_full, split_iter):
    """Run CV and print per-fold metrics. Returns mean accuracy."""
    accs, macro_f1s, weighted_f1s = [], [], []
    pipe = build_svm_pipeline()

    for fold, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        X_tr, y_tr = X_train_full[tr_idx], y_train_full[tr_idx]
        X_va, y_va = X_train_full[va_idx], y_train_full[va_idx]

        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_va)

        acc = accuracy_score(y_va, y_pred)
        macro = f1_score(y_va, y_pred, average="macro", zero_division=0)
        w = f1_score(y_va, y_pred, average="weighted", zero_division=0)

        accs.append(acc)
        macro_f1s.append(macro)
        weighted_f1s.append(w)

        print(f"Fold {fold}: acc={acc:.4f}  macroF1={macro:.4f}  wF1={w:.4f} "
              f"| train={len(tr_idx)} val={len(va_idx)}")

    print(f"\n[CV Summary]")
    print(f"Accuracy   : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Macro F1   : {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
    print(f"Weighted F1: {np.mean(weighted_f1s):.4f} ± {np.std(weighted_f1s):.4f}")
    return np.mean(accs)


def train_final(X_train_full, y_train_full, save_path: str = "outputs/models/rbf_svm_no_aug.joblib"):
    """Train on full training set and save the model."""
    pipe = build_svm_pipeline()
    pipe.fit(X_train_full, y_train_full)
    joblib.dump(pipe, save_path)
    print(f"[Model] Saved to {save_path}")
    return pipe
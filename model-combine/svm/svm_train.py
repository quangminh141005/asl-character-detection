import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

CSV_PATH   = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/dataset-maker-for-svm/asl_svm_dataset.csv"
MODEL_PATH = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/model-combine/svm_asl_model.joblib"  # saved to Drive

print("[INFO] Loading dataset:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

y = df["label"].values
X = df.drop(columns=["label"]).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=10, gamma="scale")
)

print("[INFO] Training SVM...")
clf.fit(X_train, y_train)

print("Train accuracy:", clf.score(X_train, y_train))
print("Test accuracy :", clf.score(X_test, y_test))

y_pred = clf.predict(X_test)
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(clf, MODEL_PATH)
print("[INFO] Model saved to:", MODEL_PATH)

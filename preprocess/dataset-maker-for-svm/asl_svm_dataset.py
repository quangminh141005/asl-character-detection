import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch

from mmpose.apis import init_model, inference_topdown
from mmengine.model.utils import revert_sync_batchnorm

POSE_CONFIG = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/model-combine/pose/rtmpose-m-hand-256x256.py"
POSE_CKPT   = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/model-combine/pose/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth"
DEVICE      = "cpu"
DATA_ROOT   = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/asl_alphabet_train/asl_alphabet_train"  # chỉnh lại path
OUTPUT_CSV  = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/dataset-maker-for-svm/asl_svm_dataset.csv"

print("[INFO] Initializing RTMPose model...")
pose_model = init_model(POSE_CONFIG, POSE_CKPT, device=DEVICE)
pose_model = revert_sync_batchnorm(pose_model)

def extract_hand_keypoints(img_bgr):
    h, w = img_bgr.shape[:2]
    bboxes = np.array([[0, 0, w, h]], dtype=np.float32)

    with torch.no_grad():
        result = inference_topdown(pose_model, img_bgr, bboxes=bboxes)

    if result is None or len(result) == 0:
        print("[WARN] inference_topdown returned no result")
        return None

    data_sample = result[0]

    # New style: DataSample with pred_instances
    if hasattr(data_sample, "pred_instances"):
        keypoints = data_sample.pred_instances.keypoints
        keypoints = np.asarray(keypoints)
        if keypoints.ndim == 3:
            keypoints = keypoints[0]
        return keypoints

    # Old style: dict with 'keypoints'
    if isinstance(data_sample, dict) and "keypoints" in data_sample:
        keypoints = np.asarray(data_sample["keypoints"])
        if keypoints.ndim == 3:
            keypoints = keypoints[0]
        return keypoints

    print("[WARN] Unknown result format:", type(data_sample))
    return None

def normalize_keypoints(keypoints):
    wrist = keypoints[0].copy()
    coords = keypoints - wrist
    dists = np.linalg.norm(coords, axis=1)
    scale = dists.max()
    if scale < 1e-6:
        scale = 1.0
    coords = coords / scale
    return coords

X, y = [], []

for label_name in sorted(os.listdir(DATA_ROOT)):
    class_dir = os.path.join(DATA_ROOT, label_name)
    if not os.path.isdir(class_dir):
        continue

    print(f"[INFO] Processing label '{label_name}'...")

    img_paths = glob.glob(os.path.join(class_dir, "*.jpg")) + \
                glob.glob(os.path.join(class_dir, "*.jpeg")) + \
                glob.glob(os.path.join(class_dir, "*.png"))

    print(f"[INFO]  Found {len(img_paths)} images for label '{label_name}'")

    for idx, img_path in enumerate(img_paths, start=1):
        print(f"[INFO]  ({label_name}) Image {idx}/{len(img_paths)}: {os.path.basename(img_path)}")

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue

        keypoints = extract_hand_keypoints(img)
        if keypoints is None:
            print(f"[WARN] No keypoints detected in: {img_path}")
            continue

        print(f"[DEBUG] keypoints shape: {keypoints.shape}")

        kp_norm = normalize_keypoints(keypoints)
        feat = kp_norm.reshape(-1)

        X.append(feat)
        y.append(label_name)

if len(X) == 0:
    print("[ERROR] No samples collected. Check if RTMPose detects anything on your images.")
else:
    X = np.stack(X, axis=0)
    num_kp = X.shape[1] // 2

    cols = []
    for i in range(num_kp):
        cols += [f"x{i}", f"y{i}"]

    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "label", y)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Saved dataset with {len(df)} samples to: {OUTPUT_CSV}")

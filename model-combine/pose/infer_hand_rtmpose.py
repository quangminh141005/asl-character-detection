import cv2
import numpy as np
from ultralytics import YOLO

from mmpose.apis import init_model, inference_topdown
from mmengine.model.utils import revert_sync_batchnorm

# ======= PATHS YOU EDIT =======
IMG_PATH = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/model-combine/data/test2.jpg"
YOLO_WEIGHTS = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/model-combine/detection/yolov8.pt"

POSE_CONFIG = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/model-combine/pose/rtmpose-m-hand-256x256.py"
POSE_CKPT = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/model-combine/pose/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth"
DEVICE = "cpu"  # or "cuda:0"
# ==============================

# 1. Load YOLO model
det_model = YOLO(YOLO_WEIGHTS)

# 2. Load RTMPose model
pose_model = init_model(POSE_CONFIG, POSE_CKPT, device=DEVICE)
pose_model = revert_sync_batchnorm(pose_model)  # optional

# 3. Run YOLO to get hand boxes
yolo_results = det_model(IMG_PATH)[0]
bboxes_xyxy = []

for box in yolo_results.boxes:
    conf = float(box.conf[0])
    if conf < 0.3:
        continue
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
    bboxes_xyxy.append([x1, y1, x2, y2])

if not bboxes_xyxy:
    print("No hands detected.")
    exit()

print(f"YOLO found {len(bboxes_xyxy)} hand(s).")

# 4. Run RTMPose top-down inference
pose_results = inference_topdown(
    pose_model,
    IMG_PATH,
    bboxes_xyxy,
    bbox_format='xyxy'
)

print(f"RTMPose returned {len(pose_results)} pose sample(s).")

# 5. Extract keypoints and visualize
img = cv2.imread(IMG_PATH)

for i, res in enumerate(pose_results):
    # res is PoseDataSample
    kpts = res.pred_instances.keypoints  # (1, K, 2) tensor or ndarray

    if hasattr(kpts, 'detach'):
        kpts = kpts.detach().cpu().numpy()
    else:
        kpts = np.asarray(kpts)

    kpts = kpts[0]  # (K, 2)

    print(f"Hand {i} keypoints shape: {kpts.shape}")
    print(kpts)

    # draw keypoints
    for (x, y) in kpts:
        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

# draw YOLO boxes
for (x1, y1, x2, y2) in bboxes_xyxy:
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                  (255, 0, 0), 2)

out_path = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/model-combine/data/test_hand_pose.jpg"
cv2.imwrite(out_path, img)
print(f"Saved visualization to {out_path}")

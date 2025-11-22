import cv2
import numpy as np
import time
from ultralytics import YOLO

from mmpose.apis import init_model, inference_topdown
from mmengine.model.utils import revert_sync_batchnorm

# ==========================
# PATHS & SETTINGS
# ==========================
YOLO_WEIGHTS = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/model-combine/detection/yolov8.pt"

POSE_CONFIG = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/model-combine/pose/rtmpose-m-hand-256x256.py"
POSE_CKPT = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/model-combine/pose/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth"

DEVICE = "cpu"   # change to "cuda:0" if you have a GPU
CONF_THRES = 0.3
WEBCAM_INDEX = 0

# ==========================
# Hand Skeleton Definition (RTMPose 21 keypoints)
# 0: wrist
# 1–4: thumb, 5–8: index, 9–12: middle, 13–16: ring, 17–20: pinky
# ==========================
SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # pinky
]

# ==========================
# Load models
# ==========================
print("[INFO] Loading YOLO model...")
det_model = YOLO(YOLO_WEIGHTS)

print("[INFO] Loading RTMPose model...")
pose_model = init_model(POSE_CONFIG, POSE_CKPT, device=DEVICE)
pose_model = revert_sync_batchnorm(pose_model)  # important for CPU

# ==========================
# Open webcam
# ==========================
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam")
    exit()

print("[INFO] Press 'q' to quit")

# ==========================
# FPS init
# ==========================
fps = 0.0
prev_time = time.perf_counter()

while True:
    # ---- Grab frame ----
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Failed to grab frame")
        break

    img_vis = frame.copy()
    bboxes_xyxy = []

    # ---- Timing: start of frame ----
    frame_start = time.perf_counter()

    # ==========================
    # 1. YOLO Hand Detection
    # ==========================
    t0 = time.perf_counter()
    yolo_results = det_model(frame, verbose=False)[0]
    t1 = time.perf_counter()
    yolo_time = (t1 - t0) * 1000.0  # ms

    for box in yolo_results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        bboxes_xyxy.append([x1, y1, x2, y2])

    # ==========================
    # 2. Pose Estimation with RTMPose
    # ==========================
    pose_time = 0.0
    if bboxes_xyxy:
        t2 = time.perf_counter()
        pose_results = inference_topdown(
            pose_model,
            img_vis,
            bboxes_xyxy,
            bbox_format='xyxy'
        )
        t3 = time.perf_counter()
        pose_time = (t3 - t2) * 1000.0  # ms

        # ---- Draw keypoints & skeleton ----
        for res in pose_results:
            kpts = res.pred_instances.keypoints  # shape: (1, K, 2)

            # tensor -> numpy
            if hasattr(kpts, 'detach'):
                kpts = kpts.detach().cpu().numpy()
            else:
                kpts = np.asarray(kpts)

            kpts = kpts[0]  # (K, 2)

            # Draw keypoints
            for (x, y) in kpts:
                cv2.circle(img_vis, (int(x), int(y)), 3, (0, 255, 0), -1)

            # Draw skeleton lines
            for a, b in SKELETON:
                x1, y1 = kpts[a]
                x2, y2 = kpts[b]
                cv2.line(
                    img_vis,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 255),
                    2
                )

    # ==========================
    # 3. Draw YOLO boxes
    # ==========================
    for (x1, y1, x2, y2) in bboxes_xyxy:
        cv2.rectangle(
            img_vis,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 0, 0),
            2
        )

    # ==========================
    # 4. FPS Calculation
    # ==========================
    frame_end = time.perf_counter()
    frame_time = frame_end - frame_start
    inst_fps = 1.0 / frame_time if frame_time > 0 else 0.0

    # smooth FPS using exponential moving average
    fps = fps * 0.9 + inst_fps * 0.1

    # ==========================
    # 5. Overlay Text: FPS + timings
    # ==========================
    cv2.putText(
        img_vis,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        img_vis,
        f"YOLO: {yolo_time:.1f} ms",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.putText(
        img_vis,
        f"Pose: {pose_time:.1f} ms",
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    # you can also print to console if you like
    # print(f"FPS: {fps:.1f}, YOLO: {yolo_time:.1f} ms, Pose: {pose_time:.1f} ms")

    # ==========================
    # 6. Show window
    # ==========================
    cv2.imshow("Hand Pose (YOLO + RTMPose)", img_vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================
# Cleanup
# ==========================
cap.release()
cv2.destroyAllWindows()

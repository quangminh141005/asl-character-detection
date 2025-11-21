import cv2
import numpy as np
from ultralytics import YOLO

from mmpose.apis import init_model, inference_topdown
from mmengine.model.utils import revert_sync_batchnorm

# PATHS & SETTINGS
YOLO_WEIGHTS = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/model-combine/detection/yolov8.pt"

POSE_CONFIG = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/model-combine/pose/rtmpose-m-hand-256x256.py"
POSE_CKPT = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/model-combine/pose/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth"

DEVICE = "cpu"  # change to "cuda:0" if cpu
CONF_THRES = 0.3
WEBCAM_INDEX = 0


# 1. Load YOLO model
print("[INFO] Loading YOLO model...")
det_model = YOLO(YOLO_WEIGHTS)

# 2. Load RTMPose model
print("[INFO] Loading RTMPose model...")
pose_model = init_model(POSE_CONFIG, POSE_CKPT, device=DEVICE)
pose_model = revert_sync_batchnorm(pose_model)  # for CPU compatibility

# 3. Open webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam")
    exit()

print("[INFO] Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Failed to grab frame")
        break

    # Optional: you can resize frame to speed up
    # frame = cv2.resize(frame, (640, 480))

    # 4. Run YOLO detection on the frame
    yolo_results = det_model(frame, verbose=False)[0]

    bboxes_xyxy = []
    for box in yolo_results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        bboxes_xyxy.append([x1, y1, x2, y2])

    img_vis = frame.copy()

    # 5. If any hands, run RTMPose
    if bboxes_xyxy:
        pose_results = inference_topdown(
            pose_model,
            img_vis,          # pass the current frame instead of path
            bboxes_xyxy,
            bbox_format='xyxy'
        )

        # 6. Draw keypoints for each detected hand
        for res in pose_results:
            kpts = res.pred_instances.keypoints  # (1, K, 2)

            # tensor -> numpy
            if hasattr(kpts, 'detach'):
                kpts = kpts.detach().cpu().numpy()
            else:
                kpts = np.asarray(kpts)

            kpts = kpts[0]  # (K, 2)

            for (x, y) in kpts:
                cv2.circle(img_vis, (int(x), int(y)), 3, (0, 255, 0), -1)

    # 7. Draw YOLO boxes on top
    for (x1, y1, x2, y2) in bboxes_xyxy:
        cv2.rectangle(
            img_vis,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 0, 0),
            2
        )

    # 8. Show real-time window
    cv2.imshow("Hand Pose (YOLO + RTMPose)", img_vis)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9. Cleanup
cap.release()
cv2.destroyAllWindows()

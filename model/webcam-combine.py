#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
from ultralytics import YOLO

# =====================================================
# CONFIG
# =====================================================
YOLO_WEIGHTS = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/model/detection/hqm_hand_palm.pt"
LANDMARK_WEIGHTS = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/model/pose/mobilenetv3-lan3.pt"

CAMERA_ID = 0

IN_SIZE = 224
HM_SIZE = 56
NUM_JOINTS = 21

YOLO_CONF = 0.05
YOLO_IMGSZ = 640
YOLO_EVERY_N = 3
MAX_HANDS = 1

PAD = 0.30               # extra bbox padding around YOLO box
SMOOTH_ALPHA = 0.7       # EMA smoothing
DRAW_IDS = False         # draw landmark index numbers
DEBUG_SAVE_INPUT = True # save debug crops to disk

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, flush=True)

SKELETON = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# ImageNet normalization (matches your dataset)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

# =====================================================
# MODEL (same architecture you trained)
# =====================================================
class MobileNetV3Heatmap(nn.Module):
    def __init__(self, num_joints=21, hm_size=56, pretrained=True):
        super().__init__()
        self.hm_size = hm_size
        weights = "DEFAULT" if pretrained else None
        base = torchvision.models.mobilenet_v3_large(weights=weights)
        self.backbone = base.features
        c = 960
        self.head = nn.Sequential(
            nn.Conv2d(c, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, num_joints, 1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return nn.functional.interpolate(
            x, size=(self.hm_size, self.hm_size),
            mode="bilinear", align_corners=False
        )

@torch.no_grad()
def heatmaps_to_coords_argmax(hm: torch.Tensor) -> torch.Tensor:
    """
    hm: [1,J,HM,HM] -> coords_hm: [J,2] in HM index space (x,y)
    """
    hm = hm[0]  # [J,H,W]
    J, H, W = hm.shape
    flat = hm.view(J, -1)
    idx = flat.argmax(dim=1)
    y = (idx // W).float()
    x = (idx %  W).float()
    return torch.stack([x, y], dim=1)

def preprocess_crop_like_training(crop_bgr: np.ndarray) -> torch.Tensor:
    """
    Match your FreiHAND dataset:
    - BGR -> RGB
    - resize to (224,224)
    - /255
    - ImageNet normalize
    """
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.resize(crop_rgb, (IN_SIZE, IN_SIZE), interpolation=cv2.INTER_AREA)

    x = torch.from_numpy(crop_rgb).float().to(device) / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)  # [1,3,224,224]
    x = (x - MEAN) / STD
    return x

def expand_and_clip_box(x1, y1, x2, y2, W, H, pad=0.3):
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    x1 = int(x1 - pad * bw); y1 = int(y1 - pad * bh)
    x2 = int(x2 + pad * bw); y2 = int(y2 + pad * bh)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W - 1, x2); y2 = min(H - 1, y2)
    return x1, y1, x2, y2

def draw_pose(frame, pts, skeleton=SKELETON):
    # skeleton
    for a, b in skeleton:
        pa = tuple(pts[a].astype(int))
        pb = tuple(pts[b].astype(int))
        cv2.line(frame, pa, pb, (255, 255, 255), 2, cv2.LINE_AA)
    # points
    for i, (x, y) in enumerate(pts):
        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1, cv2.LINE_AA)
        if DRAW_IDS:
            cv2.putText(frame, str(i), (int(x)+4, int(y)-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1, cv2.LINE_AA)

# =====================================================
# LOAD MODELS
# =====================================================
print("Loading YOLO...", flush=True)
yolo = YOLO(YOLO_WEIGHTS)
print("YOLO loaded. classes:", yolo.names, flush=True)

print("Loading landmark model...", flush=True)
landmark_model = MobileNetV3Heatmap(num_joints=NUM_JOINTS, hm_size=HM_SIZE, pretrained=False).to(device)

ckpt = torch.load(LANDMARK_WEIGHTS, map_location=device)
state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
landmark_model.load_state_dict(state, strict=True)
landmark_model.eval()
print("Landmark model loaded.", flush=True)

# =====================================================
# WEBCAM
# =====================================================
print("Opening webcam...", flush=True)
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Try CAMERA_ID=1 or check permissions.")

cv2.namedWindow("Hand Pose", cv2.WINDOW_NORMAL)
print("Starting loop. Press 'q' to quit.", flush=True)

frame_idx = 0
last_boxes = None
smoothed_pts = None

prev_time = time.time()
fps = 0.0

with torch.no_grad():
    while True:
        ok, frame = cap.read()
        if not ok:
            print("❌ Failed to read frame from webcam", flush=True)
            break

        frame_idx += 1
        H, W = frame.shape[:2]

        # YOLO every N frames
        if (frame_idx % YOLO_EVERY_N == 0) or (last_boxes is None):
            r = yolo.predict(frame, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, verbose=False)[0]
            last_boxes = r.boxes

        if last_boxes is not None and len(last_boxes) > 0:
            # pick highest confidence box
            confs = last_boxes.conf.detach().cpu().numpy()
            i = int(np.argmax(confs))

            x1, y1, x2, y2 = last_boxes.xyxy[i].detach().cpu().numpy().astype(int)
            x1, y1, x2, y2 = expand_and_clip_box(x1, y1, x2, y2, W, H, pad=PAD)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                if DEBUG_SAVE_INPUT:
                    os.makedirs("debug", exist_ok=True)
                    cv2.imwrite("debug/crop_raw.png", crop)

                x = preprocess_crop_like_training(crop)  # [1,3,224,224]

                hm = landmark_model(x)  # [1,21,56,56]
                if isinstance(hm, (tuple, list)):
                    hm = hm[0]

                coords_hm = heatmaps_to_coords_argmax(hm).cpu().numpy()  # [21,2] in HM indices

                # HM -> 224 input space (use HM_SIZE - 1 for correct scaling)
                coords_224 = coords_hm * (IN_SIZE / (HM_SIZE - 1))

                # 224 -> crop space (original crop w/h)
                cw, ch = (x2 - x1), (y2 - y1)
                xs = coords_224[:, 0] * (cw / IN_SIZE) + x1
                ys = coords_224[:, 1] * (ch / IN_SIZE) + y1
                pts = np.stack([xs, ys], axis=1)

                # smooth
                if smoothed_pts is None:
                    smoothed_pts = pts
                else:
                    smoothed_pts = SMOOTH_ALPHA * smoothed_pts + (1.0 - SMOOTH_ALPHA) * pts

                # draw bbox + pose
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                draw_pose(frame, smoothed_pts)

        # FPS
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 1.0 / max(dt, 1e-6)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Hand Pose", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print("Done.", flush=True)

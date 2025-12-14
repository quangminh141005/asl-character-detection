import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from ultralytics import YOLO
import time

# =====================================================
# CONFIG
# =====================================================
YOLO_WEIGHTS = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/model/detection/hqm_hand_palm.pt"
LANDMARK_WEIGHTS = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/model/pose/mobilenetv3-lan1.pt"

CAMERA_ID = 0

IN_SIZE = 224
HM_SIZE = 56

YOLO_CONF = 0.05
YOLO_IMGSZ = 960
MAX_HANDS = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# =====================================================
# LANDMARK MODEL (EXACT MATCH)
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
        x = nn.functional.interpolate(
            x, size=(self.hm_size, self.hm_size),
            mode="bilinear", align_corners=False
        )
        return x


@torch.no_grad()
def heatmaps_to_coords(hm):
    hm = hm[0]
    J, H, W = hm.shape
    flat = hm.view(J, -1)
    idx = flat.argmax(dim=1)
    y = idx // W
    x = idx % W
    return torch.stack([x, y], dim=1).float()


# =====================================================
# PREPROCESS
# =====================================================
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

def preprocess_crop(crop_bgr):
    crop = cv2.resize(crop_bgr, (IN_SIZE, IN_SIZE))
    x = torch.from_numpy(crop).float().to(device) / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    return (x - MEAN) / STD


# =====================================================
# LOAD MODELS
# =====================================================
yolo = YOLO(YOLO_WEIGHTS)
print("YOLO classes:", yolo.names)

landmark_model = MobileNetV3Heatmap(
    num_joints=21, hm_size=HM_SIZE, pretrained=False
).to(device)

ckpt = torch.load(LANDMARK_WEIGHTS, map_location=device)
state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
landmark_model.load_state_dict(state)
landmark_model.eval()
print("Landmark model loaded ✔")

# =====================================================
# WEBCAM
# =====================================================
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press 'q' to quit")

prev = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # -------------------------------------------------
    # 1) YOLO (USE BGR DIRECTLY — SAME AS WORKING SCRIPT)
    # -------------------------------------------------
    results = yolo.predict(
        frame,
        conf=YOLO_CONF,
        imgsz=YOLO_IMGSZ,
        verbose=False
    )[0]

    # ALWAYS show YOLO raw output
    vis = results.plot()

    # Debug print
    n = 0 if results.boxes is None else len(results.boxes)
    print(f"YOLO boxes: {n}", end="\r")

    # -------------------------------------------------
    # 2) LANDMARKS (ONLY IF YOLO FOUND A HAND)
    # -------------------------------------------------
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes
        confs = boxes.conf.cpu().numpy()
        order = np.argsort(-confs)

        for i in order[:MAX_HANDS]:
            x1,y1,x2,y2 = boxes.xyxy[i].cpu().numpy().astype(int)

            # padding
            bw, bh = x2-x1, y2-y1
            pad = 0.25
            x1 = max(0, int(x1 - pad*bw))
            y1 = max(0, int(y1 - pad*bh))
            x2 = min(frame.shape[1]-1, int(x2 + pad*bw))
            y2 = min(frame.shape[0]-1, int(y2 + pad*bh))

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            x = preprocess_crop(crop)
            hm = landmark_model(x)
            coords_hm = heatmaps_to_coords(hm).cpu().numpy()

            coords_in = coords_hm * (IN_SIZE / HM_SIZE)
            cw, ch = x2-x1, y2-y1
            xs = coords_in[:,0] * (cw / IN_SIZE) + x1
            ys = coords_in[:,1] * (ch / IN_SIZE) + y1

            # draw landmarks ON TOP OF YOLO VIS
            for px, py in zip(xs, ys):
                cv2.circle(vis, (int(px), int(py)), 3, (0,0,255), -1)

    # -------------------------------------------------
    # FPS
    # -------------------------------------------------
    now = time.time()
    fps = 1.0 / (now - prev)
    prev = now
    cv2.putText(
        vis, f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0,255,0),
        2
    )

    cv2.imshow("YOLO + LANDMARK DEBUG", vis)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print()

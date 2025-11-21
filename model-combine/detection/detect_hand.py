# detect_hand.py
import cv2
from ultralytics import YOLO
import json
import os


# 1. Load YOLO model
model = YOLO("/home/qminh/Documents/qm/USTH/COURSES/B3/Project/asl-character-detection/model-combine/detection/yolov8.pt")   # <-- your trained hand detector

# 2. Load image
image_path = "data/test.png"   # <-- input image
img = cv2.imread(image_path)

results = model(image_path)[0]   # predict on image

# 3. Extract bboxes (xyxy)
hand_bboxes = []   # will store list of [x1, y1, x2, y2]

for box in results.boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
    conf = float(box.conf[0])

    if conf < 0.3:  # threshold, adjust if needed
        continue

    # Append xyxy bbox
    hand_bboxes.append([x1, y1, x2, y2])

# 4. Print bboxes
print("Detected hand boxes (xyxy):")
for b in hand_bboxes:
    print(b)

# 5. Visualize YOLO detection(optional)
for (x1, y1, x2, y2) in hand_bboxes:
    cv2.rectangle(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 255, 0), 2
    )

cv2.imwrite("data/test_detected.jpg", img)
print("Saved: data/test_detected.jpg")

# 6. save to JSON (optional)
os.makedirs("bboxes", exist_ok=True)
json_path = "bboxes/test_bbox.json"

with open(json_path, "w") as f:
    json.dump({"bboxes": hand_bboxes}, f, indent=2)

print("Saved YOLO bboxes to:", json_path)

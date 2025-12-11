import argparse
import cv2 
import torch
import numpy as np

from ultralytics import yolo
from torchvision import models, transforms
import torch.nn as nn

# 1. Landmark Model Wrapper
class MobileNetV3HandLandmarks(nn.Module):
    """
    - input: 3x224x224 image
    - output: num_landmarks * 2 * (x, y), same as in training:
        model = model.mobilenet_v3_small(pretrained=True)
        model.classifier[-1] = nn.Linear(in_features, 2 * num_keypoints)
    """

    def __init__(self, num_landmarks: int = 21):
        super().__init__()
        self.num_landmarks = num_landmarks

        # architechture as training
        backbone = models.mobilenet_v3_small(pretrained=False)

        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, 2 * num_landmarks)

        self.net = backbone

    def foward(self, x):
        return self.net(x)
    
# 2. preprocessing and postprocessing
def get_transform():
    # Standard image normalization for MobileNetV3
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ReSize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def denorm_landmark(output, num_landmarks, crop_box):
    """
    Convert normalized landmark prediction back to the original value

    Args:
        output: tensor of shape (num_landmarks * 2) with values in [0, 1]
        num_landmarks: number of landmarks
        crop_box: (x1, y1, x2, y2) in original image coordinates

    Returns:
        np.array of shape (num_landmarks, 2) with pixel coordinates (x, y)
    """

    x1, y1, x2, y2 = crop_box
    w = x2 - x1
    h = y2 - y1
    coords = output.view(num_landmarks, 2).detach().cpu().numpy()
    xs = coords[:, 0] * w + x1
    ys = coords[:, 1] * h + y1

    return np.stack([xs, ys], axis=1)

def expand_box(x1, y1, x2, y2, img_w, img_h, scale=0.2):
    """
    Expand boudning box by same raito and clamp to image size
    """
    w = x2 - x1
    h = y2 - y1 
    cx = x1 + w / 2
    cy = y1 + h / 2

    new_w = w * (1 + scale)
    new_h = h * (1 + scale)

    nx1 = max(0, int(cx - new_w / 2))
    nx2 = max(0, int(cy - new_h / 2))
    nx2 = min(img_w - 1, int(cx + new_w / 2))
    ny2 = min(img_h - 1, int(cy + new_h / 2))


# 3. Drawing utilities 
def draw_landmarks(frame, landmarksm, color=(0, 255, 0), radius=2):
    """
    Draw landmarks on the frame
    landmarks: np.ndarray [N, 2]
    """
    for x, y in landmarks:
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)


def draw_boxes(frame, boxes, color=(255, 0, 0), thickness=2):
    for (x1, y1, x2, y2) in boxes:
        cv2.regtangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

# 4. drawing inference loop

def run(
    yolo_model_path: str = "",
    landmark_model_path: str="",
    source: str = "0",
    num_landmarks: int = 21,
    conf_thres: float = 0.5,
    device: str = None,
):
    # Check device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load yolo11 (hand palm detection)
    print(f"Loading yolo11 right now from {yolo_model_path}...")
    yolo = YOLO(yolo_model_path)

    # Load MobileNetV3 (hand landmark estimation - will test more models later on)
    print(f"Loading MobileNetV3 from {landmark_model_path}...")
    landmark_model = MobileNetV3HandLandmarks(num_landmarks=num_landmarks)

    # Load MobileNet weight and put it into the model
    state_dict = torch.load(landmark_model_path, map_location=device) 
    landmark_model.load_state_dict(state_dict) # put the weights into the model

    landmark_model.to(device)
    landmark_model.eval() # now in inference mode

    preprocess = get_transform()

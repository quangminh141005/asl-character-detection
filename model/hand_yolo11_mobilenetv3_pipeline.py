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
    
# preprocessing and postprocessing
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

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


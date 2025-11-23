# ASL Character Detection

## 1. Quick Guide
1. Create the conda environment using:
   - `environment_gpu.yml` (if using GPU), or  
   - `environment_cpu.yml` (if using CPU)
2. Run `./model-combine/pose/infer_hand_rtmpose_webcam.py` to test hand-landmark detection via webcam.  
   Make sure to update the `device` setting according to your hardware (**cpu** or **cuda:0**).

## 2. Project Structure
```
ASL-CHARACTER-DETECTION/model-combine
├── conda_env
│   ├── enviroment_cpu.yml
│   └── enviroment_gpu.yml
├── dataset-maker-for-svm
│   ├── asl_svm_dataset.csv
│   └── asl_svm_dataset.py
├── edit_dataset.py
├── hand_palm_detection_model.ipynb
├── model-combine
│   ├── bboxes
│   │   └── test_bbox.json
│   ├── data
│   │   ├── test2.jpg
│   │   ├── test_detected.jpg
│   │   ├── test_hand_pose.jpg
│   │   └── test.png
│   ├── detection
│   │   ├── detect_hand.py
│   │   ├── hqm_hand_palm.pt
│   │   └── yolov8.pt
│   ├── pose
│   │   ├── infer_hand_rtmpose.py
│   │   ├── infer_hand_rtmpose_webcam.py
│   │   ├── rtmpose-m-hand-256x256.py
│   │   └── rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth
│   └── svm
│       ├── result.txt
│       ├── svm_asl_model.joblib
│       └── svm_train.py
├── Pre_process_data.ipynb
├── README.md
└── requirements.txt

```

## 3. Description
This repository contains the model components for the 3-phase ASL character recognition pipeline:

1. **Hand Detection** – YOLOv8 hand palm model  
2. **Hand Landmark Detection** – RTMPose hand keypoint model in MMPoseLab  
3. **ASL Character Classification** – SVM classifier trained on hand-landmark features 

This model is used in the following project:  
👉 **[ASL Detection Website](https://github.com/Tuan-Nguyen-Minhh/web-american-sign-language)**  




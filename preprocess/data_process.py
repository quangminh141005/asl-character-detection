import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
DATASET_PATH = "../asl_dataset"  # Path to your folders A, B, C...
OUTPUT_FILE = "./hand_landmarks.csv"
USE_Z = False        # Set True for (x,y,z), False for (x,y)
FRAME_SKIP = 2       # Process every 2nd frame (0, 2, 4...)

# Initialize MediaPipe
mp_hands = mp.solutions.hands

def normalize_landmarks(landmarks, use_z=False, flip=False):
    """
    Converts landmarks to numpy, applies flipping, centers at wrist,
    and scales by maximum Euclidean distance.
    """
    # 1. Convert to Numpy Array
    data = []
    for lm in landmarks.landmark:
        x = (1.0 - lm.x) if flip else lm.x
        if use_z:
            data.append([x, lm.y, lm.z])
        else:
            data.append([x, lm.y])
    
    keypoints = np.array(data)

    # 2. Centering (Relative to Wrist)
    # Landmark 0 is always the wrist
    wrist = keypoints[0].copy()
    coords = keypoints - wrist
    
    # 3. Scaling (Distance Invariance)
    # Calculate Euclidean distance (L2 norm) for each point from wrist
    dists = np.linalg.norm(coords, axis=1)
    scale = dists.max()
    
    if scale < 1e-6:
        scale = 1.0
        
    coords = coords / scale
    
    # Return flattened list for CSV row
    return coords.flatten().tolist()

def process_dataset():
    all_data = []
    
    # Setup Column Names
    cols = ['video_id', 'label', 'type']
    for i in range(21):
        cols.extend([f'{i}_x', f'{i}_y'])
        if USE_Z:
            cols.append(f'{i}_z')

    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7, # Higher confidence for cleaner data
        min_tracking_confidence=0.5) as hands:

        if not os.path.exists(DATASET_PATH):
            print(f"Error: Dataset path '{DATASET_PATH}' not found.")
            return

        # Walk through labels (A, B, C...)
        for label_dir in sorted(os.listdir(DATASET_PATH)):
            label_path = os.path.join(DATASET_PATH, label_dir)
            
            if not os.path.isdir(label_path) or label_dir.startswith('.'):
                continue
            
            print(f"Processing Label: {label_dir}")

            for video_file in os.listdir(label_path):
                if not video_file.lower().endswith(('.mp4', '.webm', '.avi', '.mov')):
                    continue
                
                video_path = os.path.join(label_path, video_file)
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                
                while cap.isOpened():
                    success, image = cap.read()
                    if not success:
                        break
                    
                    # Frame Skipping
                    if frame_count % FRAME_SKIP == 0:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = hands.process(image_rgb)
                        
                        if results.multi_hand_landmarks:
                            hand_landmarks = results.multi_hand_landmarks[0]
                            
                            # ORIGINAL
                            row_orig = [video_file, label_dir, 'original']
                            row_orig.extend(normalize_landmarks(hand_landmarks, USE_Z, flip=False))
                            all_data.append(row_orig)
                            
                            # FLIPPED (Data Augmentation)
                            row_flip = [video_file, label_dir, 'flipped']
                            row_flip.extend(normalize_landmarks(hand_landmarks, USE_Z, flip=True))
                            all_data.append(row_flip)

                    frame_count += 1
                
                cap.release()

    # Save to CSV
    print(f"Extraction complete. Total samples: {len(all_data)}")
    df = pd.DataFrame(all_data, columns=cols)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_dataset()
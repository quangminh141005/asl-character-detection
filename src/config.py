# Data
CSV_FILE = "data/hand_landmarks.csv"
NUM_LANDMARKS = 21

# Augmentation
NOISE_STD = 0.01
ROT_DEG = 8.0
SCALE_JITTER = 0.05

# Split
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS = 5

# SVM
SVM_C = 10.0
SVM_KERNEL = "rbf"
SVM_GAMMA = "scale"
SVM_CLASS_WEIGHT = "balanced"

# Visualization
NUM_SAMPLES = 5
NUM_AUG_SAMPLES = 4
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
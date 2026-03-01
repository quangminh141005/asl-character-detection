# src/augmentation.py

import numpy as np
from src.config import NOISE_STD, ROT_DEG, SCALE_JITTER


def augment_landmarks_xy(
    X: np.ndarray,
    n_copies: int = 1,
    noise_std: float = NOISE_STD,
    rot_deg: float = ROT_DEG,
    scale_jitter: float = SCALE_JITTER,
    seed: int = 0,
) -> list:
    """
    Apply random rotation, scaling, and Gaussian noise to (N, D) landmark arrays.
    D = 2 * num_landmarks (interleaved x, y).
    Returns a list of n_copies augmented arrays, each of shape (N, D).
    """
    rng = np.random.default_rng(seed)
    N, D = X.shape
    K = D // 2
    X2 = X.reshape(N, K, 2).copy()
    center = X2[:, 0:1, :]
    Xc = X2 - center

    outs = []
    for _ in range(n_copies):
        ang = rng.uniform(-rot_deg, rot_deg, size=(N, 1, 1)) * (np.pi / 180.0)
        c, s = np.cos(ang), np.sin(ang)
        R = np.concatenate(
            [np.concatenate([c, -s], axis=2), np.concatenate([s, c], axis=2)],
            axis=1,
        )
        Xr = Xc @ np.transpose(R, (0, 2, 1))
        sc = rng.uniform(1.0 - scale_jitter, 1.0 + scale_jitter, size=(N, 1, 1))
        Xs = Xr * sc
        Xn = Xs + rng.normal(0.0, noise_std, size=Xs.shape)
        outs.append((Xn + center).reshape(N, D))
    return outs
"""Extract 478 3D face landmarks from an image using MediaPipe Face Landmarker.

The iOS app will send ARKit landmarks instead; both paths end up as an (N, 3)
array in a normalised face-centred coordinate frame via `normalise()`.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MODEL_PATH = Path(os.environ.get(
    "FACESCORE_LANDMARKER",
    Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task",
))

_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        _landmarker = vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


class NoFaceError(ValueError):
    pass


def extract(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (landmarks[478,3] in image-normalised coords, pose matrix[4,4])."""
    import cv2
    import mediapipe as mp

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = _get_landmarker().detect(mp_img)
    if not res.face_landmarks:
        raise NoFaceError("no face detected")
    lm = res.face_landmarks[0]
    h, w = rgb.shape[:2]
    pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=np.float32)
    pose = np.array(res.facial_transformation_matrixes[0], dtype=np.float32) \
        if res.facial_transformation_matrixes else np.eye(4, dtype=np.float32)
    return pts, pose


def extract_from_path(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return extract(img)


# Canonical MediaPipe indices used throughout
IDX = {
    "left_eye_outer": 33, "left_eye_inner": 133,
    "right_eye_inner": 362, "right_eye_outer": 263,
    "left_eye_top": 159, "left_eye_bottom": 145,
    "right_eye_top": 386, "right_eye_bottom": 374,
    "nose_tip": 1, "nose_bridge": 6, "nose_left": 48, "nose_right": 278,
    "subnasale": 2,
    "mouth_left": 61, "mouth_right": 291,
    "upper_lip_top": 0, "lower_lip_bottom": 17,
    "chin": 152, "forehead": 10,
    "jaw_left": 172, "jaw_right": 397,
    "gonion_left": 58, "gonion_right": 288,
    "cheek_left": 234, "cheek_right": 454,
    "brow_left_inner": 107, "brow_right_inner": 336,
    "brow_left_outer": 70, "brow_right_outer": 300,
    "glabella": 9,
}


def normalise(pts: np.ndarray) -> np.ndarray:
    """Centre on midpoint between eyes, rotate so eye line is horizontal,
    scale so inter-ocular distance == 1. Removes translation/roll/scale."""
    pts = pts.astype(np.float64).copy()
    le = pts[IDX["left_eye_outer"]]
    re = pts[IDX["right_eye_outer"]]
    centre = (le + re) / 2
    pts -= centre
    # roll correction in the image plane
    dx, dy = (re - le)[:2]
    ang = -np.arctan2(dy, dx)
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pts = pts @ rot.T
    iod = np.linalg.norm(pts[IDX["right_eye_outer"]] - pts[IDX["left_eye_outer"]])
    if iod < 1e-6:
        raise ValueError("degenerate landmarks")
    return pts / iod

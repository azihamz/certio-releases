"""Geometry features from normalised landmarks.

These are the "measurements" half of the product: facial thirds, symmetry,
jaw angle, ratios. They're interpretable on their own and also the input
to the learned rating model.
"""
from __future__ import annotations

import numpy as np

from .landmarks import IDX, normalise

# Left/right mirror pairs for symmetry (MediaPipe canonical)
_MIRROR_PAIRS = [
    (33, 263), (133, 362), (159, 386), (145, 374), (61, 291), (48, 278),
    (172, 397), (58, 288), (234, 454), (107, 336), (70, 300), (130, 359),
    (127, 356), (93, 323), (132, 361), (136, 365), (150, 379), (176, 400),
    (78, 308), (95, 324), (88, 318), (40, 270), (39, 269), (37, 267),
]

FEATURE_NAMES = [
    "upper_third", "middle_third", "lower_third",
    "face_width_height", "jaw_width_ratio", "cheek_width_ratio",
    "jaw_angle_deg", "chin_height_ratio",
    "eye_width_ratio", "interocular_ratio", "eye_aspect",
    "canthal_tilt_deg", "brow_height_ratio",
    "nose_width_ratio", "nose_length_ratio",
    "mouth_width_ratio", "lip_height_ratio", "philtrum_ratio",
    "symmetry_rms", "symmetry_max",
    "midface_ratio", "fwhr",
]


def _d(a, b):
    return float(np.linalg.norm(a - b))


def _angle(a, b, c) -> float:
    """Angle ABC in degrees."""
    v1, v2 = a - b, c - b
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))


def compute(raw_pts: np.ndarray) -> dict[str, float]:
    p = normalise(raw_pts)
    g = lambda k: p[IDX[k]]  # noqa: E731

    forehead, glab, subn, chin = g("forehead"), g("glabella"), g("subnasale"), g("chin")
    face_h = abs(chin[1] - forehead[1])
    upper = abs(glab[1] - forehead[1]) / face_h
    middle = abs(subn[1] - glab[1]) / face_h
    lower = abs(chin[1] - subn[1]) / face_h

    cheek_w = _d(g("cheek_left"), g("cheek_right"))
    jaw_w = _d(g("gonion_left"), g("gonion_right"))
    eye_w_l = _d(g("left_eye_outer"), g("left_eye_inner"))
    eye_w_r = _d(g("right_eye_outer"), g("right_eye_inner"))
    eye_w = (eye_w_l + eye_w_r) / 2
    inter = _d(g("left_eye_inner"), g("right_eye_inner"))
    eye_h = (_d(g("left_eye_top"), g("left_eye_bottom")) + _d(g("right_eye_top"), g("right_eye_bottom"))) / 2

    # canthal tilt: outer canthus vs inner canthus, positive = outer higher (y down in image coords)
    tilt_l = np.degrees(np.arctan2(g("left_eye_inner")[1] - g("left_eye_outer")[1], abs(g("left_eye_inner")[0] - g("left_eye_outer")[0])))
    tilt_r = np.degrees(np.arctan2(g("right_eye_inner")[1] - g("right_eye_outer")[1], abs(g("right_eye_inner")[0] - g("right_eye_outer")[0])))
    canthal = float((tilt_l + tilt_r) / 2)

    jaw_angle = (_angle(g("cheek_left"), g("gonion_left"), chin) + _angle(g("cheek_right"), g("gonion_right"), chin)) / 2

    mouth_w = _d(g("mouth_left"), g("mouth_right"))
    lip_h = _d(g("upper_lip_top"), g("lower_lip_bottom"))
    philtrum = abs(g("upper_lip_top")[1] - subn[1])
    nose_w = _d(g("nose_left"), g("nose_right"))
    nose_len = abs(g("nose_tip")[1] - g("nose_bridge")[1])
    brow_h = (abs(g("brow_left_inner")[1] - g("left_eye_top")[1]) + abs(g("brow_right_inner")[1] - g("right_eye_top")[1])) / 2
    chin_h = abs(chin[1] - g("lower_lip_bottom")[1])

    # symmetry: mirror right side across x=0 and compare to left
    diffs = []
    for li, ri in _MIRROR_PAIRS:
        l, r = p[li].copy(), p[ri].copy()
        r[0] = -r[0]
        diffs.append(np.linalg.norm(l - r))
    diffs = np.array(diffs)

    feats = {
        "upper_third": upper, "middle_third": middle, "lower_third": lower,
        "face_width_height": cheek_w / face_h,
        "jaw_width_ratio": jaw_w / cheek_w,
        "cheek_width_ratio": cheek_w / face_h,
        "jaw_angle_deg": jaw_angle,
        "chin_height_ratio": chin_h / face_h,
        "eye_width_ratio": eye_w / cheek_w,
        "interocular_ratio": inter / eye_w,
        "eye_aspect": eye_h / eye_w,
        "canthal_tilt_deg": canthal,
        "brow_height_ratio": brow_h / eye_h,
        "nose_width_ratio": nose_w / inter,
        "nose_length_ratio": nose_len / face_h,
        "mouth_width_ratio": mouth_w / nose_w,
        "lip_height_ratio": lip_h / mouth_w,
        "philtrum_ratio": philtrum / lip_h,
        "symmetry_rms": float(np.sqrt((diffs ** 2).mean())),
        "symmetry_max": float(diffs.max()),
        "midface_ratio": middle / cheek_w,
        "fwhr": cheek_w / abs(g("upper_lip_top")[1] - glab[1]),
    }
    return {k: float(feats[k]) for k in FEATURE_NAMES}


def to_vector(feats: dict[str, float]) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)


# Plain-English readouts for the report. (low, high) = "normal" band; outside -> label.
REPORT_BANDS = {
    "upper_third": (0.30, 0.37, "Short forehead", "Tall forehead"),
    "middle_third": (0.30, 0.37, "Compact midface", "Long midface"),
    "lower_third": (0.30, 0.37, "Short lower face", "Long lower face"),
    "jaw_width_ratio": (0.80, 0.95, "Tapered jaw", "Wide jaw"),
    "jaw_angle_deg": (115, 135, "Sharp jaw angle", "Soft jaw angle"),
    "canthal_tilt_deg": (-2.0, 6.0, "Negative canthal tilt", "Positive canthal tilt"),
    "symmetry_rms": (0.0, 0.05, "", "Asymmetry noted"),
    "fwhr": (1.7, 2.1, "Narrow face", "Wide face"),
    "interocular_ratio": (0.9, 1.15, "Close-set eyes", "Wide-set eyes"),
}


def readout(feats: dict[str, float]) -> list[dict]:
    out = []
    for k, (lo, hi, low_label, high_label) in REPORT_BANDS.items():
        v = feats[k]
        label = low_label if v < lo else high_label if v > hi else "Balanced"
        if label:
            out.append({"metric": k, "value": round(v, 3), "label": label})
    return out

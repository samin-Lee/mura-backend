import numpy as np

from analysis.face_landmarks import get_face_landmark_points


MIN_DENOMINATOR = 1e-6
NOSE_BASE = 2
UPPER_LIP_TOP = 0
UPPER_LIP_INNER = 13
LOWER_LIP_INNER = 14
LOWER_LIP_BOTTOM = 17
CHIN_BOTTOM = 152
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291
LEFT_IRIS_REFERENCE = 469
RIGHT_IRIS_REFERENCE = 476

SHORT_CHIN_TO_PHILTRUM_RATIO_THRESHOLD = 1.7
LONG_CHIN_TO_PHILTRUM_RATIO_THRESHOLD = 2.2
THIN_LIP_TO_LOWER_FACE_RATIO_THRESHOLD = 0.24
THICK_LIP_TO_LOWER_FACE_RATIO_THRESHOLD = 0.32
THIN_LOWER_LIP_RATIO_THRESHOLD = 1.0
THIN_UPPER_LIP_RATIO_THRESHOLD = 2.0
NARROW_MOUTH_TO_IRIS_RATIO_THRESHOLD = 0.9
WIDE_MOUTH_TO_IRIS_RATIO_THRESHOLD = 1.1


def _distance(first, second):
    return float(np.linalg.norm(first - second))


def _safe_ratio(numerator, denominator):
    return float(numerator) / max(float(denominator), MIN_DENOMINATOR)


def calculate_mouth_scores(landmark_points):
    philtrum_length = _distance(
        landmark_points[NOSE_BASE][:2],
        landmark_points[UPPER_LIP_TOP][:2],
    )
    chin_length = _distance(
        landmark_points[LOWER_LIP_BOTTOM][:2],
        landmark_points[CHIN_BOTTOM][:2],
    )
    upper_lip_thickness = _distance(
        landmark_points[UPPER_LIP_TOP][:2],
        landmark_points[UPPER_LIP_INNER][:2],
    )
    lower_lip_thickness = _distance(
        landmark_points[LOWER_LIP_INNER][:2],
        landmark_points[LOWER_LIP_BOTTOM][:2],
    )
    total_lip_thickness = _distance(
        landmark_points[UPPER_LIP_TOP][:2],
        landmark_points[LOWER_LIP_BOTTOM][:2],
    )
    lower_face_length = _distance(
        landmark_points[NOSE_BASE][:2],
        landmark_points[CHIN_BOTTOM][:2],
    )
    mouth_width = _distance(
        landmark_points[LEFT_MOUTH_CORNER][:2],
        landmark_points[RIGHT_MOUTH_CORNER][:2],
    )
    iris_distance = _distance(
        landmark_points[LEFT_IRIS_REFERENCE][:2],
        landmark_points[RIGHT_IRIS_REFERENCE][:2],
    )

    return {
        "chin_to_philtrum": _safe_ratio(chin_length, philtrum_length),
        "lip_thickness_to_lower_face": _safe_ratio(
            total_lip_thickness,
            lower_face_length,
        ),
        "lower_lip_to_upper_lip": _safe_ratio(
            lower_lip_thickness,
            upper_lip_thickness,
        ),
        "mouth_width_to_iris_distance": _safe_ratio(mouth_width, iris_distance),
    }


def classify_mouth_scores(scores):
    chin_ratio = scores["chin_to_philtrum"]
    lip_thickness_ratio = scores["lip_thickness_to_lower_face"]
    lower_upper_lip_ratio = scores["lower_lip_to_upper_lip"]
    mouth_width_ratio = scores["mouth_width_to_iris_distance"]

    if chin_ratio > LONG_CHIN_TO_PHILTRUM_RATIO_THRESHOLD:
        chin_philtrum = "long_chin_or_short_philtrum"
    elif chin_ratio < SHORT_CHIN_TO_PHILTRUM_RATIO_THRESHOLD:
        chin_philtrum = "short_chin_or_long_philtrum"
    else:
        chin_philtrum = "balanced"

    if lip_thickness_ratio > THICK_LIP_TO_LOWER_FACE_RATIO_THRESHOLD:
        lip_thickness = "thick"
    elif lip_thickness_ratio < THIN_LIP_TO_LOWER_FACE_RATIO_THRESHOLD:
        lip_thickness = "thin"
    else:
        lip_thickness = "balanced"

    if lower_upper_lip_ratio > THIN_UPPER_LIP_RATIO_THRESHOLD:
        lip_balance = "thin_upper_lip"
    elif lower_upper_lip_ratio < THIN_LOWER_LIP_RATIO_THRESHOLD:
        lip_balance = "thin_lower_lip"
    else:
        lip_balance = "balanced"

    if mouth_width_ratio < NARROW_MOUTH_TO_IRIS_RATIO_THRESHOLD:
        mouth_width = "narrow"
    elif mouth_width_ratio > WIDE_MOUTH_TO_IRIS_RATIO_THRESHOLD:
        mouth_width = "wide"
    else:
        mouth_width = "balanced"

    return {
        "chin_philtrum": chin_philtrum,
        "lip_thickness": lip_thickness,
        "lip_balance": lip_balance,
        "mouth_width": mouth_width,
    }


def calculate_mouth_metrics(image_bgr, landmark_points=None):
    points = (
        landmark_points
        if landmark_points is not None
        else get_face_landmark_points(image_bgr)
    )
    scores = calculate_mouth_scores(points)

    return {
        "scores": scores,
        "classification": classify_mouth_scores(scores),
    }

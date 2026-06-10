import numpy as np

from analysis.face_landmarks import get_face_landmark_points


MIN_DENOMINATOR = 1e-6
NOSE_TOP = 8
NOSE_BOTTOM = 19
NOSE_LEFT = 48
NOSE_RIGHT = 278
LEFT_INNER_EYE = 133
RIGHT_INNER_EYE = 362
SHORT_NOSE_LENGTH_RATIO_THRESHOLD = 0.30
LONG_NOSE_LENGTH_RATIO_THRESHOLD = 0.38
WIDE_NOSE_RATIO_THRESHOLD = 1.1


def _distance(first, second):
    return float(np.linalg.norm(first - second))


def _safe_ratio(numerator, denominator):
    return float(numerator) / max(float(denominator), MIN_DENOMINATOR)


def calculate_nose_scores(landmark_points, face_length):
    nose_length = _distance(
        landmark_points[NOSE_TOP][:2],
        landmark_points[NOSE_BOTTOM][:2],
    )
    nose_width = _distance(
        landmark_points[NOSE_LEFT][:2],
        landmark_points[NOSE_RIGHT][:2],
    )
    eye_distance = _distance(
        landmark_points[LEFT_INNER_EYE][:2],
        landmark_points[RIGHT_INNER_EYE][:2],
    )

    return {
        "nose_length_to_face_length": _safe_ratio(nose_length, face_length),
        "nose_width_to_eye_distance": _safe_ratio(nose_width, eye_distance),
    }


def classify_nose_scores(scores):
    length_ratio = scores["nose_length_to_face_length"]
    width_ratio = scores["nose_width_to_eye_distance"]

    if length_ratio >= LONG_NOSE_LENGTH_RATIO_THRESHOLD:
        nose_length = "long"
    elif length_ratio <= SHORT_NOSE_LENGTH_RATIO_THRESHOLD:
        nose_length = "short"
    else:
        nose_length = "balanced"

    return {
        "nose_length": nose_length,
        "nose_width": "wide" if width_ratio >= WIDE_NOSE_RATIO_THRESHOLD else "not_wide",
    }


def calculate_nose_metrics(image_bgr, face_length, landmark_points=None):
    points = (
        landmark_points
        if landmark_points is not None
        else get_face_landmark_points(image_bgr)
    )
    scores = calculate_nose_scores(points, face_length)

    return {
        "scores": scores,
        "classification": classify_nose_scores(scores),
    }

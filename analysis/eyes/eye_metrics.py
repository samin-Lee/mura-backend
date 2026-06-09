import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_DENOMINATOR = 1e-6
INSIGHTFACE_DET_SIZE = (640, 640)
INSIGHTFACE_DET_THRESH = 0.5


LEFT_EYE = {
    "outer_corner": 33,
    "inner_corner": 133,
    "top": 159,
    "bottom": 145,
    "iris": [468, 469, 470, 471, 472],
}

RIGHT_EYE = {
    "inner_corner": 362,
    "outer_corner": 263,
    "top": 386,
    "bottom": 374,
    "iris": [473, 474, 475, 476, 477],
}

INSIGHTFACE_LEFT_EYE_REFERENCE = 40
INSIGHTFACE_LEFT_BROW_LINE = (45, 47)
INSIGHTFACE_RIGHT_EYE_REFERENCE = 94
INSIGHTFACE_RIGHT_BROW_LINE = (98, 99)


def load_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "mediapipe is required. Install it with: pip install mediapipe"
        ) from exc
    return mp


def load_insightface():
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise RuntimeError(
            "insightface is required. Install it with: pip install insightface"
        ) from exc
    return FaceAnalysis


@lru_cache(maxsize=1)
def get_insightface_app():
    FaceAnalysis = load_insightface()
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(
        ctx_id=-1,
        det_size=INSIGHTFACE_DET_SIZE,
        det_thresh=INSIGHTFACE_DET_THRESH,
    )
    return app


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _safe_ratio(numerator, denominator):
    denominator = max(float(denominator), MIN_DENOMINATOR)
    return float(numerator) / denominator


def _distance(first, second):
    return float(np.linalg.norm(first - second))


def _landmark_points(image_bgr):
    if image_bgr is None:
        raise ValueError("Image is empty")

    mp = load_mediapipe()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_bgr.shape[:2]

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        result = face_mesh.process(image_rgb)

    if not result.multi_face_landmarks:
        raise ValueError("No MediaPipe face landmarks were detected.")

    landmarks = result.multi_face_landmarks[0].landmark
    return np.array(
        [
            (
                landmark.x * (width - 1),
                landmark.y * (height - 1),
                landmark.z * max(width, height),
            )
            for landmark in landmarks
        ],
        dtype=np.float32,
    )


def _insightface_landmarks(image_bgr):
    if image_bgr is None:
        raise ValueError("Image is empty")

    faces = get_insightface_app().get(image_bgr)
    if not faces:
        raise ValueError("No InsightFace face landmarks were detected.")

    def bbox_area(face):
        x1, y1, x2, y2 = face.bbox
        return max(float(x2 - x1), 0.0) * max(float(y2 - y1), 0.0)

    face = max(faces, key=bbox_area)
    landmarks = getattr(face, "landmark_2d_106", None)
    if landmarks is None:
        raise ValueError("InsightFace landmark_2d_106 is not available.")

    return np.array(landmarks, dtype=np.float32)


def _line_y_at_x(first, second, x):
    x1, y1 = first
    x2, y2 = second
    if abs(float(x2 - x1)) <= MIN_DENOMINATOR:
        return float((y1 + y2) / 2.0)

    slope = float((y2 - y1) / (x2 - x1))
    return float(y1 + slope * (x - x1))


def _insightface_eyebrow_eye_distances(image_bgr):
    landmarks = _insightface_landmarks(image_bgr)
    left_eye = landmarks[INSIGHTFACE_LEFT_EYE_REFERENCE]
    right_eye = landmarks[INSIGHTFACE_RIGHT_EYE_REFERENCE]

    left_brow_y = _line_y_at_x(
        landmarks[INSIGHTFACE_LEFT_BROW_LINE[0]],
        landmarks[INSIGHTFACE_LEFT_BROW_LINE[1]],
        left_eye[0],
    )
    right_brow_y = _line_y_at_x(
        landmarks[INSIGHTFACE_RIGHT_BROW_LINE[0]],
        landmarks[INSIGHTFACE_RIGHT_BROW_LINE[1]],
        right_eye[0],
    )

    eyebrow_y_average = (left_brow_y + right_brow_y) / 2.0
    eye_y_average = (float(left_eye[1]) + float(right_eye[1])) / 2.0
    average_distance = abs(eye_y_average - eyebrow_y_average)

    left_distance = abs(float(left_eye[1]) - left_brow_y)
    right_distance = abs(float(right_eye[1]) - right_brow_y)
    return left_distance, right_distance, average_distance


def _eye_measurements(points, eye):
    outer = points[eye["outer_corner"]][:2]
    inner = points[eye["inner_corner"]][:2]
    top = points[eye["top"]][:2]
    bottom = points[eye["bottom"]][:2]
    iris = [points[index][:2] for index in eye["iris"]]

    horizontal = _distance(outer, inner)
    vertical = _distance(top, bottom)

    iris_center = iris[0]
    iris_radius = float(np.mean([_distance(iris_center, point) for point in iris[1:]]))
    iris_diameter = iris_radius * 2.0

    return {
        "horizontal_length": horizontal,
        "vertical_length": vertical,
        "outer_corner": outer.tolist(),
        "inner_corner": inner.tolist(),
        "center": ((outer + inner) / 2.0).tolist(),
        "upturned_score": _safe_ratio(float(inner[1] - outer[1]), horizontal),
        "iris_diameter": iris_diameter,
    }


def classify_eye_metrics(ratios, scores):
    horizontal_vertical_ratio = ratios["eye_horizontal_to_vertical"]
    horizontal_distance_ratio = ratios["eye_horizontal_to_eye_distance"]
    upturned_angle = scores["upturned_angle_degrees"]
    eyebrow_iris_ratio = ratios["eyebrow_eye_to_iris_diameter"]

    if horizontal_vertical_ratio < 3.0:
        eye_shape = "토끼 눈"
    else:
        eye_shape = "고양이 눈"

    if horizontal_distance_ratio < 0.9:
        eye_distance = "좁은 미간"
    elif horizontal_distance_ratio <= 1.1:
        eye_distance = "보통 미간"
    else:
        eye_distance = "넓은 미간"

    if upturned_angle < 3.0:
        eye_tail = "내려간 눈"
    elif upturned_angle <= 8.0:
        eye_tail = "보통 눈"
    else:
        eye_tail = "올라간 눈"

    if eyebrow_iris_ratio < 0.8:
        eyebrow_eye_distance = "좁은 간격"
    elif eyebrow_iris_ratio <= 1.2:
        eyebrow_eye_distance = "보통 간격"
    else:
        eyebrow_eye_distance = "넓은 간격"

    return {
        "eye_shape": eye_shape,
        "eye_distance": eye_distance,
        "eye_tail": eye_tail,
        "eyebrow_eye_distance": eyebrow_eye_distance,
    }


def calculate_eye_metrics(image_bgr):
    points = _landmark_points(image_bgr)
    left = _eye_measurements(points, LEFT_EYE)
    right = _eye_measurements(points, RIGHT_EYE)
    (
        left_eyebrow_eye_distance,
        right_eyebrow_eye_distance,
        average_eyebrow_eye_distance,
    ) = (
        _insightface_eyebrow_eye_distances(image_bgr)
    )
    left["eyebrow_eye_distance"] = left_eyebrow_eye_distance
    right["eyebrow_eye_distance"] = right_eyebrow_eye_distance

    average_eye_horizontal_length = (
        left["horizontal_length"] + right["horizontal_length"]
    ) / 2.0
    average_eye_vertical_length = (
        left["vertical_length"] + right["vertical_length"]
    ) / 2.0
    average_iris_diameter = (left["iris_diameter"] + right["iris_diameter"]) / 2.0
    eye_distance = _distance(
        np.array(left["inner_corner"], dtype=np.float32),
        np.array(right["inner_corner"], dtype=np.float32),
    )
    upturned_score = (left["upturned_score"] + right["upturned_score"]) / 2.0
    upturned_angle_degrees = math.degrees(math.atan(upturned_score))

    ratios = {
        "eye_horizontal_to_vertical": _safe_ratio(
            average_eye_horizontal_length,
            average_eye_vertical_length,
        ),
        "eye_horizontal_to_eye_distance": _safe_ratio(
            average_eye_horizontal_length,
            eye_distance,
        ),
        "eyebrow_eye_to_iris_diameter": _safe_ratio(
            average_eyebrow_eye_distance,
            average_iris_diameter,
        ),
    }
    scores = {
        "upturned_angle_degrees": upturned_angle_degrees,
    }

    return {
        "ratios": ratios,
        "scores": scores,
        "classification": classify_eye_metrics(ratios, scores),
        "measurements": {
            "average_eye_horizontal_length": average_eye_horizontal_length,
            "average_eye_vertical_length": average_eye_vertical_length,
            "eye_distance": eye_distance,
            "average_eyebrow_eye_distance": average_eyebrow_eye_distance,
            "average_iris_diameter": average_iris_diameter,
            "left_eye": left,
            "right_eye": right,
        },
    }


def calculate_eye_metrics_path(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {image_path.suffix}")

    image = read_image(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    return calculate_eye_metrics(image)

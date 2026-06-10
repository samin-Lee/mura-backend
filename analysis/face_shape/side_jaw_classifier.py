import os
from pathlib import Path

import cv2
import numpy as np
from analysis.face_shape.draw_face_outline import create_session, make_face_mask, parse_face
from analysis.face_shape.draw_face_outline_points import (
    get_face_oval_landmarks,
    get_largest_contour,
    project_landmarks_to_contour,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CHEEKBONE_THRESHOLD = 0.091872
JAW_THRESHOLD = 0.747861
DEFAULT_MODEL_PATH = None
SHORT_FACE_RATIO_THRESHOLD = 1.32
LONG_FACE_RATIO_THRESHOLD = 1.4
FACE_SHAPE_OVAL = "계란형"
FACE_SHAPE_LONG = "긴형"
FACE_SHAPE_ROUND = "둥근형"
FACE_SHAPE_SQUARE = "사각형"
FACE_SHAPE_INVERTED_TRIANGLE_OR_HEART = "역삼각형/하트형"
FACE_SHAPE_HEXAGON = "육각형"


class SuppressStderr:
    def __enter__(self):
        self.saved_stderr = os.dup(2)
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.devnull, 2)
        os.close(self.devnull)

    def __exit__(self, exc_type, exc, traceback):
        os.dup2(self.saved_stderr, 2)
        os.close(self.saved_stderr)


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def get_outline_points(session, image):
    parsing = parse_face(session, image)
    mask = make_face_mask(parsing)
    contour = get_largest_contour(mask)
    landmarks = get_face_oval_landmarks(image)
    points = project_landmarks_to_contour(landmarks, contour)
    return np.array(points, dtype=np.float32)


def symmetric_width(points, index):
    opposite_index = 36 - index
    return abs(float(points[index, 0] - points[opposite_index, 0]))


def point_distance(points, first_index, second_index):
    return float(np.linalg.norm(points[first_index] - points[second_index]))


def calculate_scores(points):
    cheek_width = symmetric_width(points, 8)  # MediaPipe 454-234
    lower_width = symmetric_width(points, 10)  # MediaPipe 361-132
    jaw_bone_width = symmetric_width(points, 12)  # MediaPipe 397-172
    face_width = max(point_distance(points, 27, 9), 1.0)  # MediaPipe 93-323
    face_length = max(point_distance(points, 0, 18), 1.0)  # MediaPipe 10-152

    return {
        "cheekbone": (cheek_width - lower_width) / face_width,
        "jaw": jaw_bone_width / face_width,
        "face_length_width_ratio": face_length / face_width,
    }


def classify_scores(scores):
    face_ratio = scores["face_length_width_ratio"]
    if face_ratio <= SHORT_FACE_RATIO_THRESHOLD:
        face_length = "short"
    elif face_ratio >= LONG_FACE_RATIO_THRESHOLD:
        face_length = "long"
    else:
        face_length = "balanced"

    cheekbone = scores["cheekbone"] >= CHEEKBONE_THRESHOLD
    jaw = scores["jaw"] >= JAW_THRESHOLD

    return {
        "cheekbone": cheekbone,
        "jaw": jaw,
        "face_length": face_length,
        "shape": classify_face_shape(jaw, cheekbone, face_length),
    }


def classify_face_shape(jaw, cheekbone, face_length):
    if jaw and cheekbone:
        return FACE_SHAPE_HEXAGON
    if jaw:
        return FACE_SHAPE_SQUARE
    if cheekbone:
        return FACE_SHAPE_INVERTED_TRIANGLE_OR_HEART
    if face_length == "long":
        return FACE_SHAPE_LONG
    if face_length == "short":
        return FACE_SHAPE_ROUND
    return FACE_SHAPE_OVAL


def classify_image(image, model_path=None):
    model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if model_path is not None and not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if image is None:
        raise ValueError("Image is empty")

    with SuppressStderr():
        session = create_session(model_path)
        points = get_outline_points(session, image)

    scores = calculate_scores(points)
    return {
        "scores": scores,
        "classification": classify_scores(scores),
    }

def classify_image_path(
    image_path,
    model_path=None,
):
    image_path = Path(image_path)
    model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if model_path is not None and not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {image_path.suffix}")

    image = read_image(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    return classify_image(image, model_path)

from pathlib import Path

import cv2
import numpy as np

from analysis.eyes.eye_metrics import (
    IMAGE_EXTENSIONS,
    _distance,
    _landmark_points,
    read_image,
)


LEFT_EYE_SCLERA = {
    "inner": 133,
    "outer": 33,
    "iris_center": 468,
    "iris_edge_sample": 469,
    "eye_poly": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
}

RIGHT_EYE_SCLERA = {
    "inner": 362,
    "outer": 263,
    "iris_center": 473,
    "iris_edge_sample": 474,
    "eye_poly": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
}


def _image_point(points, index, width, height):
    x = int(round(float(points[index][0])))
    y = int(round(float(points[index][1])))
    return min(max(x, 0), width - 1), min(max(y, 0), height - 1)


def _line_y_at_x(first, second, x):
    x1, y1 = first
    x2, y2 = second
    if x2 == x1:
        return int(round((y1 + y2) / 2.0))

    slope = (y2 - y1) / (x2 - x1)
    return int(round(y1 + slope * (x - x1)))


def _single_eye_sclera(image_bgr, points, eye):
    height, width = image_bgr.shape[:2]

    inner = _image_point(points, eye["inner"], width, height)
    outer = _image_point(points, eye["outer"], width, height)
    iris_center = _image_point(points, eye["iris_center"], width, height)
    iris_edge = _image_point(points, eye["iris_edge_sample"], width, height)
    eye_poly = np.array(
        [_image_point(points, index, width, height) for index in eye["eye_poly"]],
        dtype=np.int32,
    )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    eye_mask = np.zeros_like(gray)
    iris_mask = np.zeros_like(gray)

    cv2.fillPoly(eye_mask, [eye_poly], 255)
    iris_radius = max(int(round(_distance(np.array(iris_center), np.array(iris_edge)))), 1)
    cv2.circle(iris_mask, iris_center, iris_radius, 255, -1)

    sclera_mask = cv2.bitwise_and(eye_mask, cv2.bitwise_not(iris_mask))
    start_x = min(iris_center[0], outer[0])
    end_x = max(iris_center[0], outer[0])

    upper_pixels = 0
    lower_pixels = 0

    for x in range(start_x, end_x + 1):
        line_y = _line_y_at_x(inner, outer, x)
        white_y = np.where(sclera_mask[:, x] == 255)[0]
        upper_pixels += int(np.sum(white_y < line_y))
        lower_pixels += int(np.sum(white_y > line_y))

    return {
        "upper_pixels": upper_pixels,
        "lower_pixels": lower_pixels,
        "inner_corner": list(inner),
        "outer_corner": list(outer),
        "iris_center": list(iris_center),
    }


def classify_eye_sclera(average_upper_pixels, average_lower_pixels):
    if average_upper_pixels > average_lower_pixels:
        return {
            "type": "upper_more",
            "label": "위쪽 흰자가 더 많음",
            "eyeline": "down",
        }
    if average_lower_pixels > average_upper_pixels:
        return {
            "type": "lower_more",
            "label": "아래쪽 흰자가 더 많음",
            "eyeline": "up",
        }
    return {
        "type": "balanced",
        "label": "위아래 흰자 양이 같음",
        "eyeline": "balanced",
    }


def calculate_eye_sclera_from_points(image_bgr, points):
    if image_bgr is None:
        raise ValueError("Image is empty")

    left = _single_eye_sclera(image_bgr, points, LEFT_EYE_SCLERA)
    right = _single_eye_sclera(image_bgr, points, RIGHT_EYE_SCLERA)
    average_upper_pixels = (left["upper_pixels"] + right["upper_pixels"]) / 2.0
    average_lower_pixels = (left["lower_pixels"] + right["lower_pixels"]) / 2.0

    return {
        "classification": classify_eye_sclera(
            average_upper_pixels,
            average_lower_pixels,
        ),
        "measurements": {
            "average_upper_pixels": average_upper_pixels,
            "average_lower_pixels": average_lower_pixels,
            "left_eye": left,
            "right_eye": right,
        },
    }


def calculate_eye_sclera(image_bgr):
    points = _landmark_points(image_bgr)
    return calculate_eye_sclera_from_points(image_bgr, points)


def calculate_eye_sclera_path(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {image_path.suffix}")

    image = read_image(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    return calculate_eye_sclera(image)

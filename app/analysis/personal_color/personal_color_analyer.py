from math import sqrt
from typing import Iterable, Sequence

import cv2
import numpy as np


def _point_to_pixel(point, width: int, height: int) -> tuple[int, int]:
    # MediaPipe의 정규화 좌표를 실제 이미지 픽셀 좌표로 변환합니다.
    x = int(np.clip(point.x * width, 0, width - 1))
    y = int(np.clip(point.y * height, 0, height - 1))
    return x, y


def _polygon_mask(shape: tuple[int, int], points: Iterable[tuple[int, int]]) -> np.ndarray:
    # 여러 랜드마크 점으로 만든 다각형 영역을 마스크 이미지로 변환합니다.
    mask = np.zeros(shape, dtype=np.uint8)
    polygon = np.array(list(points), dtype=np.int32)
    if len(polygon) >= 3:
        cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def _circle_mask(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    # 눈동자처럼 원형으로 샘플링할 영역을 마스크로 만듭니다.
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, center, max(radius, 1), 255, -1)
    return mask


def _combine_masks(shape: tuple[int, int], masks: Sequence[np.ndarray]) -> np.ndarray:
    # 여러 샘플 영역을 하나의 마스크로 합칩니다.
    combined = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        combined = cv2.bitwise_or(combined, mask)
    return combined


def _mean_lab(image_lab: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    # 지정한 마스크 영역의 Lab 평균값을 구합니다. 영역이 비어 있으면 전체 평균으로 보정합니다.
    pixels = image_lab[mask > 0]
    if pixels.size == 0:
        pixels = image_lab.reshape(-1, 3)
    mean_l, mean_a, mean_b = np.mean(pixels, axis=0)
    return float(mean_l), float(mean_a), float(mean_b)


def _lab_distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    # 두 Lab 색상 사이의 단순 유클리드 거리로 색 대비를 계산합니다.
    return sqrt(
        (first[0] - second[0]) ** 2
        + (first[1] - second[1]) ** 2
        + (first[2] - second[2]) ** 2
    )


def _gray_world_white_balance(image_bgr: np.ndarray) -> np.ndarray:
    # 사진마다 다른 조명을 줄이기 위해 각 BGR 채널 평균을 회색에 가깝게 맞춥니다.
    image_float = image_bgr.astype(np.float32)
    channel_means = image_float.reshape(-1, 3).mean(axis=0)
    gray_mean = float(channel_means.mean())
    scale = gray_mean / np.maximum(channel_means, 1.0)
    balanced = image_float * scale
    return np.clip(balanced, 0, 255).astype(np.uint8)


def _normalize_lighting(image_bgr: np.ndarray) -> np.ndarray:
    # 화이트밸런스 후 Lab의 L 채널에 CLAHE를 적용해 밝기 차이를 완화합니다.
    balanced = _gray_world_white_balance(image_bgr)
    image_lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(image_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized_l = clahe.apply(l_channel)
    normalized_lab = cv2.merge([normalized_l, a_channel, b_channel])
    return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)


def _skin_mask(landmarks, width: int, height: int) -> np.ndarray:
    # 피부톤은 양쪽 볼과 이마 중앙을 샘플링해 입술/눈/그림자의 영향을 줄입니다.
    image_shape = (height, width)
    left_cheek_indices = [116, 117, 123, 147, 187, 205, 50, 101]
    right_cheek_indices = [345, 346, 352, 376, 411, 425, 280, 330]
    forehead_indices = [67, 109, 10, 338, 297, 9]

    masks = [
        _polygon_mask(image_shape, [_point_to_pixel(landmarks[index], width, height) for index in left_cheek_indices]),
        _polygon_mask(image_shape, [_point_to_pixel(landmarks[index], width, height) for index in right_cheek_indices]),
        _polygon_mask(image_shape, [_point_to_pixel(landmarks[index], width, height) for index in forehead_indices]),
    ]
    return _combine_masks(image_shape, masks)


def _iris_mask(landmarks, width: int, height: int) -> np.ndarray:
    # refine_landmarks=True일 때 제공되는 홍채 랜드마크를 우선 사용해 눈동자 색을 샘플링합니다.
    image_shape = (height, width)
    if len(landmarks) > 477:
        left_center = _point_to_pixel(landmarks[468], width, height)
        right_center = _point_to_pixel(landmarks[473], width, height)
        left_radius = int(max(2, np.linalg.norm(np.array(_point_to_pixel(landmarks[469], width, height)) - np.array(_point_to_pixel(landmarks[471], width, height))) / 2))
        right_radius = int(max(2, np.linalg.norm(np.array(_point_to_pixel(landmarks[474], width, height)) - np.array(_point_to_pixel(landmarks[476], width, height))) / 2))
        return _combine_masks(
            image_shape,
            [
                _circle_mask(image_shape, left_center, left_radius),
                _circle_mask(image_shape, right_center, right_radius),
            ],
        )

    # 홍채 랜드마크가 없는 경우에는 눈 영역 중심부를 보수적으로 사용합니다.
    left_eye = [_point_to_pixel(landmarks[index], width, height) for index in [33, 133, 159, 145]]
    right_eye = [_point_to_pixel(landmarks[index], width, height) for index in [362, 263, 386, 374]]
    return _combine_masks(
        image_shape,
        [
            _polygon_mask(image_shape, left_eye),
            _polygon_mask(image_shape, right_eye),
        ],
    )


def _hair_mask(image_bgr: np.ndarray, landmarks) -> np.ndarray:
    # 모발은 얼굴 bounding box 위쪽 영역에서 어둡고 채도가 있는 픽셀을 우선 샘플링합니다.
    height, width = image_bgr.shape[:2]
    image_shape = (height, width)
    xs = [point.x for point in landmarks]
    ys = [point.y for point in landmarks]
    left = int(np.clip(min(xs) * width, 0, width - 1))
    right = int(np.clip(max(xs) * width, 0, width - 1))
    face_top = int(np.clip(min(ys) * height, 0, height - 1))
    face_height = int(np.clip((max(ys) - min(ys)) * height, 1, height))

    top = max(0, face_top - int(face_height * 0.45))
    bottom = max(1, face_top + int(face_height * 0.08))
    side_margin = int((right - left) * 0.18)
    left = max(0, left - side_margin)
    right = min(width - 1, right + side_margin)

    mask = np.zeros(image_shape, dtype=np.uint8)
    mask[top:bottom, left:right] = 255

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    candidate = ((mask > 0) & (value < 150) & (saturation > 20)).astype(np.uint8) * 255

    # 밝은 염색모나 조명이 강한 사진에서는 위 조건이 부족할 수 있어 상단 영역 전체로 보완합니다.
    if cv2.countNonZero(candidate) < 50:
        candidate = mask
    return candidate


def analyze(image_bgr: np.ndarray, landmarks):
    # 모든 사진을 최대한 같은 조명 아래에서 찍은 것처럼 맞춘 뒤 Lab 색 공간에서 분석합니다.
    normalized_bgr = _normalize_lighting(image_bgr)
    normalized_lab = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2LAB)
    height, width = normalized_bgr.shape[:2]

    skin = _mean_lab(normalized_lab, _skin_mask(landmarks, width, height))
    iris = _mean_lab(normalized_lab, _iris_mask(landmarks, width, height))
    hair = _mean_lab(normalized_lab, _hair_mask(normalized_bgr, landmarks))

    # 요구사항에 따라 Lab의 a 채널 평균이 b 채널 평균보다 크면 쿨톤, 반대면 웜톤으로 분류합니다.
    tone = "cool" if skin[1] > skin[2] else "warm"

    eye_hair_contrast = _lab_distance(iris, hair)
    skin_hair_contrast = _lab_distance(skin, hair)
    weighted_contrast = (skin_hair_contrast * 0.65) + (eye_hair_contrast * 0.35)
    contrast_level = "high" if weighted_contrast > 50 else "low"

    # 쿨톤은 대비가 높으면 겨울, 낮으면 여름입니다. 웜톤은 대비가 높으면 봄, 낮으면 가을입니다.
    if tone == "cool":
        season = "winter" if contrast_level == "high" else "summer"
    else:
        season = "spring" if contrast_level == "high" else "autumn"

    return {
        "available": True,
        "tone": tone,
        "season": season,
        "contrast_level": contrast_level,
        "lab": {
            "skin": {"l": round(skin[0], 4), "a": round(skin[1], 4), "b": round(skin[2], 4)},
            "iris": {"l": round(iris[0], 4), "a": round(iris[1], 4), "b": round(iris[2], 4)},
            "hair": {"l": round(hair[0], 4), "a": round(hair[1], 4), "b": round(hair[2], 4)},
        },
        "metrics": {
            "skin_a_value": round(skin[1], 4),
            "skin_b_value": round(skin[2], 4),
            "eye_hair_contrast": round(eye_hair_contrast, 4),
            "skin_hair_contrast": round(skin_hair_contrast, 4),
            "weighted_contrast": round(weighted_contrast, 4),
            "skin_hair_weight": 0.65,
            "eye_hair_weight": 0.35,
            "season_threshold": 50.0,
        },
    }

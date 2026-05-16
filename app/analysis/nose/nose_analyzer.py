from math import hypot


def _distance(a, b) -> float:
    # 정규화된 2D 좌표 기준으로 두 랜드마크 사이의 거리를 계산합니다.
    return hypot(a.x - b.x, a.y - b.y)


def _classify(value: float, low: float, high: float, labels: tuple[str, str, str]) -> str:
    # 비율값을 미리 정한 임계값에 따라 문자열 라벨로 분류합니다.
    if value < low:
        return labels[0]
    if value > high:
        return labels[2]
    return labels[1]


def analyze(landmarks):
    # 얼굴 폭을 기준값으로 삼아 코의 폭과 길이를 상대 비율로 계산합니다.
    face_width = _distance(landmarks[234], landmarks[454])
    nose_width = _distance(landmarks[49], landmarks[279])
    nose_length = _distance(landmarks[168], landmarks[2])
    bridge_width = _distance(landmarks[193], landmarks[417])
    # z 좌표 차이는 코끝이 얼굴 중심에서 얼마나 돌출되어 보이는지의 보조 지표입니다.
    tip_depth = abs(landmarks[1].z - landmarks[168].z)

    width_ratio = nose_width / max(face_width, 1e-6)
    length_ratio = nose_length / max(face_width, 1e-6)
    bridge_ratio = bridge_width / max(nose_width, 1e-6)
    projection_ratio = tip_depth / max(nose_length, 1e-6)

    # 코 부위별 라벨과 원본 지표를 함께 반환합니다.
    return {
        "nose_size": _classify(width_ratio, 0.24, 0.31, ("small", "balanced", "large")),
        "nose_length": _classify(length_ratio, 0.32, 0.42, ("short", "balanced", "long")),
        "nose_width": _classify(width_ratio, 0.24, 0.31, ("narrow", "balanced", "wide")),
        "bridge": _classify(bridge_ratio, 0.42, 0.58, ("slim", "balanced", "broad")),
        "projection": _classify(projection_ratio, 0.04, 0.1, ("low", "balanced", "high")),
        "metrics": {
            "nose_width_face_ratio": round(width_ratio, 4),
            "nose_length_face_ratio": round(length_ratio, 4),
            "bridge_nose_ratio": round(bridge_ratio, 4),
            "projection_ratio": round(projection_ratio, 4),
        },
    }

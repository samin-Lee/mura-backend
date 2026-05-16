from math import hypot


def _distance(a, b) -> float:
    # 정규화된 2D 좌표 기준으로 두 랜드마크 사이의 거리를 구합니다.
    return hypot(a.x - b.x, a.y - b.y)


def _avg(values) -> float:
    # 좌우 눈처럼 여러 측정값을 안정적으로 평균 내기 위한 유틸입니다.
    values = list(values)
    return sum(values) / max(len(values), 1)


def _classify(value: float, low: float, high: float, labels: tuple[str, str, str]) -> str:
    # 계산된 비율을 낮음/보통/높음 계열의 문자열 라벨로 변환합니다.
    if value < low:
        return labels[0]
    if value > high:
        return labels[2]
    return labels[1]


def _eye_metrics(landmarks, outer: int, inner: int, upper: tuple[int, int], lower: tuple[int, int]):
    # 눈의 좌우 끝점과 위아래 눈꺼풀 점을 이용해 눈 너비, 높이, 개안 비율을 계산합니다.
    width = _distance(landmarks[outer], landmarks[inner])
    height = _avg(
        [
            _distance(landmarks[upper[0]], landmarks[lower[0]]),
            _distance(landmarks[upper[1]], landmarks[lower[1]]),
        ]
    )
    return width, height, height / max(width, 1e-6)


def analyze(landmarks):
    # 왼쪽 눈과 오른쪽 눈을 각각 측정한 뒤 평균값으로 전체 눈 특성을 판단합니다.
    left_width, left_height, left_open_ratio = _eye_metrics(landmarks, 33, 133, (159, 158), (145, 153))
    right_width, right_height, right_open_ratio = _eye_metrics(landmarks, 362, 263, (386, 385), (374, 380))
    eye_width = _avg([left_width, right_width])
    eye_height = _avg([left_height, right_height])
    eye_open_ratio = _avg([left_open_ratio, right_open_ratio])

    face_width = _distance(landmarks[234], landmarks[454])
    inner_eye_spacing = _distance(landmarks[133], landmarks[362])
    eye_spacing_ratio = inner_eye_spacing / max(eye_width, 1e-6)
    eye_face_ratio = eye_width / max(face_width, 1e-6)

    # 눈꺼풀 상단과 눈썹 랜드마크 사이의 거리를 재서 눈-눈썹 간격을 계산합니다.
    left_brow_distance = _avg(
        [
            _distance(landmarks[159], landmarks[65]),
            _distance(landmarks[158], landmarks[105]),
        ]
    )
    right_brow_distance = _avg(
        [
            _distance(landmarks[386], landmarks[295]),
            _distance(landmarks[385], landmarks[334]),
        ]
    )
    brow_eye_distance = _avg([left_brow_distance, right_brow_distance])
    brow_eye_ratio = brow_eye_distance / max(eye_height, 1e-6)

    # 얼굴 폭 대비 눈 크기, 눈 너비 대비 눈 사이 거리 등을 라벨과 지표로 반환합니다.
    return {
        "eye_size": _classify(eye_face_ratio, 0.16, 0.2, ("small", "balanced", "large")),
        "eye_opening": _classify(eye_open_ratio, 0.22, 0.32, ("narrow", "balanced", "open")),
        "eye_spacing": _classify(eye_spacing_ratio, 0.85, 1.15, ("close", "balanced", "wide")),
        "eye_eyebrow_distance": _classify(brow_eye_ratio, 1.1, 1.8, ("low", "balanced", "high")),
        "metrics": {
            "eye_face_ratio": round(eye_face_ratio, 4),
            "eye_open_ratio": round(eye_open_ratio, 4),
            "eye_spacing_ratio": round(eye_spacing_ratio, 4),
            "eye_eyebrow_distance_ratio": round(brow_eye_ratio, 4),
        },
    }

from math import hypot


def _distance(a, b) -> float:
    # 정규화된 2D 좌표 기준으로 두 랜드마크 사이의 거리를 계산합니다.
    return hypot(a.x - b.x, a.y - b.y)


def _avg(values) -> float:
    # 여러 입술 두께 측정값을 평균 내 안정적인 결과를 만듭니다.
    values = list(values)
    return sum(values) / max(len(values), 1)


def _classify(value: float, low: float, high: float, labels: tuple[str, str, str]) -> str:
    # 계산된 비율을 낮음/보통/높음 계열 라벨로 변환합니다.
    if value < low:
        return labels[0]
    if value > high:
        return labels[2]
    return labels[1]


def analyze(landmarks):
    # 얼굴 폭 대비 입 너비를 계산해 입 크기를 상대적으로 평가합니다.
    face_width = _distance(landmarks[234], landmarks[454])
    mouth_width = _distance(landmarks[61], landmarks[291])
    # 윗입술은 중앙과 양쪽 지점을 함께 사용해 두께 편차를 줄입니다.
    upper_lip_height = _avg(
        [
            _distance(landmarks[13], landmarks[0]),
            _distance(landmarks[82], landmarks[87]),
            _distance(landmarks[312], landmarks[317]),
        ]
    )
    # 아랫입술도 중앙과 양쪽 지점을 함께 측정해 평균 두께를 구합니다.
    lower_lip_height = _avg(
        [
            _distance(landmarks[14], landmarks[17]),
            _distance(landmarks[87], landmarks[178]),
            _distance(landmarks[317], landmarks[402]),
        ]
    )
    lip_height = upper_lip_height + lower_lip_height
    mouth_opening = _distance(landmarks[13], landmarks[14])
    # 양쪽 입꼬리의 y 좌표 차이를 이용해 입꼬리 방향을 추정합니다.
    corner_slope = (landmarks[291].y - landmarks[61].y) / max(mouth_width, 1e-6)

    width_ratio = mouth_width / max(face_width, 1e-6)
    lip_fullness_ratio = lip_height / max(mouth_width, 1e-6)
    upper_lower_ratio = upper_lip_height / max(lower_lip_height, 1e-6)
    opening_ratio = mouth_opening / max(lip_height, 1e-6)

    # 정규화 좌표에서 y가 작을수록 위쪽이므로 slope가 음수면 오른쪽 입꼬리가 올라간 형태입니다.
    if corner_slope < -0.03:
        corner_shape = "upturned"
    elif corner_slope > 0.03:
        corner_shape = "downturned"
    else:
        corner_shape = "balanced"

    # 입 크기, 입술 두께, 위아래 균형, 입꼬리, 벌어짐 정도를 반환합니다.
    return {
        "mouth_size": _classify(width_ratio, 0.34, 0.44, ("small", "balanced", "wide")),
        "lip_fullness": _classify(lip_fullness_ratio, 0.18, 0.28, ("thin", "balanced", "full")),
        "upper_lower_balance": _classify(
            upper_lower_ratio,
            0.65,
            0.95,
            ("lower_lip_dominant", "balanced", "upper_lip_dominant"),
        ),
        "mouth_corner": corner_shape,
        "mouth_opening": _classify(opening_ratio, 0.18, 0.38, ("closed", "relaxed", "open")),
        "metrics": {
            "mouth_width_face_ratio": round(width_ratio, 4),
            "lip_fullness_ratio": round(lip_fullness_ratio, 4),
            "upper_lower_lip_ratio": round(upper_lower_ratio, 4),
            "mouth_opening_ratio": round(opening_ratio, 4),
        },
    }

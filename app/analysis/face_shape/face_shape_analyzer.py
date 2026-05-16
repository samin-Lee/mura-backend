from math import hypot


def _distance(a, b) -> float:
    # 정규화된 2D 좌표 기준으로 두 랜드마크 사이의 유클리드 거리를 계산합니다.
    return hypot(a.x - b.x, a.y - b.y)


def _width(landmarks, left: int, right: int) -> float:
    # 좌우 랜드마크 인덱스를 받아 특정 얼굴 부위의 가로 폭을 계산합니다.
    return _distance(landmarks[left], landmarks[right])


def _label(value: float, small: float, large: float, labels: tuple[str, str, str]) -> str:
    # 비율값을 낮음/보통/높음 범주로 변환하는 공통 분류 함수입니다.
    if value < small:
        return labels[0]
    if value > large:
        return labels[2]
    return labels[1]


def analyze(landmarks):
    # 얼굴 세로 길이와 주요 가로 폭을 측정해 얼굴형 판단의 기본 비율을 만듭니다.
    face_height = _distance(landmarks[10], landmarks[152])
    face_width = _width(landmarks, 234, 454)
    cheekbone_width = _width(landmarks, 127, 356)
    forehead_width = _width(landmarks, 67, 297)
    jaw_width = _width(landmarks, 172, 397)

    length_width_ratio = face_height / max(face_width, 1e-6)
    jaw_face_ratio = jaw_width / max(face_width, 1e-6)
    cheek_face_ratio = cheekbone_width / max(face_width, 1e-6)
    forehead_jaw_ratio = forehead_width / max(jaw_width, 1e-6)
    cheek_jaw_ratio = cheekbone_width / max(jaw_width, 1e-6)

    # 얼굴 길이, 턱 폭, 광대 폭, 이마-턱 비율을 조합해 대표 얼굴형을 분류합니다.
    if length_width_ratio >= 1.62:
        shape = "long"
    elif length_width_ratio <= 1.28 and jaw_face_ratio >= 0.78:
        shape = "round"
    elif jaw_face_ratio >= 0.82 and length_width_ratio <= 1.5:
        shape = "square"
    elif forehead_jaw_ratio >= 1.12 and jaw_face_ratio < 0.78:
        shape = "heart"
    elif cheek_jaw_ratio >= 1.2 and cheek_face_ratio >= 0.94:
        shape = "diamond"
    else:
        shape = "oval"

    # 최종 라벨과 원본 비율 지표를 함께 반환해 클라이언트가 UI에 활용할 수 있게 합니다.
    return {
        "shape": shape,
        "face_length": _label(length_width_ratio, 1.3, 1.6, ("short", "balanced", "long")),
        "jawline": _label(jaw_face_ratio, 0.72, 0.84, ("soft", "balanced", "prominent")),
        "cheekbone": _label(cheek_face_ratio, 0.86, 0.96, ("subtle", "balanced", "prominent")),
        "forehead": _label(forehead_jaw_ratio, 0.95, 1.12, ("narrow", "balanced", "wide")),
        "metrics": {
            "length_width_ratio": round(length_width_ratio, 4),
            "jaw_face_ratio": round(jaw_face_ratio, 4),
            "cheek_face_ratio": round(cheek_face_ratio, 4),
            "forehead_jaw_ratio": round(forehead_jaw_ratio, 4),
        },
    }

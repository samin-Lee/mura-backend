def analyze(image_bgr):
    # 현재 MVP 범위는 얼굴 지오메트리 분석이므로 퍼스널 컬러는 확장 지점만 유지합니다.
    return {
        "available": False,
        "message": "personal_color analysis is not part of the current face geometry response",
    }

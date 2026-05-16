from typing import Sequence

from fastapi import HTTPException, status

from app.analysis.eyes import eyes_analyzer
from app.analysis.face_shape import face_shape_analyzer
from app.analysis.mouth import mouse_analyzer
from app.analysis.nose import nose_analyzer
from app.common.mediapipe_loader import get_face_landmarks


def _bounding_box(landmarks: Sequence) -> dict[str, float]:
    # 모든 랜드마크 좌표의 최소/최대값으로 정규화된 얼굴 영역을 계산합니다.
    xs = [point.x for point in landmarks]
    ys = [point.y for point in landmarks]
    left = min(xs)
    top = min(ys)
    right = max(xs)
    bottom = max(ys)
    return {
        "x": round(left, 6),
        "y": round(top, 6),
        "width": round(right - left, 6),
        "height": round(bottom - top, 6),
    }


def _serialize_landmarks(landmarks: Sequence) -> list[dict[str, float]]:
    # MediaPipe 랜드마크 객체를 JSON으로 직렬화 가능한 dict 목록으로 변환합니다.
    return [
        {
            "index": index,
            "x": round(point.x, 6),
            "y": round(point.y, 6),
            "z": round(point.z, 6),
            "visibility": round(getattr(point, "visibility", 0.0), 6),
            "presence": round(getattr(point, "presence", 0.0), 6),
        }
        for index, point in enumerate(landmarks)
    ]


async def run_analysis_pipeline(image_bgr):
    # 공통 MediaPipe 로더를 통해 얼굴 랜드마크를 한 번만 추출합니다.
    landmarks = get_face_landmarks(image_bgr)
    if not landmarks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Face not detected",
        )

    # 이미지 메타데이터와 각 분석기 결과를 하나의 응답 구조로 조립합니다.
    image_height, image_width = image_bgr.shape[:2]
    return {
        "success": True,
        "message": "analysis completed",
        "image": {
            "width": int(image_width),
            "height": int(image_height),
            "channels": int(image_bgr.shape[2]) if len(image_bgr.shape) == 3 else 1,
        },
        "detection": {
            "detected": True,
            "landmark_count": len(landmarks),
            "bounding_box": _bounding_box(landmarks),
        },
        "analysis": {
            # 각 분석기는 같은 랜드마크 입력을 받아 독립적으로 결과를 계산합니다.
            "face": face_shape_analyzer.analyze(landmarks),
            "eyes": eyes_analyzer.analyze(landmarks),
            "nose": nose_analyzer.analyze(landmarks),
            "mouth": mouse_analyzer.analyze(landmarks),
        },
        "landmarks": _serialize_landmarks(landmarks),
    }

import imghdr

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status

from app.analysis.face_pipeline import run_analysis_pipeline


class FaceAnalysisService:
    async def analyze_upload(self, file: UploadFile) -> dict:
        # 업로드된 파일 전체를 읽어 이미지 검증과 디코딩을 수행합니다.
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty",
            )

        # MIME 타입과 실제 파일 시그니처를 함께 확인해 잘못된 파일 업로드를 막습니다.
        if not self._is_supported_image(file.content_type, content):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only JPEG, PNG, BMP, and WebP images are supported",
            )

        # OpenCV가 처리할 수 있는 BGR 이미지 배열로 변환한 뒤 분석 파이프라인에 넘깁니다.
        image = self._decode_image(content)
        return await run_analysis_pipeline(image)

    @staticmethod
    def _is_supported_image(content_type: str | None, content: bytes) -> bool:
        # 일부 모바일 클라이언트는 application/octet-stream으로 이미지를 보내므로 허용 목록에 포함합니다.
        allowed_content_types = {
            "image/jpeg",
            "image/jpg",
            "image/png",
            "image/bmp",
            "image/webp",
            "application/octet-stream",
            None,
        }
        # imghdr로 파일 내용 기반 이미지 형식을 확인합니다.
        detected_type = imghdr.what(None, content)
        return content_type in allowed_content_types and detected_type in {"jpeg", "png", "bmp", "webp"}

    @staticmethod
    def _decode_image(content: bytes):
        # bytes를 numpy 버퍼로 바꾼 뒤 OpenCV imdecode로 이미지 행렬을 생성합니다.
        buffer = np.frombuffer(content, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not decode uploaded image",
            )
        return image


# 라우터에서 재사용할 서비스 인스턴스입니다.
face_analysis_service = FaceAnalysisService()

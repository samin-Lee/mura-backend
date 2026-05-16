from fastapi import APIRouter, File, UploadFile

from app.models.analysis_response import AnalysisResponse
from app.services.face_service import face_analysis_service


# 얼굴 분석 관련 REST API를 모아두는 라우터입니다.
router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("/face", response_model=AnalysisResponse)
async def analyze_face(file: UploadFile = File(...)):
    # Android 앱이 multipart/form-data로 업로드한 이미지를 서비스 계층으로 전달합니다.
    return await face_analysis_service.analyze_upload(file)


@router.post("", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)):
    # `/analysis` 경로도 동일한 얼굴 분석 기능을 제공해 클라이언트 연동을 단순화합니다.
    return await face_analysis_service.analyze_upload(file)

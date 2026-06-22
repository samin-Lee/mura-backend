from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from datetime import datetime
from schemas import AnalysisResponse
from services.image_service import analyze_skin_from_r2
from analysis.personal_color.personal_color_analyzer import (
    classify_skin_tone,
    classify_brightness,
    classify_season,
    recommend_makeup,
    load_all_data,
    save_all_data
)

router = APIRouter()

class AnalysisRequest(BaseModel):
    file_key: str  
    user_id: str

@router.post("/analyze", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
async def upload_and_analyze(payload: AnalysisRequest):
    # Cloudflare R2에서 이미지 분석 및 OpenCV 처리
    try:
        skin = await analyze_skin_from_r2(payload.file_key)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"이미지 분석 중 서버 에러가 발생했습니다: {str(e)}"
        )
    
    if not skin or "error" in skin:
        error_msg = skin.get("error", "피부 톤 분석에 실패했습니다.") if isinstance(skin, dict) else "피부 톤 분석 실패"
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"이미지 분석 실패: {error_msg}. 얼굴이 명확하게 나온 사진으로 다시 시도해 주세요."
        )

    # 퍼스널 컬러 알고리즘 분석
    season = classify_season(skin)
    tone = classify_skin_tone(skin)
    brightness = classify_brightness(skin)

    # 맞춤 메이크업 추천룩 생성
    makeup = recommend_makeup(season)

    # 분석 결과와 추천룩을 데이터베이스에 저장
    new_record = {
        "user_id": payload.user_id,
        "file_key": payload.file_key,
        "lab_values": skin,
        "analysis_result": {
            "tone": tone,
            "brightness": brightness,
            "season": season,
        },
        "makeup_recommendation": makeup,
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 예외 처리(try-except)로 안전하게 저장하기
    try:
        db_data = load_all_data()
        db_data.append(new_record)
        save_all_data(db_data)
    except Exception as e:
        print(f" [데이터베이스 저장 실패]: {e}")

    # 5. 결과 반환
    return {
        "status": "completed",
        "season": season,
        "tone": tone,
        "makeup_recommendation": makeup
    }

@router.get("/recommendations/{user_id}")
async def get_user_recommendations(user_id: str):
    db_data = load_all_data()

    user_records = [record for record in db_data if record["user_id"] == user_id]

    if not user_records:
        raise HTTPException(
            status_code=404,
            detail=f"User '{user_id}'에 대한 분석 기록을 찾을 수 없습니다."
        )
    return {
        "status": "success",
        "total_records": len(user_records),
        "history": user_records
    }
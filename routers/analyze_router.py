from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from schemas import AnalysisResponse
from services.image_service import analyze_skin_from_r2
from services.recommendation import load_all_data, recommend_makeup, save_all_data


router = APIRouter()


class AnalysisRequest(BaseModel):
    file_key: str
    user_id: str


@router.post("/analyze", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
async def upload_and_analyze(payload: AnalysisRequest):
    try:
        analysis_result = await analyze_skin_from_r2(payload.file_key)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"이미지 분석 중 서버 오류가 발생했습니다: {str(e)}"
        )

    if not analysis_result or "error" in analysis_result:
        error_msg = (
            analysis_result.get("error", "분석에 실패했습니다.")
            if isinstance(analysis_result, dict)
            else "분석 실패"
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"이미지 분석 실패: {error_msg}. 얼굴이 명확하게 나온 사진으로 다시 시도해 주세요.",
        )

    lab_values = analysis_result["lab_values"]
    tone = analysis_result["tone"]
    brightness = analysis_result["brightness"]
    season = analysis_result["season"]
    makeup = recommend_makeup(tone, brightness)

    new_record = {
        "user_id": payload.user_id,
        "file_key": payload.file_key,
        "lab_values": lab_values,
        "analysis_result": analysis_result,
        "makeup_recommendation": makeup,
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 예외 처리(try-except)로 안전하게 저장하기
    try:
        db_data = load_all_data()
        db_data.append(new_record)
        save_all_data(db_data)
    except Exception as e:
        print(f"[데이터베이스 저장 실패]: {e}")

    # 결과 반환
    return {
        "status": "completed",
        "season": season,
        "tone": tone,
        "makeup_recommendation": makeup,
    }


@router.get("/recommendations/{user_id}")
async def get_user_recommendations(user_id: str):
    db_data = load_all_data()
    user_records = [record for record in db_data if record["user_id"] == user_id]

    if not user_records:
        raise HTTPException(
            status_code=404,
            detail=f"User '{user_id}'에 대한 분석 기록을 찾을 수 없습니다.",
        )

    return {
        "status": "success",
        "total_records": len(user_records),
        "history": user_records,
    }

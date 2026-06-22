from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from schemas import AnalysisResponse
from services.history_service import load_all_data, save_all_data
from services.image_service import analyze_skin_from_r2
from services.recommended_data import recommend_makeup


router = APIRouter()


class AnalysisRequest(BaseModel):
    file_key: str
    user_id: str


@router.post("/analyze", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
async def upload_and_analyze(payload: AnalysisRequest):
    try:
        analysis_result = await analyze_skin_from_r2(payload.file_key)
    except ValueError as exc:
        error_msg = str(exc).rstrip(".")
        if error_msg.startswith("이미지 분석 실패:"):
            error_msg = error_msg.split(":", 1)[1].strip()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"이미지 분석 실패: {error_msg}. 얼굴이 명확하게 나온 사진으로 다시 시도해 주세요.",
        ) from exc
    except Exception as exc:
        print(f"[analysis server error] {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="이미지 분석 중 서버 오류가 발생했습니다.",
        ) from exc

    response = {
        **analysis_result,
        "makeup_recommendation": recommend_makeup(analysis_result["season"]),
    }

    new_record = {
        "user_id": payload.user_id,
        "file_key": payload.file_key,
        "analysis_result": response,
        "makeup_recommendation": response["makeup_recommendation"],
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        db_data = load_all_data()
        db_data.append(new_record)
        save_all_data(db_data)
    except Exception as exc:
        print(f"[데이터베이스 저장 실패]: {exc}")

    return response


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

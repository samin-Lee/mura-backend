from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from schemas import AnalysisResponse
from services.history_service import load_all_data, save_all_data
from services.image_service import AnalysisProxyError, analyze_skin_from_r2
from services.recommended_data import recommend_makeup


router = APIRouter()


class AnalysisRequest(BaseModel):
    file_key: str
    user_id: str


@router.post("/analyze", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
async def upload_and_analyze(payload: AnalysisRequest):
    try:
        print(
            f"[analysis router] request received user_id={payload.user_id}, "
            f"file_key={payload.file_key}"
        )
        analysis_result = await analyze_skin_from_r2(payload.file_key)
    except ValueError as exc:
        error_msg = str(exc).rstrip(".")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"image analysis failed: {error_msg}",
        ) from exc
    except AnalysisProxyError as exc:
        print(
            "[analysis upstream error] "
            f"status={exc.status_code}, content_type={exc.content_type}, "
            f"response_text={exc.response_text}"
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail=f"analysis server returned {exc.status_code}: {exc.response_text}",
        ) from exc
    except Exception as exc:
        print(f"[analysis server error] {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="image analysis server error",
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
        print(f"[database save failed] {exc}")

    return response


@router.get("/recommendations/{user_id}")
async def get_user_recommendations(user_id: str):
    db_data = load_all_data()
    user_records = [record for record in db_data if record["user_id"] == user_id]

    if not user_records:
        raise HTTPException(
            status_code=404,
            detail=f"no analysis history found for user '{user_id}'",
        )

    return {
        "status": "success",
        "total_records": len(user_records),
        "history": user_records,
    }

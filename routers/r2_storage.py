from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.r2_service import generate_presigned_url

router = APIRouter()

class UrlRequest(BaseModel):
    file_extension: str  # 예: "jpg", "png"

@router.post("/presigned-url")
def get_upload_url(request: UrlRequest):
    result = generate_presigned_url(request.file_extension)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result
import os

import httpx

from services.r2_image_service import download_image_from_r2


ANALYSIS_SERVER_URL = os.getenv(
    "ANALYSIS_SERVER_URL",
    "https://leesy23-mura-analysis.hf.space",
).rstrip("/")
HF_ANALYSIS_TOKEN = os.getenv("HF_ANALYSIS_TOKEN")
ANALYSIS_TIMEOUT_SECONDS = 300.0


async def analyze_skin_from_r2(file_key: str):
    image_bytes = await download_image_from_r2(file_key)
    headers = {}
    if HF_ANALYSIS_TOKEN:
        headers["Authorization"] = f"Bearer {HF_ANALYSIS_TOKEN}"

    files = {
        "file": (
            file_key.rsplit("/", 1)[-1] or "image.jpg",
            image_bytes,
            "application/octet-stream",
        )
    }

    async with httpx.AsyncClient(timeout=ANALYSIS_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{ANALYSIS_SERVER_URL}/analyze",
            files=files,
            headers=headers,
        )

    if response.status_code == 422:
        detail = response.json().get("detail", "이미지 분석에 실패했습니다.")
        raise ValueError(detail)
    response.raise_for_status()
    return response.json()

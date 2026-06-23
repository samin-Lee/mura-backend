import os

import httpx

from services.r2_image_service import download_image_from_r2
from services.r2_image_service import R2ConfigError, R2DownloadError, R2ObjectNotFoundError


ANALYSIS_SERVER_URL = os.getenv(
    "ANALYSIS_SERVER_URL",
    "https://leesy23-mura-analysis.hf.space",
).rstrip("/")
HF_ANALYSIS_TOKEN = (os.getenv("HF_ANALYSIS_TOKEN") or "").strip() or None
ANALYSIS_TIMEOUT_SECONDS = 300.0


class AnalysisProxyError(Exception):
    def __init__(self, status_code: int, response_text: str, content_type: str | None = None):
        self.status_code = status_code
        self.response_text = response_text
        self.content_type = content_type
        message = f"analysis server returned {status_code}"
        if response_text:
            message = f"{message}: {response_text}"
        super().__init__(message)


class AnalysisRequestError(Exception):
    pass


async def analyze_skin_from_r2(file_key: str):
    print(
        f"[analysis proxy] start file_key={file_key}, "
        f"analysis_url={ANALYSIS_SERVER_URL}, token_present={bool(HF_ANALYSIS_TOKEN)}"
    )

    image_bytes = await download_image_from_r2(file_key)
    print(f"[analysis proxy] downloaded bytes={len(image_bytes)}")

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
        try:
            response = await client.post(
                f"{ANALYSIS_SERVER_URL}/analyze",
                files=files,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            print(f"[analysis proxy] request failed: {type(exc).__name__}: {exc}")
            raise AnalysisRequestError(f"failed to contact analysis server: {type(exc).__name__}: {exc}") from exc

    print(
        f"[analysis proxy] response status={response.status_code}, "
        f"content_type={response.headers.get('content-type')}"
    )

    if response.status_code >= 400:
        print(f"[analysis proxy] error body={response.text}")

    if response.status_code == 422:
        try:
            detail = response.json().get("detail", "image analysis failed")
        except ValueError:
            detail = response.text or "image analysis failed"
        raise ValueError(detail)

    if response.status_code >= 400:
        raise AnalysisProxyError(
            status_code=response.status_code,
            response_text=response.text,
            content_type=response.headers.get("content-type"),
        )

    return response.json()

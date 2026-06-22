from starlette.concurrency import run_in_threadpool

from analysis.analysis_pipeline import analyze_image_bytes
from services.r2_image_service import download_image_from_r2


async def analyze_skin_from_r2(file_key: str):
    image_bytes = await download_image_from_r2(file_key)
    return await run_in_threadpool(analyze_image_bytes, image_bytes)

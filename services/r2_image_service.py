from starlette.concurrency import run_in_threadpool

from services.r2_service import R2_BUCKET_NAME, s3_client


def _download_image_from_r2_sync(file_key: str) -> bytes:
    response = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=file_key)
    return response["Body"].read()


async def download_image_from_r2(file_key: str) -> bytes:
    return await run_in_threadpool(_download_image_from_r2_sync, file_key)

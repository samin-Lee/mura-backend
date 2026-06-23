from starlette.concurrency import run_in_threadpool

from services.r2_service import R2_BUCKET_NAME, s3_client


class R2DownloadError(Exception):
    pass


class R2ObjectNotFoundError(R2DownloadError):
    pass


class R2ConfigError(R2DownloadError):
    pass


def _download_image_from_r2_sync(file_key: str) -> bytes:
    try:
        response = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=file_key)
        return response["Body"].read()
    except Exception as exc:
        error_name = type(exc).__name__
        message = str(exc)
        if error_name in {"NoCredentialsError", "PartialCredentialsError"}:
            raise R2ConfigError(f"R2 credentials are missing or incomplete: {message}") from exc
        if error_name == "ClientError" and ("NoSuchKey" in message or "404" in message):
            raise R2ObjectNotFoundError(f"R2 object not found for key '{file_key}': {message}") from exc
        raise R2DownloadError(f"Failed to download '{file_key}' from R2: {error_name}: {message}") from exc


async def download_image_from_r2(file_key: str) -> bytes:
    return await run_in_threadpool(_download_image_from_r2_sync, file_key)

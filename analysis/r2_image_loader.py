from services.r2_service import R2_BUCKET_NAME, s3_client


async def download_image_from_r2(file_key: str) -> bytes:
    response = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=file_key)
    return response["Body"].read()

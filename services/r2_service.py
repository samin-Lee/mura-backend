import os
import uuid


R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")


def _validate_r2_config():
    missing = [
        name
        for name, value in {
            "R2_ACCOUNT_ID": R2_ACCOUNT_ID,
            "R2_ACCESS_KEY_ID": R2_ACCESS_KEY,
            "R2_SECRET_ACCESS_KEY": R2_SECRET_ACCESS_KEY,
            "R2_BUCKET_NAME": R2_BUCKET_NAME,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing R2 environment variables: {', '.join(missing)}")


def _get_s3_client():
    _validate_r2_config()

    import boto3
    from botocore.client import Config

    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


class LazyS3Client:
    def get_object(self, *args, **kwargs):
        return _get_s3_client().get_object(*args, **kwargs)

    def generate_presigned_url(self, *args, **kwargs):
        return _get_s3_client().generate_presigned_url(*args, **kwargs)


s3_client = LazyS3Client()


def generate_presigned_url(file_extension: str) -> dict:
    unique_filename = f"uploads/{uuid.uuid4()}.{file_extension}"

    try:
        presigned_url = s3_client.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": R2_BUCKET_NAME,
                "Key": unique_filename,
                "ContentType": f"image/{file_extension}",
            },
            ExpiresIn=300,
        )
        return {
            "upload_url": presigned_url,
            "file_key": unique_filename,
        }
    except Exception as exc:
        print(f"Pre-signed URL 발급 오류: {exc}")
        return {"error": "URL 발급 실패"}

from services.r2_service import s3_client, R2_BUCKET_NAME

# 상대 경로 방식 (현재 위치 기준으로 상위의 analysis/analysis 폴더를 찾음)
from analysis.image_analyzer import decode_image, extract_skin_region
from analysis.lab_analyzer import calculate_lab_average

async def analyze_skin_from_r2(file_key: str):
    try:
        # Cloudflare R2에서 이미지 파일 받아오기
        response = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=file_key)
        image_bytes = response['Body'].read()
        image = decode_image(image_bytes)
        skin_region = extract_skin_region(image)
        skin_result = calculate_lab_average(skin_region)
        
        return skin_result 
        
    except ValueError as ve:
        print(f"이미지 디코딩 실패 에러: {ve}")
        return {"error": str(ve)}
    except Exception as e:
        print(f"이미지 분석 처리 에러: {e}")
        return {"error": "분석 중 예외 발생"}
import cv2
import numpy as np
from services.r2_service import s3_client, R2_BUCKET_NAME

async def analyze_skin_from_r2(file_key: str):
    try:
        # Cloudflare R2에서 이미지 파일 받아오기
        response = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=file_key)
        image_bytes = response['Body'].read()
        
        # 바이트 데이터를 OpenCV 형식 이미지로 변환
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            print("OpenCV 이미지 디코딩 실패")
            return {"error": "이미지 읽기 실패"}
        
        # 얼굴 정면 탐지 (Haar Cascade)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # 얼굴 영역 크롭 결정 (버그 수정 완료)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = image[y:y+h, x:x+w]
        else:
            face = image  # 얼굴 인식 실패 시 전체 이미지 사용

        # 크롭된 영역의 중심부 추출 (피부 영역 확률 극대화)
        h, w, _ = face.shape
        center = face[h//4:h*3//4, w//4:w*3//4]

        # center 영역에 대한 CIE L*a*b* 평균값 계산
        lab = cv2.cvtColor(center, cv2.COLOR_BGR2LAB)
        avg_color = lab.mean(axis=0).mean(axis=0)

        return {
            "L": int(avg_color[0]),
            "A": int(avg_color[1]),
            "B": int(avg_color[2])
        }
        
    except Exception as e:
        print(f"이미지 분석 처리 에러: {e}")
        return {"error": "분석 중 예외 발생"}
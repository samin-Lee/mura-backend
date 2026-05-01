from app.common.mediapipe_loader import get_face_landmarks
from app.analysis.face_shape import face_shape_analyzer
# 나중에 여기에 eyes, nose 등 추가 가능

async def run_analysis_pipeline(img):
    # 1. 랜드마크 딱 한 번만 추출
    landmarks = get_face_landmarks(img)
    if not landmarks:
        return {"error": "Face not detected"}

    # 2. 얼굴형 분석기 실행
    face_shape_results = face_shape_analyzer.analyze(landmarks)
    
    # 3. 결과 합치기 (나중에 눈, 코, 입 결과도 여기에 추가)
    total_results = {
        "face_shape": face_shape_results
    }
    
    return total_results

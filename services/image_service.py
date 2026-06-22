from analysis.analysis_pipeline import analyze_image_from_r2


async def analyze_skin_from_r2(file_key: str):
    try:
        return await analyze_image_from_r2(file_key)
    except Exception as exc:
        print(f"이미지 분석 처리 오류: {exc}")
        return {"error": "분석 중 예외 발생"}

from analysis.analysis_pipeline import analyze_image_from_r2


async def analyze_skin_from_r2(file_key: str):
    return await analyze_image_from_r2(file_key)

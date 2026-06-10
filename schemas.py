from pydantic import BaseModel
from typing import List

class MakeupRecommendation(BaseModel):
    lip: List[str]
    blush: List[str]
    eye: List[str]

class AnalysisResponse(BaseModel):
    status: str
    season: str
    tone: str
    face_shape: dict
    eye_metrics: dict
    nose_metrics: dict
    makeup_recommendation: MakeupRecommendation

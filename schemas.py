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
    makeup_recommendation: MakeupRecommendation

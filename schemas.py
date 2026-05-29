from pydantic import BaseModel
from typing import List

class MakeupRecomendation(BaseModel):
    lip: List[str]
    blush: List[str]
    eye: List[str]

class AnalysisResponse(BaseModel):
    status: str
    season: str
    tone: str
    makeup_recommendation: MakeupRecomendation
from typing import List

from pydantic import BaseModel, Field


class ProductRecommendation(BaseModel):
    category: str
    brand: str
    name: str
    reason: str


class MakeupRecommendation(BaseModel):
    look_name: str
    description: str
    worst_colors: List[str] = Field(default_factory=list)
    products: List[ProductRecommendation]


class AnalysisResponse(BaseModel):
    status: str
    season: str
    tone: str
    face_shape: dict
    eye_metrics: dict
    nose_metrics: dict
    mouth_metrics: dict
    makeup_recommendation: MakeupRecommendation

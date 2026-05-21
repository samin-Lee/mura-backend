from typing import Dict, List

from pydantic import BaseModel, Field


class LandmarkPoint(BaseModel):
    # MediaPipe FaceMesh의 개별 랜드마크 좌표를 클라이언트가 재사용할 수 있게 제공합니다.
    index: int
    x: float
    y: float
    z: float
    visibility: float = 0.0
    presence: float = 0.0


class FaceDetection(BaseModel):
    # 얼굴 검출 여부와 검출된 얼굴의 정규화 bounding box 정보를 담습니다.
    detected: bool
    landmark_count: int
    bounding_box: Dict[str, float]


class FaceShapeAnalysis(BaseModel):
    # 얼굴형 분석 결과와 관련 비율 지표입니다.
    shape: str
    face_length: str
    jawline: str
    cheekbone: str
    forehead: str
    metrics: Dict[str, float]


class EyesAnalysis(BaseModel):
    # 눈 크기, 눈 사이 거리, 눈썹과 눈 사이 거리 분석 결과입니다.
    eye_size: str
    eye_opening: str
    eye_spacing: str
    eye_eyebrow_distance: str
    metrics: Dict[str, float]


class NoseAnalysis(BaseModel):
    # 코의 크기, 길이, 너비, 콧대, 돌출 정도 분석 결과입니다.
    nose_size: str
    nose_length: str
    nose_width: str
    bridge: str
    projection: str
    metrics: Dict[str, float]


class MouthAnalysis(BaseModel):
    # 입 크기, 입술 두께, 위아래 입술 균형, 입꼬리 분석 결과입니다.
    mouth_size: str
    lip_fullness: str
    upper_lower_balance: str
    mouth_corner: str
    mouth_opening: str
    metrics: Dict[str, float]


class LabColor(BaseModel):
    # Lab 색 공간에서 특정 영역의 평균 L, a, b 값을 표현합니다.
    l: float
    a: float
    b: float


class PersonalColorLab(BaseModel):
    # 퍼스널컬러 분석에 사용한 피부, 눈동자, 모발 영역의 Lab 평균값입니다.
    skin: LabColor
    iris: LabColor
    hair: LabColor


class PersonalColorAnalysis(BaseModel):
    # Lab 기반 웜톤/쿨톤과 대비 기반 사계절 퍼스널컬러 분석 결과입니다.
    available: bool
    tone: str
    season: str
    contrast_level: str
    lab: PersonalColorLab
    metrics: Dict[str, float]


class FaceAnalysisData(BaseModel):
    # 기능 요구사항의 Face/Eyes/Nose/Mouth 분석 결과를 묶는 객체입니다.
    face: FaceShapeAnalysis
    eyes: EyesAnalysis
    nose: NoseAnalysis
    mouth: MouthAnalysis
    personal_color: PersonalColorAnalysis


class AnalysisResponse(BaseModel):
    # `/analysis` API가 반환하는 최상위 JSON 응답 모델입니다.
    success: bool = True
    message: str = "analysis completed"
    image: Dict[str, int]
    detection: FaceDetection
    analysis: FaceAnalysisData
    landmarks: List[LandmarkPoint] = Field(default_factory=list)

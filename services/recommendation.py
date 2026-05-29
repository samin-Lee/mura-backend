import json
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()
DATA_FILE_PATH = "data/recommendations.json"

# 폴더가 없을 경우 자동 생성
os.makedirs(os.path.dirname(DATA_FILE_PATH), exist_ok=True)

class LabInput(BaseModel):
    user_id: str
    L: float
    A: float
    B: float

def classify_skin_tone(lab):
    if lab["A"] > 138:
        return "웜톤"
    elif lab["A"] < 132:
        return "쿨톤"
    else:
        return "중립"

def classify_brightness(lab):
    if lab["L"] > 180:
        return "밝음"
    elif lab["L"] > 120:
        return "중간"
    else:
        return "어두움"

def recommend_makeup(tone, brightness):
    if tone == "웜톤":
        if brightness == "밝음":
            return {
                "lip": ["코랄", "피치"],
                "blush": ["살구"],
                "eye": ["골드", "브라운"]
            }
        else:
            return {
                "lip": ["오렌지", "브릭"],
                "blush": ["오렌지"],
                "eye": ["브라운", "카키"]
            }
    elif tone == "쿨톤": 
        if brightness == "밝음":
            return {
                "lip": ["핑크", "로즈"],
                "blush": ["라벤더"],
                "eye": ["카키", "그레이"]
            }
        else:
            return {
                "lip": ["버건디", "플럼"],
                "blush": ["와인"],
                "eye": ["퍼플", "네이비"]
            }
    else:  # 💡 중립(뉴트럴) 톤일 때의 예외 방어 코드 추가
        return {
            "lip": ["로즈베이지", "말린장미"],
            "blush": ["누드핑크"],
            "eye": ["음영브라운", "로즈브라운"]
        }

def classify_season(lab):
    L = lab["L"]
    A = lab["A"]
    B = lab["B"]

    if A > 138 and B > 138:
        tone = "warm"
    elif A < 132 and B < 132:
        tone = "cool"
    else:
        tone = "neutral"

    if L > 170:
        brightness = "light"
    else:
        brightness = "dark"

    if tone == "warm" and brightness == "light":
        return "봄웜"
    elif tone == "warm" and brightness == "dark":
        return "가을웜"
    elif tone == "cool" and brightness == "dark":
        return "겨쿨"
    elif tone == "cool" and brightness == "light":
        return "여쿨"
    else:
        return "중립(뉴트럴톤)"


def load_all_data() -> list:
    if not os.path.exists(DATA_FILE_PATH):
        return []
    try:
        with open(DATA_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_all_data(data: list):
    with open(DATA_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
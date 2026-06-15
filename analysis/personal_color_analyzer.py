import json
import os

# 1. 데이터베이스 로드 및 저장 기능

DATA_FILE_PATH = "data/recommendations.json"

# 폴더가 없을 경우 자동 생성
os.makedirs(os.path.dirname(DATA_FILE_PATH), exist_ok=True)

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


# 퍼스널 컬러 분석 알고리즘

def classify_skin_tone(lab: dict) -> str:
    if lab["A"] > 138:
        return "웜톤"
    if lab["A"] < 132:
        return "쿨톤"
    return "중립"

def classify_brightness(lab: dict) -> str:
    if lab["L"] > 180:
        return "밝음"
    if lab["L"] > 120:
        return "중간"
    return "어두움"

def classify_season(lab: dict) -> str:
    l_value = lab["L"]
    a_value = lab["A"]
    b_value = lab["B"]

    if a_value > 138 and b_value > 138:
        tone = "warm"
    elif a_value < 132 and b_value < 132:
        tone = "cool"
    else:
        tone = "neutral"

    brightness = "light" if l_value > 170 else "dark"

    if tone == "warm" and brightness == "light":
        return "봄웜"
    if tone == "warm" and brightness == "dark":
        return "가을웜"
    if tone == "cool" and brightness == "dark":
        return "겨울쿨"
    if tone == "cool" and brightness == "light":
        return "여름쿨"
    return "중립"


def recommend_makeup(tone: str, brightness: str) -> dict:
    
    result = {
        "lip_color": "내추럴 핑크",
        "blusher": "피치 코랄",
        "shadow": "베이지 브라운",
        "tip": "자연스러운 피부 결을 살린 메이크업이 잘 어울립니다."
    }

    if tone == "웜톤":
        if brightness == "밝음":
            result = {
                "lip_color": "화사한 코랄 핑크, 살구 오렌지",
                "blusher": "라이트 피치, 살구빛 코랄",
                "shadow": "밀크티 베이지, 소프트 브라운",
                "tip": "글로시하고 생기 있는 투명 메이크업이 베스트입니다!"
            }
        else: # 중간이나 어두움
            result = {
                "lip_color": "칠리 레드, 브릭 오렌지, 말린 장미",
                "blusher": "소프트 오렌지, 누드 베이지",
                "shadow": "음영 브라운, 카키 베이지, 골드 펄",
                "tip": "매트하고 그윽한 가을 감성의 음영 메이크업이 아주 잘 어울려요."
            }
    elif tone == "쿨톤":
        if brightness == "밝음":
            result = {
                "lip_color": "딸기우유 핑크, 플럼 로즈, 체리 레드",
                "blusher": "라벤더 핑크, 파스텔 핑크",
                "shadow": "로즈 핑크, 소프트 그레이",
                "tip": "맑고 깨끗한 피부 톤을 강조하는 시원하고 투명한 메이크업을 추천합니다."
            }
        else: # 중간이나 어두움
            result = {
                "lip_color": "강렬한 푸시아 핑크, 버건디 레드, 딥 플럼",
                "blusher": "모브 핑크, 원색적인 핑크",
                "shadow": "실버 글리터, 클래식 네이비, 회갈색",
                "tip": "라인을 또렷하게 살린 세련되고 도시적인 포인트 메이크업이 찰떡입니다."
            }
    
    return result
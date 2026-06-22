import json
import os

from analysis.recommend_data import DEFAULT_RECOMMENDATION, MAKEUP_RECOMMENDATIONS


DATA_FILE_PATH = "data/recommendations.json"

os.makedirs(os.path.dirname(DATA_FILE_PATH), exist_ok=True)


def load_all_data() -> list:
    if not os.path.exists(DATA_FILE_PATH):
        return []
    try:
        with open(DATA_FILE_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return []


def save_all_data(data: list):
    with open(DATA_FILE_PATH, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def classify_skin_tone(lab: dict) -> str:
    if lab["A"] > 138:
        return "웜톤"
    if lab["A"] < 132:
        return "쿨톤"
    return "중립"


def recommend_makeup(season: str) -> dict:
    return MAKEUP_RECOMMENDATIONS.get(season, DEFAULT_RECOMMENDATION)


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

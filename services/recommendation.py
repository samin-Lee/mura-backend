import json
import os


DATA_FILE_PATH = "data/recommendations.json"

os.makedirs(os.path.dirname(DATA_FILE_PATH), exist_ok=True)


def recommend_makeup(tone: str, brightness: str) -> dict:
    if tone == "웜톤":
        if brightness == "밝음":
            return {
                "lip": ["코랄", "피치"],
                "blush": ["살구"],
                "eye": ["골드", "브라운"],
            }
        return {
            "lip": ["오렌지", "브릭"],
            "blush": ["오렌지"],
            "eye": ["브라운", "카키"],
        }

    if tone == "쿨톤":
        if brightness == "밝음":
            return {
                "lip": ["핑크", "로즈"],
                "blush": ["라벤더"],
                "eye": ["카키", "그레이"],
            }
        return {
            "lip": ["버건디", "플럼"],
            "blush": ["모브"],
            "eye": ["퍼플", "네이비"],
        }

    return {
        "lip": ["로즈베이지", "말린장미"],
        "blush": ["누드핑크"],
        "eye": ["소프트브라운", "로즈브라운"],
    }


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

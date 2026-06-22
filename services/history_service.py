import json
import os


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

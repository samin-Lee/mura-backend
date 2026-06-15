import json
import os

DATA_FILE_PATH = "data/recommendations.json"

# 폴더가 없을 경우 자동 생성
os.makedirs(os.path.dirname(DATA_FILE_PATH), exist_ok=True)

# 데이터베이스 로드 및 저장 기능
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
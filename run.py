import uvicorn


# 로컬 개발 환경에서 `python run.py`로 FastAPI 서버를 실행하는 진입점입니다.
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

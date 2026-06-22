import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.analyze_router import router as analysis_router 
from routers.r2_storage import router as r2_router

app = FastAPI(title="AI Makeup Recommendation Service")

CORS_ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials="*" not in CORS_ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(r2_router, prefix="/api/storage", tags=["Storage"])
app.include_router(analysis_router, prefix="/api/analysis", tags=["Analysis"])


@app.get("/")
def read_root():
    return {"message": "AI 메이크업 추천 서비스 서버가 정상 가동 중입니다."}

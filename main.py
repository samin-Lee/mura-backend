from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.analyze_router import router as analysis_router 
from routers.r2_storage import router as r2_router

app = FastAPI(title="AI Makeup Recommendation Service")

# 프론트엔드와 원활한 통신을 위해 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(r2_router, prefix="/api/storage", tags=["Storage"])
app.include_router(analysis_router, prefix="/api/analysis", tags=["Analysis"])

@app.get("/")
def read_root():
    return {"message": "AI 메이크업 추천 서비스 서버가 정상 가동 중입니다."}
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.analysis import router as analysis_router


# FastAPI 애플리케이션의 메타 정보와 전체 라우터를 구성합니다.
app = FastAPI(
    title="Mura Face Analysis Backend",
    description="FastAPI and MediaPipe backend for mobile face analysis.",
    version="1.0.0",
)

# Android 앱이나 웹 클라이언트가 다른 출처에서 API를 호출할 수 있도록 CORS를 허용합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 얼굴 분석 API 라우터를 메인 앱에 연결합니다.
app.include_router(analysis_router)


@app.get("/")
async def root():
    # 서버가 정상 실행 중인지 간단히 확인할 수 있는 루트 응답입니다.
    return {"service": "mura-face-analysis", "status": "running"}


@app.get("/health")
async def health():
    # 배포 환경이나 모니터링 도구에서 사용할 헬스 체크 엔드포인트입니다.
    return {"status": "ok"}

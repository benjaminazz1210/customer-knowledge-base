import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import upload, files, chat
from .logging_config import setup_logging
import uvicorn

setup_logging()
logger = logging.getLogger("nexusai.main")

app = FastAPI(title="NexusAI Backend")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    logger.info("🚀 NexusAI backend starting up on port 8001")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("🛑 NexusAI backend shutting down")

# Register routers
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(files.router, prefix="/api", tags=["Files"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])

@app.get("/")
async def root():
    return {"message": "NexusAI API is running"}

@app.get("/api/health")
async def health():
    import requests as http_requests
    qdrant_ok = False
    try:
        from .config import config
        resp = http_requests.get(f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}/", timeout=3)
        qdrant_ok = resp.status_code == 200
    except Exception:
        pass
    logger.info(f"Health check: qdrant_ok={qdrant_ok}")
    return {
        "status": "ok",
        "qdrant": qdrant_ok,
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)

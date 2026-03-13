import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .logging_config import setup_logging
from .routers import chat, files, upload, workflows
from .routers.admin import router as admin_router

setup_logging()
logger = logging.getLogger("nexusai.main")

app = FastAPI(title="NexusAI Backend")
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


app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(files.router, prefix="/api", tags=["Files"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(workflows.router, prefix="/api", tags=["Workflows"])
app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])


@app.get("/")
async def root():
    return {"message": "NexusAI API is running"}


@app.get("/api/health")
async def health():
    import requests as http_requests

    from .config import config

    qdrant_ok = False
    neo4j_enabled = bool(config.graph_rag_enabled)
    try:
        resp = http_requests.get("http://%s:%s/" % (config.QDRANT_HOST, config.QDRANT_PORT), timeout=3)
        qdrant_ok = resp.status_code == 200
    except Exception:
        pass
    logger.info("Health check: qdrant_ok=%s neo4j_enabled=%s", qdrant_ok, neo4j_enabled)
    return {
        "status": "ok",
        "qdrant": qdrant_ok,
        "neo4j_enabled": neo4j_enabled,
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)

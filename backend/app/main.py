import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis

from .config import config
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
    if not (config.admin_api_key or "").strip():
        logger.warning("⚠️ Admin endpoints are fail-closed because ADMIN_API_KEY is not configured.")


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("🛑 NexusAI backend shutting down")


app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(files.router, prefix="/api", tags=["Files"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(workflows.router, prefix="/api", tags=["Workflows"])
app.include_router(admin_router, prefix="/api", tags=["Admin"])


@app.get("/")
async def root():
    return {"message": "NexusAI API is running"}


@app.get("/api/health")
async def health():
    import requests as http_requests

    qdrant_ok = False
    redis_ok = False
    neo4j_ok = False
    neo4j_enabled = bool(config.graph_rag_enabled)
    try:
        resp = http_requests.get("http://%s:%s/" % (config.qdrant_host, config.qdrant_port), timeout=3)
        qdrant_ok = resp.status_code == 200
    except Exception:
        pass
    try:
        redis_client = Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            decode_responses=True,
        )
        redis_ok = bool(redis_client.ping())
    except Exception:
        redis_ok = False
    if neo4j_enabled and config.neo4j_password:
        try:
            from neo4j import GraphDatabase  # type: ignore

            driver = GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password))
            driver.verify_connectivity()
            neo4j_ok = True
            driver.close()
        except Exception:
            neo4j_ok = False
    logger.info(
        "Health check: qdrant_ok=%s redis_ok=%s neo4j_ok=%s neo4j_enabled=%s observability_enabled=%s",
        qdrant_ok,
        redis_ok,
        neo4j_ok,
        neo4j_enabled,
        config.observability_enabled,
    )
    return {
        "status": "ok",
        "qdrant": qdrant_ok,
        "redis": redis_ok,
        "neo4j": neo4j_ok if neo4j_enabled else None,
        "neo4j_enabled": neo4j_enabled,
        "observability_enabled": bool(config.observability_enabled),
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=config.port, reload=config.debug)

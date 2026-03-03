import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..services.workflow_service import WorkflowService

logger = logging.getLogger("nexusai.workflows")
router = APIRouter()
workflow_service = WorkflowService()


class GenerateWorkflowRequest(BaseModel):
    prompt: str = Field(..., min_length=2, description="生成需求")
    session_id: str = "default"
    file_type: Optional[str] = Field(default=None, description="docx/pptx，可不填自动识别")
    target_words: Optional[int] = Field(default=None, ge=100, le=50000)
    target_slides: Optional[int] = Field(default=None, ge=1, le=60)
    template_name: Optional[str] = None
    use_rag: bool = True


class ReviseWorkflowRequest(BaseModel):
    job_id: str = Field(..., min_length=4)
    feedback: str = Field(..., min_length=2)
    session_id: str = "default"


@router.post("/workflows/generate")
async def generate_workflow_file(request: GenerateWorkflowRequest):
    logger.info("🧩 Workflow generate request received")
    try:
        result = workflow_service.generate(request.model_dump())
        return result
    except Exception as exc:
        logger.error("❌ Workflow generate failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/workflows/revise")
async def revise_workflow_file(request: ReviseWorkflowRequest):
    logger.info("🔁 Workflow revise request received: %s", request.job_id)
    try:
        result = workflow_service.revise(request.model_dump())
        return result
    except Exception as exc:
        logger.error("❌ Workflow revise failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/workflows/jobs")
async def list_workflow_jobs(limit: int = 30):
    limit = max(1, min(limit, 200))
    try:
        return workflow_service.list_jobs(limit=limit)
    except Exception as exc:
        logger.error("❌ Workflow list jobs failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/workflows/jobs/{job_id}")
async def get_workflow_job(job_id: str):
    try:
        data = workflow_service.get_job(job_id)
        if not data:
            raise HTTPException(status_code=404, detail="job not found")
        return data
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("❌ Workflow get job failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/workflows/files/{filename}")
async def download_workflow_file(filename: str):
    try:
        file_path = workflow_service.resolve_file_path(filename)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="file not found")

        media_type = "application/octet-stream"
        name_lower = file_path.name.lower()
        if name_lower.endswith(".docx"):
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif name_lower.endswith(".pptx"):
            media_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

        return FileResponse(path=str(file_path), filename=file_path.name, media_type=media_type)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("❌ Workflow download failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

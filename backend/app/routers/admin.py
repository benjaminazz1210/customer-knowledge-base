import json
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from ..config import config
from ..evaluation import Evaluator
from ..services.ab_test import ABTestManager
from ..services.document_version_service import DocumentVersionService
from ..services.embedding_service import EmbeddingService
from ..services.feedback_service import FeedbackService
from ..services.rag_service import RAGService
from ..services.vector_store import VectorStore

router = APIRouter()
rag_service = RAGService()
evaluator = Evaluator(rag_service=rag_service)
ab_test_manager = ABTestManager()
feedback_service = FeedbackService()
version_service = DocumentVersionService()
embedding_service = EmbeddingService()
vector_store = VectorStore()


def require_admin(x_admin_api_key: Optional[str] = Header(default=None)):
    if config.admin_api_key and x_admin_api_key != config.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin API key")
    return True


@router.post("/admin/evaluate", dependencies=[Depends(require_admin)])
async def run_evaluation():
    results = evaluator.run()
    return {
        "status": "completed",
        "aggregate": results.get("aggregate", {}),
        "output_path": results.get("output_path"),
        "passed": evaluator.passes_thresholds(results),
    }


@router.get("/admin/experiments/{experiment_id}/results", dependencies=[Depends(require_admin)])
async def get_experiment_results(experiment_id: str):
    if hasattr(ab_test_manager, "results"):
        return ab_test_manager.results(experiment_id)
    return {"experiment_id": experiment_id, "results": ab_test_manager.get_results(experiment_id)}


@router.get("/admin/feedback/export", dependencies=[Depends(require_admin)])
async def export_feedback():
    rows = feedback_service.export_feedback()
    return {
        "count": len(rows),
        "items": rows,
        "jsonl": "\n".join(json.dumps(item, ensure_ascii=False) for item in rows),
    }


@router.get("/admin/documents/versions/{filename}", dependencies=[Depends(require_admin)])
async def get_document_versions(filename: str):
    return {"filename": filename, "versions": version_service.get_versions(filename)}


@router.post("/admin/documents/rollback/{filename}", dependencies=[Depends(require_admin)])
async def rollback_document(filename: str, version_id: str = Query(...)):
    version = version_service.get_version(filename, version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    chunks = version.get("chunks", [])
    embedding_texts = [str(chunk.get("chunk_text", "")) for chunk in chunks if isinstance(chunk, dict)]
    embeddings = embedding_service.get_embeddings(embedding_texts)
    vector_store.replace_file_chunks(filename, chunks, embeddings)
    return {
        "status": "rolled_back",
        "filename": filename,
        "version_id": version_id,
        "chunks_count": len(chunks),
    }

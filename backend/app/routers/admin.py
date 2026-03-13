import json
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from ..config import config
from ..evaluation import Evaluator
from ..services.ab_test import ABTestManager
from ..services.document_version_service import DocumentVersionService
from ..services.embedding_service import EmbeddingService
from ..services.feedback_service import FeedbackService
from ..services.graph_store import GraphStore
from ..services.rag_service import RAGService
from ..services.vector_store import VectorStore

router = APIRouter(prefix="/admin")
rag_service = RAGService()
evaluator = Evaluator(rag_service=rag_service)
ab_test_manager = ABTestManager()
feedback_service = FeedbackService()
version_service = DocumentVersionService()
embedding_service = EmbeddingService()
vector_store = VectorStore()
graph_store = GraphStore()


def require_admin(x_admin_api_key: Optional[str] = Header(default=None)):
    if config.admin_api_key and x_admin_api_key != config.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin API key")
    return True


@router.post("/evaluate", dependencies=[Depends(require_admin)])
async def run_evaluation(
    session_id: str = Query("evaluation"),
    experiment_id: Optional[str] = Query(default=None),
    variant_id: Optional[str] = Query(default=None),
):
    overrides = None
    if experiment_id and variant_id:
        assigned_variant = ab_test_manager.assign_variant(experiment_id, session_id)
        if assigned_variant and str(assigned_variant.get("id")) == variant_id:
            overrides = assigned_variant.get("overrides", {}) or {}
    results = evaluator.run(
        session_id=session_id,
        overrides=overrides,
        experiment_id=experiment_id,
        variant_id=variant_id,
    )
    if experiment_id and variant_id:
        ab_test_manager.record_result(
            experiment_id,
            variant_id,
            {
                "session_id": session_id,
                "evaluation": results.get("aggregate", {}),
                "output_path": results.get("output_path"),
            },
        )
    return {
        "status": "completed",
        "aggregate": results.get("aggregate", {}),
        "output_path": results.get("output_path"),
        "passed": evaluator.passes_thresholds(results),
    }


@router.get("/experiments/{experiment_id}/results", dependencies=[Depends(require_admin)])
async def get_experiment_results(experiment_id: str):
    return {"experiment_id": experiment_id, "results": ab_test_manager.get_results(experiment_id)}


@router.get("/feedback/export", dependencies=[Depends(require_admin)])
async def export_feedback():
    rows = feedback_service.export_feedback()
    return {
        "count": len(rows),
        "items": rows,
        "jsonl": "\n".join(json.dumps(item, ensure_ascii=False) for item in rows),
    }


@router.get("/documents/versions/{filename}", dependencies=[Depends(require_admin)])
async def get_document_versions(filename: str):
    return {"filename": filename, "versions": version_service.get_versions(filename)}


@router.post("/documents/rollback/{filename}", dependencies=[Depends(require_admin)])
async def rollback_document(filename: str, version_id: str = Query(...)):
    restore_plan = version_service.rollback(filename, version_id)
    if not restore_plan:
        raise HTTPException(status_code=404, detail="Version not found")

    chunks = restore_plan.get("chunks", [])
    embeddings = restore_plan.get("embeddings", [])
    if len(embeddings) != len(chunks):
        embedding_texts = [str(chunk.get("chunk_text", "")) for chunk in chunks if isinstance(chunk, dict)]
        embeddings = embedding_service.get_embeddings(embedding_texts)
    vector_store.replace_file_chunks(filename, chunks, embeddings)
    if hasattr(graph_store, "replace_document"):
        graph_store.replace_document(filename, chunks)
    elif config.graph_rag_enabled:
        graph_store.ingest_document(filename, chunks)
    return {
        "status": "rolled_back",
        "filename": filename,
        "version_id": version_id,
        "chunks_count": len(chunks),
        "restored_points_count": len(embeddings),
        "restored_chunk_hashes": [
            str(chunk.get("metadata", {}).get("chunk_hash"))
            for chunk in chunks
            if isinstance(chunk, dict) and chunk.get("metadata", {}).get("chunk_hash")
        ],
    }

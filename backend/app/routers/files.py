import logging
from fastapi import APIRouter, HTTPException
from ..services.vector_store import VectorStore

logger = logging.getLogger("nexusai.files")
router = APIRouter()
vector_store = VectorStore()

@router.get("/files")
async def list_files():
    try:
        files = vector_store.get_all_files()
        logger.info(f"📁 List files: {len(files)} file(s) in knowledge base")
        return [{"filename": f, "status": "Ready"} for f in files]
    except Exception as e:
        logger.error(f"❌ List files error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/files/{filename}")
async def delete_file(filename: str):
    try:
        logger.info(f"🗑️  Delete request: {filename}")
        vector_store.delete_by_file(filename)
        logger.info(f"✅ Deleted: {filename}")
        return {"filename": filename, "status": "Deleted"}
    except Exception as e:
        logger.error(f"❌ Delete error for {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

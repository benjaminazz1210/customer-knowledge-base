from fastapi import APIRouter, HTTPException
from ..services.vector_store import VectorStore

router = APIRouter()
vector_store = VectorStore()

@router.get("/files")
async def list_files():
    try:
        files = vector_store.get_all_files()
        # For MVP, we just return the filenames. In a full app, we'd include metadata like upload date.
        return [{"filename": f, "status": "Ready"} for f in files]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/files/{filename}")
async def delete_file(filename: str):
    try:
        vector_store.delete_by_file(filename)
        return {"filename": filename, "status": "Deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

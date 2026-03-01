import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from ..services.document_parser import DocumentParser
from ..services.text_chunker import TextChunker
from ..services.embedding_service import EmbeddingService
from ..services.vector_store import VectorStore
import time

logger = logging.getLogger("nexusai.upload")

router = APIRouter()
parser = DocumentParser()
chunker = TextChunker()
embedding_service = EmbeddingService()
vector_store = VectorStore()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    logger.info(f"📄 Upload started: {file.filename}")
    try:
        content = await file.read()
        logger.info(f"   File size: {len(content)/1024:.1f} KB")
        text = parser.parse(content, file.filename)
        chunks = chunker.chunk(text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="File is empty or could not be parsed.")
        
        logger.info(f"   Parsed into {len(chunks)} chunks, generating embeddings...")
        embeddings = embedding_service.get_embeddings(chunks)
        vector_store.upsert_chunks(file.filename, chunks, embeddings)
        logger.info(f"✅ Upload complete: {file.filename} ({len(chunks)} chunks stored)")
        
        return {
            "filename": file.filename,
            "status": "Ready",
            "chunks_count": len(chunks),
            "timestamp": time.time()
        }
    except HTTPException:
        # Preserve explicit client-facing status codes (e.g. empty file).
        raise
    except ValueError as e:
        # Parser-level validation errors should be client errors.
        logger.warning(f"⚠️  Upload rejected for {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Upload failed for {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

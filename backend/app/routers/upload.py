import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from ..services.document_parser import DocumentParser
from ..services.text_chunker import TextChunker
from ..services.embedding_service import EmbeddingService
from ..services.vector_store import VectorStore
from ..services.vision_service import VisionService
import time

logger = logging.getLogger("nexusai.upload")

router = APIRouter()
parser = DocumentParser()
chunker = TextChunker()
embedding_service = EmbeddingService()
vector_store = VectorStore()
vision_service = VisionService()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    logger.info(f"📄 Upload started: {file.filename}")
    try:
        content = await file.read()
        logger.info(f"   File size: {len(content)/1024:.1f} KB")

        parsed_doc = parser.parse_structured(content, file.filename)
        text_chunks = chunker.chunk_structured(parsed_doc.sections)
        image_chunks = vision_service.describe_images(parsed_doc.images, source_file=file.filename)
        chunks = text_chunks + image_chunks

        if not chunks and parsed_doc.full_text:
            # Fallback for edge-case documents with no structural sections
            chunks = chunker.chunk(parsed_doc.full_text)

        if not chunks:
            raise HTTPException(status_code=400, detail="File is empty or could not be parsed.")

        embedding_texts = [
            c.get("chunk_text", "") if isinstance(c, dict) else c
            for c in chunks
        ]

        logger.info(
            "   Parsed via %s: %s sections, %s images, %s chunks, generating embeddings...",
            parsed_doc.backend_used,
            len(parsed_doc.sections),
            len(parsed_doc.images),
            len(chunks),
        )
        embeddings = embedding_service.get_embeddings(embedding_texts)
        vector_store.upsert_chunks(file.filename, chunks, embeddings)
        logger.info(f"✅ Upload complete: {file.filename} ({len(chunks)} chunks stored)")
        
        return {
            "filename": file.filename,
            "status": "Ready",
            "chunks_count": len(chunks),
            "text_chunks_count": len(text_chunks),
            "image_description_chunks_count": len(image_chunks),
            "sections_count": len(parsed_doc.sections),
            "images_count": len(parsed_doc.images),
            "parser_backend": parsed_doc.backend_used,
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

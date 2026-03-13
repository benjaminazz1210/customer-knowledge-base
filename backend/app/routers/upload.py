import logging
import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..services.document_parser import DocumentParser
from ..services.document_version_service import DocumentVersionService
from ..services.embedding_service import EmbeddingService
from ..services.graph_store import GraphStore
from ..services.text_chunker import TextChunker
from ..services.vector_store import VectorStore
from ..services.vision_service import VisionService

logger = logging.getLogger("nexusai.upload")

router = APIRouter()
parser = DocumentParser()
chunker = TextChunker()
embedding_service = EmbeddingService()
vector_store = VectorStore()
vision_service = VisionService()
version_service = DocumentVersionService()
graph_store = GraphStore()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    logger.info("📄 Upload started: %s", file.filename)
    try:
        content = await file.read()
        logger.info("   File size: %.1f KB", len(content) / 1024)
        content_hash = version_service.compute_content_hash(content)
        if version_service.is_unchanged(file.filename, content_hash):
            latest = version_service.latest(file.filename) or {}
            return {
                "filename": file.filename,
                "status": "Ready",
                "skipped": True,
                "reason": "unchanged_content_hash",
                "version_id": latest.get("version_id"),
                "timestamp": time.time(),
            }

        parsed_doc = parser.parse_structured(content, file.filename)
        text_chunks = chunker.chunk_document(parsed_doc.sections, parsed_doc.full_text)
        image_chunks = vision_service.describe_images(parsed_doc.images, source_file=file.filename)
        chunks = text_chunks + image_chunks

        if not chunks and parsed_doc.full_text:
            chunks = chunker.chunk_document([], parsed_doc.full_text)

        if not chunks:
            raise HTTPException(status_code=400, detail="File is empty or could not be parsed.")

        version_record = version_service.record_version(
            filename=file.filename,
            content_hash=content_hash,
            chunks=chunks,
        )
        version_id = version_record["version_id"]

        prepared_chunks = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                metadata = dict(chunk.get("metadata", {}))
                metadata.update({"content_hash": content_hash, "version_id": version_id})
                prepared_chunks.append({**chunk, "metadata": metadata})
            else:
                prepared_chunks.append(
                    {
                        "chunk_text": str(chunk),
                        "metadata": {"content_hash": content_hash, "version_id": version_id},
                    }
                )

        embedding_texts = [c.get("chunk_text", "") if isinstance(c, dict) else str(c) for c in prepared_chunks]

        logger.info(
            "   Parsed via %s: %s sections, %s images, %s chunks, generating embeddings...",
            parsed_doc.backend_used,
            len(parsed_doc.sections),
            len(parsed_doc.images),
            len(prepared_chunks),
        )
        embeddings = embedding_service.get_embeddings(embedding_texts)
        vector_store.replace_file_chunks(file.filename, prepared_chunks, embeddings)
        graph_store.ingest_document(file.filename, prepared_chunks)
        logger.info("✅ Upload complete: %s (%s chunks stored)", file.filename, len(prepared_chunks))

        return {
            "filename": file.filename,
            "status": "Ready",
            "chunks_count": len(prepared_chunks),
            "text_chunks_count": len(text_chunks),
            "image_description_chunks_count": len(image_chunks),
            "sections_count": len(parsed_doc.sections),
            "images_count": len(parsed_doc.images),
            "parser_backend": parsed_doc.backend_used,
            "version_id": version_id,
            "content_hash": content_hash,
            "timestamp": time.time(),
        }
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("⚠️ Upload rejected for %s: %s", file.filename, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("❌ Upload failed for %s: %s", file.filename, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

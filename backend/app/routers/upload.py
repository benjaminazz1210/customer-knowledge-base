import logging
import time
from dataclasses import asdict

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

        version_id = version_service.generate_version_id()

        prepared_chunks = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                metadata = dict(chunk.get("metadata", {}))
                chunk_text = str(chunk.get("chunk_text", ""))
                metadata.update(
                    {
                        "content_hash": content_hash,
                        "version_id": version_id,
                        "chunk_hash": version_service.compute_content_hash(chunk_text.encode("utf-8")),
                    }
                )
                prepared_chunks.append({**chunk, "metadata": metadata})
            else:
                chunk_text = str(chunk)
                prepared_chunks.append(
                    {
                        "chunk_text": chunk_text,
                        "metadata": {
                            "content_hash": content_hash,
                            "version_id": version_id,
                            "chunk_hash": version_service.compute_content_hash(chunk_text.encode("utf-8")),
                        },
                    }
                )

        existing_chunks = vector_store.get_file_chunks(file.filename, include_vectors=True)
        existing_vectors = {}
        for row in existing_chunks:
            payload = row.get("payload", {}) or {}
            chunk_hash = payload.get("chunk_hash")
            vector = row.get("vector")
            if chunk_hash and vector:
                existing_vectors[str(chunk_hash)] = vector

        embedding_texts = []
        embedding_indexes = []
        embeddings = [None] * len(prepared_chunks)
        reused_embeddings = 0
        for idx, chunk in enumerate(prepared_chunks):
            metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
            chunk_hash = str(metadata.get("chunk_hash", ""))
            reused_vector = existing_vectors.get(chunk_hash)
            if reused_vector is not None:
                embeddings[idx] = reused_vector
                reused_embeddings += 1
                continue
            embedding_texts.append(chunk.get("chunk_text", "") if isinstance(chunk, dict) else str(chunk))
            embedding_indexes.append(idx)

        logger.info(
            "   Parsed via %s: %s sections, %s images, %s chunks, generating embeddings (%s reused)...",
            parsed_doc.backend_used,
            len(parsed_doc.sections),
            len(parsed_doc.images),
            len(prepared_chunks),
            reused_embeddings,
        )
        if embedding_texts:
            fresh_embeddings = embedding_service.get_embeddings(embedding_texts)
            for idx, vector in zip(embedding_indexes, fresh_embeddings):
                embeddings[idx] = vector
        if any(vector is None for vector in embeddings):
            raise RuntimeError("embedding generation did not return a vector for every chunk")
        embeddings = list(embeddings)
        vector_store.replace_file_chunks(file.filename, prepared_chunks, embeddings)
        graph_store.ingest_document(file.filename, prepared_chunks)
        version_record = version_service.record_version(
            filename=file.filename,
            content_hash=content_hash,
            chunks=prepared_chunks,
            raw_content=parsed_doc.full_text,
            version_id=version_id,
            metadata={
                "parser_backend": parsed_doc.backend_used,
                "sections": [asdict(section) for section in parsed_doc.sections],
            },
        )
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
            "version_id": version_record["version_id"],
            "content_hash": content_hash,
            "reused_embeddings": reused_embeddings,
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

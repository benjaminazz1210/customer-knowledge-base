from fastapi import APIRouter, UploadFile, File, HTTPException
from ..services.document_parser import DocumentParser
from ..services.text_chunker import TextChunker
from ..services.embedding_service import EmbeddingService
from ..services.vector_store import VectorStore
import time

router = APIRouter()
parser = DocumentParser()
chunker = TextChunker()
embedding_service = EmbeddingService()
vector_store = VectorStore()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = parser.parse(content, file.filename)
        chunks = chunker.chunk(text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="File is empty or could not be parsed.")
            
        embeddings = embedding_service.get_embeddings(chunks)
        vector_store.upsert_chunks(file.filename, chunks, embeddings)
        
        return {
            "filename": file.filename,
            "status": "Ready",
            "chunks_count": len(chunks),
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

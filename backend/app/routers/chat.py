import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ..services.rag_service import RAGService
import json
import asyncio

logger = logging.getLogger("nexusai.chat")
router = APIRouter()
rag_service = RAGService()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
async def chat(request: ChatRequest):
    logger.info(f"💬 Chat request: \"{request.message[:80]}{'...' if len(request.message)>80 else ''}\"")
    try:
        response_gen, sources = rag_service.generate_response(request.message)
        logger.info(f"   Sources found: {sources}")
        
        async def stream_response():
            token_count = 0
            # First send sources
            yield f"data: {json.dumps({'sources': sources})}\n\n"
            
            # Then send tokens
            for chunk in response_gen:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    token_count += 1
                    yield f"data: {json.dumps({'token': token})}\n\n"
            
            logger.info(f"✅ Chat complete: {token_count} tokens streamed")
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

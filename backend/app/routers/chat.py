from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ..services.rag_service import RAGService
import json
import asyncio

router = APIRouter()
rag_service = RAGService()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        response_gen, sources = rag_service.generate_response(request.message)
        
        async def stream_response():
            # First send sources
            yield f"data: {json.dumps({'sources': sources})}\n\n"
            
            # Then send tokens
            for chunk in response_gen:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'token': token})}\n\n"
            
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

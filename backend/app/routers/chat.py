import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from ..services.rag_service import RAGService
from ..services.history_service import HistoryService
from openai import BadRequestError
import json
import asyncio

logger = logging.getLogger("nexusai.chat")
router = APIRouter()
rag_service = RAGService()
history_service = HistoryService()

class ChatRequest(BaseModel):
    message: str

class HistoryRequest(BaseModel):
    messages: List[Dict[str, Any]]

@router.get("/history")
async def get_history():
    return history_service.get_history()

@router.post("/history")
async def save_history(request: HistoryRequest):
    history_service.save_history("default", request.messages)
    return {"status": "success"}

@router.delete("/history")
async def clear_history():
    history_service.clear_history()
    return {"status": "success"}

@router.post("/chat")
async def chat(request: ChatRequest):
    logger.info(f"💬 Chat request: \"{request.message[:80]}{'...' if len(request.message)>80 else ''}\"")
    try:
        history = history_service.get_history()
        response_gen, sources = rag_service.generate_response(request.message, history=history)
        logger.info(f"   Sources found: {len(sources) if sources else 0} items")

        async def stream_response():
            token_count = 0
            yield f"data: {json.dumps({'sources': sources})}\n\n"
            try:
                for chunk in response_gen:
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        token_count += 1
                        yield f"data: {json.dumps({'token': token})}\n\n"
                logger.info(f"✅ Chat complete: {token_count} tokens streamed")
            except BadRequestError as e:
                err_msg = str(e)
                if "Content Exists Risk" in err_msg:
                    logger.warning(f"⚠️  Content risk triggered by DeepSeek, sending fallback message")
                    yield f"data: {json.dumps({'token': '抱歉，该文档中的部分内容触发了 AI 安全审核，无法直接回答。请尝试换一种提问方式，或上传其他文档。'})}\n\n"
                else:
                    logger.error(f"❌ BadRequestError in stream: {e}")
                    yield f"data: {json.dumps({'token': f'请求错误：{err_msg}'})}\n\n"
            except Exception as e:
                logger.error(f"❌ Stream error: {e}", exc_info=True)
                yield f"data: {json.dumps({'token': '生成回答时出现错误，请稍后重试。'})}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from ..services.rag_service import RAGService
from ..services.history_service import HistoryService
from openai import BadRequestError
import json

logger = logging.getLogger("nexusai.chat")
router = APIRouter()
rag_service = RAGService()
history_service = HistoryService()


def _get_attr_or_key(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: List[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue

        text_val = item.get("text")
        if isinstance(text_val, dict):
            text_val = text_val.get("value") or text_val.get("content")
        if text_val is None:
            text_val = item.get("value") or item.get("content")
        if isinstance(text_val, str):
            parts.append(text_val)
    return "".join(parts)


def _extract_stream_token(chunk: Any) -> str:
    choices = _get_attr_or_key(chunk, "choices")
    if not choices or not isinstance(choices, list):
        return ""

    choice = choices[0]

    # OpenAI-compatible chat stream:
    # chunk.choices[0].delta.content
    delta = _get_attr_or_key(choice, "delta")
    token = _normalize_content(_get_attr_or_key(delta, "content"))
    if token:
        return token

    # Some gateways send final text under message.content or text.
    message = _get_attr_or_key(choice, "message")
    token = _normalize_content(_get_attr_or_key(message, "content"))
    if token:
        return token

    text_val = _get_attr_or_key(choice, "text")
    return text_val if isinstance(text_val, str) else ""


def _chunk_shape(chunk: Any) -> str:
    choices = _get_attr_or_key(chunk, "choices")
    if not choices or not isinstance(choices, list):
        return f"{type(chunk).__name__}(no-choices)"

    first = choices[0]
    delta = _get_attr_or_key(first, "delta")
    message = _get_attr_or_key(first, "message")
    delta_content = _get_attr_or_key(delta, "content")
    message_content = _get_attr_or_key(message, "content")
    text_val = _get_attr_or_key(first, "text")
    return (
        f"{type(chunk).__name__}("
        f"delta.content={type(delta_content).__name__}, "
        f"message.content={type(message_content).__name__}, "
        f"text={type(text_val).__name__})"
    )


async def _iter_chunks(response_gen):
    if hasattr(response_gen, "__aiter__"):
        async for chunk in response_gen:
            yield chunk
    else:
        for chunk in response_gen:
            yield chunk

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
            chunk_count = 0
            first_chunk_shape = ""
            yield f"data: {json.dumps({'sources': sources})}\n\n"
            try:
                async for chunk in _iter_chunks(response_gen):
                    chunk_count += 1
                    if not first_chunk_shape:
                        first_chunk_shape = _chunk_shape(chunk)

                    token = _extract_stream_token(chunk)
                    if token:
                        token_count += 1
                        yield f"data: {json.dumps({'token': token})}\n\n"
                if token_count == 0:
                    logger.warning(
                        "⚠️ Chat stream ended with 0 text chunks (chunks_seen=%s, first_chunk=%s)",
                        chunk_count,
                        first_chunk_shape or "N/A",
                    )
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

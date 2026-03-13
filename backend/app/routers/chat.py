import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openai import BadRequestError
from pydantic import BaseModel

from ..config import config
from ..services.ab_test import ABTestManager
from ..services.feedback_service import FeedbackService
from ..services.guardrails_service import GuardrailsService
from ..services.history_service import HistoryService
from ..services.rag_service import RAGResponse, RAGService
from ..services.self_rag import SelfRAGController

logger = logging.getLogger("nexusai.chat")
router = APIRouter()
rag_service = RAGService()
self_rag = SelfRAGController(rag_service)
history_service = HistoryService()
guardrails_service = GuardrailsService()
feedback_service = FeedbackService()
ab_test_manager = ABTestManager()


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
    delta = _get_attr_or_key(choice, "delta")
    token = _normalize_content(_get_attr_or_key(delta, "content"))
    if token:
        return token
    message = _get_attr_or_key(choice, "message")
    token = _normalize_content(_get_attr_or_key(message, "content"))
    if token:
        return token
    text_val = _get_attr_or_key(choice, "text")
    return text_val if isinstance(text_val, str) else ""


async def _iter_chunks(response_gen):
    if hasattr(response_gen, "__aiter__"):
        async for chunk in response_gen:
            yield chunk
    else:
        for chunk in response_gen:
            yield chunk


def _normalize_rag_result(result: Any) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
    if isinstance(result, RAGResponse):
        metadata = {
            "trace_id": result.trace_id,
            "confidence_score": result.confidence_score,
            "experiment_id": result.experiment_id,
            "variant_id": result.variant_id,
        }
        metadata.update(result.metadata or {})
        return result.response_gen, result.sources, metadata
    if isinstance(result, tuple) and len(result) == 3:
        return result
    if isinstance(result, tuple) and len(result) == 2:
        response_gen, sources = result
        return response_gen, sources, {}
    raise TypeError("Unsupported RAG response shape")


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class HistoryRequest(BaseModel):
    messages: List[Dict[str, Any]]
    session_id: str = "default"


class FeedbackRequest(BaseModel):
    trace_id: str
    message_id: Optional[str] = None
    rating: str
    comment: str = ""


@router.get("/history")
async def get_history(session_id: str = "default"):
    return history_service.get_history(session_id=session_id)


@router.post("/history")
async def save_history(request: HistoryRequest):
    history_service.save_history(request.session_id, request.messages)
    return {"status": "success"}


@router.delete("/history")
async def clear_history(session_id: str = "default"):
    history_service.clear_history(session_id=session_id)
    return {"status": "success"}


@router.post("/chat/feedback")
async def chat_feedback(request: FeedbackRequest):
    payload = feedback_service.submit_feedback(
        trace_id=request.trace_id,
        message_id=request.message_id or request.trace_id,
        rating=request.rating,
        comment=request.comment,
    )
    return {"status": "success", "feedback": payload}


@router.post("/chat")
async def chat(request: ChatRequest):
    logger.info('💬 Chat request: "%s"', request.message[:80] + ("..." if len(request.message) > 80 else ""))
    feedback_service.capture_implicit_feedback(request.session_id, request.message)
    guardrail = guardrails_service.check_input(request.message)
    if not guardrail.allowed:
        trace_id = str(uuid.uuid4())

        async def blocked_response():
            yield "data: %s\n\n" % json.dumps(
                {
                    "sources": [],
                    "trace_id": trace_id,
                    "confidence_score": 0.0,
                    "message_id": trace_id,
                }
            )
            yield "data: %s\n\n" % json.dumps({"token": guardrail.message or config.guardrails_block_message})
            yield "data: [DONE]\n\n"

        return StreamingResponse(blocked_response(), media_type="text/event-stream")

    try:
        history = history_service.get_history(session_id=request.session_id)
        assignment = ab_test_manager.assign_active_variant(request.session_id) or {}
        response_gen, sources, metadata = _normalize_rag_result(
            self_rag.generate_response(
                request.message,
                history=history,
                session_id=request.session_id,
                overrides=assignment.get("overrides"),
                experiment_id=assignment.get("experiment_id"),
                variant_id=assignment.get("variant_id"),
            )
        )
        trace_id = metadata.get("trace_id") or str(uuid.uuid4())
        confidence_score = float(metadata.get("confidence_score", 0.0))
        message_id = trace_id

        async def stream_response():
            rendered_tokens: List[str] = []
            yield "data: %s\n\n" % json.dumps(
                {
                    "sources": sources,
                    "trace_id": trace_id,
                    "confidence_score": confidence_score,
                    "message_id": message_id,
                    "experiment_id": metadata.get("experiment_id"),
                    "variant_id": metadata.get("variant_id"),
                }
            )
            try:
                if config.guardrails_enabled:
                    buffered = ""
                    async for chunk in _iter_chunks(response_gen):
                        buffered += _extract_stream_token(chunk)
                    checked = guardrails_service.check_output(buffered)
                    final_text = checked.sanitized_text or checked.message or config.guardrails_block_message
                    for token in final_text.split(" "):
                        if token:
                            rendered_tokens.append(token + " ")
                            yield "data: %s\n\n" % json.dumps({"token": token + " "})
                else:
                    async for chunk in _iter_chunks(response_gen):
                        token = _extract_stream_token(chunk)
                        if not token:
                            continue
                        rendered_tokens.append(token)
                        yield "data: %s\n\n" % json.dumps({"token": token})
            except BadRequestError as exc:
                err_msg = str(exc)
                if "Content Exists Risk" in err_msg:
                    fallback = "抱歉，该文档中的部分内容触发了 AI 安全审核，无法直接回答。请尝试换一种提问方式，或上传其他文档。"
                    rendered_tokens.append(fallback)
                    yield "data: %s\n\n" % json.dumps({"token": fallback})
                else:
                    rendered_tokens.append("请求错误：%s" % err_msg)
                    yield "data: %s\n\n" % json.dumps({"token": "请求错误：%s" % err_msg})
            except Exception as exc:
                logger.error("❌ Stream error: %s", exc, exc_info=True)
                rendered_tokens.append("生成回答时出现错误，请稍后重试。")
                yield "data: %s\n\n" % json.dumps({"token": "生成回答时出现错误，请稍后重试。"})
            finally:
                feedback_service.register_turn(
                    trace_id=trace_id,
                    session_id=request.session_id,
                    query=request.message,
                    answer="".join(rendered_tokens).strip(),
                    sources=sources,
                    confidence_score=confidence_score,
                )
                if metadata.get("experiment_id") and metadata.get("variant_id"):
                    ab_test_manager.record_result(
                        str(metadata.get("experiment_id")),
                        str(metadata.get("variant_id")),
                        {
                            "trace_id": trace_id,
                            "session_id": request.session_id,
                            "confidence_score": confidence_score,
                            "sources_count": len(sources),
                        },
                    )
                yield "data: [DONE]\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")
    except Exception as exc:
        logger.error("❌ Chat error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

from ..config import config
from ..observability.tracer import PipelineTracer
from .confidence_service import LowConfidenceService
from .embedding_service import EmbeddingService
from .graph_store import GraphStore
from .llm_utils import create_llm_client, is_mock_backend
from .query_transformer import QueryTransformer
from .reranker_service import RerankerService
from .self_rag import SelfRAGController
from .vector_store import VectorStore

logger = logging.getLogger("nexusai.rag")


class _MockDelta:
    def __init__(self, content: str):
        self.content = content


class _MockChoice:
    def __init__(self, content: str):
        self.delta = _MockDelta(content)


class _MockChunk:
    def __init__(self, content: str):
        self.choices = [_MockChoice(content)]


@dataclass
class RAGResponse:
    response_gen: Iterable[Any]
    sources: List[Dict[str, Any]]
    trace_id: str
    confidence_score: float
    experiment_id: Optional[str] = None
    variant_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.reranker = RerankerService()
        self.query_transformer = QueryTransformer()
        self.confidence_service = LowConfidenceService()
        self.graph_store = GraphStore()
        self.self_rag = SelfRAGController(self)
        self.mock_llm = is_mock_backend()
        self.provider = config.llm_provider.strip().lower()
        self.llm_client: Optional[OpenAI] = None if self.mock_llm else create_llm_client()

        if self.mock_llm:
            logger.info("🧪 LLM backend: MOCK mode")
        else:
            logger.info("🤖 LLM provider: %s — model: %s", self.provider, config.llm_model)

    @staticmethod
    def _progress_bar() -> str:
        return "[████████████████████] 100%"

    @staticmethod
    def _mock_stream(answer: str):
        for token in answer.split(" "):
            if token:
                yield _MockChunk(token + " ")

    @staticmethod
    def _extract_token_from_chunk(chunk: Any) -> str:
        choices = getattr(chunk, "choices", None)
        if not choices and isinstance(chunk, dict):
            choices = chunk.get("choices")
        if not choices:
            return ""
        choice = choices[0]
        delta = getattr(choice, "delta", None) if not isinstance(choice, dict) else choice.get("delta")
        if delta is not None:
            content = getattr(delta, "content", None) if not isinstance(delta, dict) else delta.get("content")
            if isinstance(content, str):
                return content
        message = getattr(choice, "message", None) if not isinstance(choice, dict) else choice.get("message")
        if message is not None:
            content = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")
            if isinstance(content, str):
                return content
        text = getattr(choice, "text", None) if not isinstance(choice, dict) else choice.get("text")
        return text if isinstance(text, str) else ""

    @staticmethod
    def _coerce_history(history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return history or []

    @staticmethod
    def _simple_direct_answer(query: str) -> Optional[str]:
        stripped = (query or "").strip()
        if re.fullmatch(r"\d+\s*[\+\-\*\/]\s*\d+", stripped):
            try:
                return str(eval(stripped, {"__builtins__": {}}, {}))
            except Exception:
                return None
        return None

    def _transform_query(self, query: str, overrides: Optional[Dict[str, Any]] = None):
        transformer = QueryTransformer()
        if overrides and overrides.get("query_transform_strategy"):
            transformer.strategy = str(overrides["query_transform_strategy"]).strip().lower()
        return transformer.transform(query)

    def _retrieve_candidates(
        self,
        query: str,
        transformed_query,
        tracer: PipelineTracer,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        reranker_enabled = bool(overrides.get("reranker_enabled", config.reranker_enabled)) if overrides else config.reranker_enabled
        search_limit = int(overrides.get("reranker_top_n", config.reranker_top_n)) if overrides and reranker_enabled else (
            config.reranker_top_n if config.reranker_enabled else 5
        )
        alpha = float(overrides.get("workflow_hybrid_alpha", config.workflow_hybrid_alpha)) if overrides else config.workflow_hybrid_alpha
        expand_to_parent = (overrides.get("chunking_strategy") if overrides else config.chunking_strategy) == "parent_child"

        rankings: List[List[Dict[str, Any]]] = []
        search_queries = transformed_query.search_queries or [query]
        for search_query in search_queries:
            embedding_query = transformed_query.embedding_query if transformed_query.strategy == "hyde" else search_query
            with tracer.step("embedding", input_size=len(embedding_query)):
                query_vector = self.embedding_service.get_embeddings([embedding_query])[0]

            with tracer.step("retrieve", query=search_query, top_n=search_limit):
                hits = self.vector_store.hybrid_search(
                    query_text=search_query,
                    query_vector=query_vector,
                    limit=search_limit,
                    alpha=alpha,
                    expand_to_parent=expand_to_parent,
                )
            rankings.append(hits)

        if transformed_query.strategy in ("decompose", "multi_query") and len(rankings) > 1:
            merged = QueryTransformer.reciprocal_rank_fusion(rankings)
        else:
            merged = rankings[0] if rankings else []

        graph_context = self.graph_store.query_context(query)
        for context_line in graph_context:
            merged.append(
                {
                    "payload": {
                        "source_file": "graph://context",
                        "chunk_text": context_line,
                        "chunk_index": 10_000 + len(merged),
                    },
                    "score": 0.45,
                    "vector_score": 0.0,
                    "keyword_score": 0.0,
                }
            )

        merged.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return merged[:search_limit]

    def _rerank_candidates(self, query: str, hits: List[Dict[str, Any]], tracer: PipelineTracer) -> List[Dict[str, Any]]:
        if not config.reranker_enabled:
            return hits[:5]
        with tracer.step("rerank", candidate_count=len(hits)):
            reranked = self.reranker.rerank_hits(query, hits[: config.reranker_top_n])
        return reranked[: config.reranker_top_k]

    @staticmethod
    def _source_details_from_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        source_details = []
        for hit in hits:
            payload = hit.get("payload", {}) or {}
            source_details.append(
                {
                    "source_file": payload.get("source_file"),
                    "content": payload.get("chunk_text", ""),
                    "score": float(hit.get("score", 0.0)),
                    "parent_id": payload.get("parent_id"),
                    "children_ids": payload.get("children_ids", []),
                }
            )
        return source_details

    @staticmethod
    def _build_context(hits: List[Dict[str, Any]]) -> str:
        return "\n\n---\n\n".join(str(hit.get("payload", {}).get("chunk_text", "")) for hit in hits if hit.get("payload"))

    def _build_messages(self, query: str, history: List[Dict[str, Any]], context_text: str) -> List[Dict[str, str]]:
        system_prompt = (
            "你是一个知识库智能助手。请根据下方检索到的文档内容回答用户的问题。"
            "如果文档中没有相关信息，请如实说明，不要编造答案。回答请简洁准确。\n\n"
            f"参考文档：\n{context_text}"
        )
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for msg in history:
            role = "assistant" if msg.get("isAi") else "user"
            messages.append({"role": role, "content": msg.get("text", "")})
        messages.append({"role": "user", "content": query})
        return messages

    def generate_response(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None,
        *,
        session_id: str = "default",
        overrides: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None,
        variant_id: Optional[str] = None,
    ) -> RAGResponse:
        history = self._coerce_history(history)
        tracer = PipelineTracer(metadata={"query": query, "session_id": session_id})

        if config.self_rag_enabled and self.self_rag.should_skip_retrieval(query):
            direct_answer = self._simple_direct_answer(query) or f"Direct response: {query}"
            return RAGResponse(
                response_gen=self._mock_stream(direct_answer) if self.mock_llm else self._mock_stream(direct_answer),
                sources=[],
                trace_id=tracer.trace_id,
                confidence_score=1.0,
                experiment_id=experiment_id,
                variant_id=variant_id,
                metadata={"self_rag_skipped_retrieval": True, "trace": tracer.export()},
            )

        with tracer.step("query_transform", input_size=len(query)):
            transformed_query = self._transform_query(query, overrides=overrides)

        hits = []
        for attempt in range(max(config.self_rag_max_retries, 0) + 1):
            hits = self._retrieve_candidates(query, transformed_query, tracer, overrides=overrides)
            confidence_score = self.confidence_service.score_hits(hits)
            if not config.self_rag_enabled or self.self_rag.critique_hits(query, hits, confidence_score):
                break
            if not self.self_rag.should_retry(hits, confidence_score, attempt):
                break
            transformed_query = QueryTransformer()._rewrite(query)

        hits = self._rerank_candidates(query, hits, tracer)
        confidence_assessment = self.confidence_service.evaluate(query, hits)
        low_confidence = bool(confidence_assessment.get("is_low_confidence", False))
        confidence_score = float(confidence_assessment.get("confidence_score", 0.0))

        with tracer.step("context", retrieved=len(hits), confidence_score=confidence_score):
            context_text = self._build_context(hits)
            source_details = self._source_details_from_hits(hits)

        if low_confidence:
            logger.info("⚠️ Low confidence triggered for query=%r score=%.4f", query, confidence_score)
            return RAGResponse(
                response_gen=self._mock_stream(config.low_confidence_message),
                sources=source_details,
                trace_id=tracer.trace_id,
                confidence_score=confidence_score,
                experiment_id=experiment_id,
                variant_id=variant_id,
                metadata={"trace": tracer.export(), "low_confidence": True},
            )

        if self.mock_llm:
            if source_details:
                source_files = ", ".join(sorted(set(str(s["source_file"]) for s in source_details if s.get("source_file"))))
                answer = f"Mock answer based on sources {source_files}. Query: {query}"
            else:
                answer = f"Mock answer: no relevant context found. Query: {query}"
            return RAGResponse(
                response_gen=self._mock_stream(answer),
                sources=source_details,
                trace_id=tracer.trace_id,
                confidence_score=confidence_score,
                experiment_id=experiment_id,
                variant_id=variant_id,
                metadata={"trace": tracer.export()},
            )

        with tracer.step("generate", prompt_size=len(context_text), history_size=len(history)):
            messages = self._build_messages(query, history, context_text)
            response = self.llm_client.chat.completions.create(
                model=config.llm_model,
                messages=messages,
                stream=True,
            )

        return RAGResponse(
            response_gen=response,
            sources=source_details,
            trace_id=tracer.trace_id,
            confidence_score=confidence_score,
            experiment_id=experiment_id,
            variant_id=variant_id,
            metadata={"trace": tracer.export()},
        )

    def generate_answer_text(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None,
        *,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        result = self.generate_response(query, history=history, session_id=session_id)
        answer_parts: List[str] = []
        for chunk in result.response_gen:
            token = self._extract_token_from_chunk(chunk)
            if token:
                answer_parts.append(token)
        return {
            "answer": "".join(answer_parts).strip(),
            "sources": result.sources,
            "trace_id": result.trace_id,
            "confidence_score": result.confidence_score,
        }

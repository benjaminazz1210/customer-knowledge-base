import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

from ..config import config
from ..observability.tracer import create_tracer
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

    @staticmethod
    def _estimate_tokens(text: Any) -> int:
        if not text:
            return 0
        if isinstance(text, list):
            return sum(RAGService._estimate_tokens(item) for item in text)
        return len(str(text).split())

    @staticmethod
    def _setting(overrides: Optional[Dict[str, Any]], key: str, default: Any) -> Any:
        if overrides and key in overrides:
            return overrides[key]
        return default

    @staticmethod
    def _completion_text(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(getattr(item, "text", "")))
            return "".join(parts).strip()
        return str(content or "").strip()

    def _transform_query(self, query: str, overrides: Optional[Dict[str, Any]] = None):
        transform_enabled = bool(self._setting(overrides, "query_transform_enabled", config.query_transform_enabled))
        transform_strategy = str(
            self._setting(overrides, "query_transform_strategy", config.query_transform_strategy)
        ).strip().lower()
        transform_model = self._setting(overrides, "query_transform_model", config.query_transform_model)
        multi_query_count = int(self._setting(overrides, "multi_query_count", config.multi_query_count))
        if overrides and "query_transform_enabled" not in overrides and transform_strategy not in ("", "none"):
            transform_enabled = True
        transformer = QueryTransformer(
            enabled=transform_enabled,
            strategy=transform_strategy,
            model=transform_model,
            multi_query_count=multi_query_count,
        )
        return transformer.transform(query)

    def _retrieve_candidates(
        self,
        query: str,
        transformed_query,
        tracer: Any,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        reranker_enabled = bool(self._setting(overrides, "reranker_enabled", config.reranker_enabled))
        search_limit = (
            int(self._setting(overrides, "reranker_top_n", config.reranker_top_n))
            if reranker_enabled
            else (config.reranker_top_n if config.reranker_enabled else max(int(config.reranker_top_k), 5))
        )
        alpha = float(self._setting(overrides, "workflow_hybrid_alpha", config.workflow_hybrid_alpha))
        expand_to_parent = self._setting(overrides, "chunking_strategy", config.chunking_strategy) == "parent_child"

        rankings: List[List[Dict[str, Any]]] = []
        search_queries = transformed_query.search_queries or [query]
        for search_query in search_queries:
            embedding_query = transformed_query.embedding_query if transformed_query.strategy == "hyde" else search_query
            with tracer.step(
                "embedding",
                input_size=len(embedding_query),
                input_tokens=self._estimate_tokens(embedding_query),
            ):
                query_vector = self.embedding_service.get_embeddings([embedding_query])[0]

            with tracer.step(
                "retrieve",
                query=search_query,
                top_n=search_limit,
                input_tokens=self._estimate_tokens(search_query),
            ):
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
        for graph_hit in graph_context:
            if isinstance(graph_hit, dict):
                merged.append(graph_hit)

        merged.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return merged[:search_limit]

    def _rerank_candidates(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        tracer: Any,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        reranker_enabled = bool(self._setting(overrides, "reranker_enabled", config.reranker_enabled))
        top_n = int(self._setting(overrides, "reranker_top_n", config.reranker_top_n))
        top_k = int(self._setting(overrides, "reranker_top_k", config.reranker_top_k))
        backend = str(self._setting(overrides, "reranker_backend", config.reranker_backend)).strip().lower()
        model_name = str(self._setting(overrides, "reranker_model", config.reranker_model))
        api_url = self._setting(overrides, "reranker_api_url", config.reranker_api_url)
        api_key = self._setting(overrides, "reranker_api_key", config.reranker_api_key)
        if not reranker_enabled:
            return hits[:top_k]
        with tracer.step(
            "rerank",
            candidate_count=len(hits),
            input_tokens=self._estimate_tokens(query),
            retrieval_scores=[round(float(hit.get("score", 0.0)), 4) for hit in hits[:top_n]],
        ):
            reranked = self.reranker.rerank_hits(
                query,
                hits[:top_n],
                enabled=reranker_enabled,
                backend=backend,
                top_k=top_k,
                model_name=model_name,
                api_url=api_url,
                api_key=api_key,
            )
        return reranked[:top_k]

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

    def _build_response(
        self,
        *,
        tracer: Any,
        response_gen: Iterable[Any],
        sources: List[Dict[str, Any]],
        confidence_score: float,
        experiment_id: Optional[str],
        variant_id: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse:
        tracer.flush()
        payload = dict(metadata or {})
        payload.setdefault("trace", tracer.export())
        return RAGResponse(
            response_gen=response_gen,
            sources=sources,
            trace_id=tracer.trace_id,
            confidence_score=confidence_score,
            experiment_id=experiment_id,
            variant_id=variant_id,
            metadata=payload,
        )

    def _direct_answer_stream(self, query: str, history: List[Dict[str, Any]], tracer: Any):
        direct_answer = self._simple_direct_answer(query)
        if direct_answer or self.mock_llm:
            return self._mock_stream(direct_answer or f"Direct response: {query}")

        with tracer.step("generate", prompt_size=0, history_size=len(history)):
            messages = [{"role": "system", "content": "直接回答用户问题；无需进行知识库检索。"}]
            for msg in history:
                role = "assistant" if msg.get("isAi") else "user"
                messages.append({"role": role, "content": msg.get("text", "")})
            messages.append({"role": "user", "content": query})
            return self.llm_client.chat.completions.create(
                model=config.llm_model,
                messages=messages,
                stream=True,
            )

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
        tracer = create_tracer(
            enabled=config.observability_enabled,
            metadata={
                "query": query,
                "session_id": session_id,
                "experiment_id": experiment_id,
                "variant_id": variant_id,
            },
        )

        if config.self_rag_enabled:
            with tracer.step("self_rag_decision", input_size=len(query), input_tokens=self._estimate_tokens(query)):
                should_skip = self.self_rag.should_skip_retrieval(query)
            if should_skip:
                return self._build_response(
                    tracer=tracer,
                    response_gen=self._direct_answer_stream(query, history, tracer),
                    sources=[],
                    confidence_score=1.0,
                    experiment_id=experiment_id,
                    variant_id=variant_id,
                    metadata={"self_rag_skipped_retrieval": True},
                )

        with tracer.step("query_transform", input_size=len(query), input_tokens=self._estimate_tokens(query)):
            transformed_query = self._transform_query(query, overrides=overrides)

        hits = []
        for attempt in range(max(config.self_rag_max_retries, 0) + 1):
            hits = self._retrieve_candidates(query, transformed_query, tracer, overrides=overrides)
            confidence_score = self.confidence_service.score_hits(hits)
            if not config.self_rag_enabled:
                break
            with tracer.step(
                "self_rag_critique",
                attempt=attempt,
                retrieved=len(hits),
                confidence_score=confidence_score,
                retrieval_scores=[round(float(hit.get("score", 0.0)), 4) for hit in hits[:5]],
            ):
                critique_passed = self.self_rag.critique_hits(query, hits, confidence_score)
            if critique_passed:
                break
            if not self.self_rag.should_retry(hits, confidence_score, attempt):
                break
            retry_overrides = dict(overrides or {})
            retry_overrides.update({"query_transform_enabled": True, "query_transform_strategy": "rewrite"})
            transformed_query = self._transform_query(query, overrides=retry_overrides)

        hits = self._rerank_candidates(query, hits, tracer, overrides=overrides)
        confidence_assessment = self.confidence_service.evaluate(query, hits)
        low_confidence = bool(confidence_assessment.get("is_low_confidence", False))
        confidence_score = float(confidence_assessment.get("confidence_score", 0.0))

        with tracer.step(
            "context",
            retrieved=len(hits),
            confidence_score=confidence_score,
            retrieval_scores=confidence_assessment.get("score_details", {}).get("scores"),
            input_tokens=self._estimate_tokens(query),
        ):
            context_text = self._build_context(hits)
            source_details = self._source_details_from_hits(hits)

        if low_confidence:
            event = self.confidence_service.log_event(
                query=query,
                hits=hits,
                assessment=confidence_assessment,
                trace_id=tracer.trace_id,
                session_id=session_id,
            )
            return self._build_response(
                tracer=tracer,
                response_gen=self._mock_stream(config.low_confidence_message),
                sources=source_details,
                confidence_score=confidence_score,
                experiment_id=experiment_id,
                variant_id=variant_id,
                metadata={
                    "low_confidence": True,
                    "low_confidence_event_id": event.get("event_id"),
                    "confidence_details": confidence_assessment.get("score_details", {}),
                },
            )

        if self.mock_llm:
            if source_details:
                source_files = ", ".join(sorted(set(str(s["source_file"]) for s in source_details if s.get("source_file"))))
                answer = f"Mock answer based on sources {source_files}. Query: {query}"
            else:
                answer = f"Mock answer: no relevant context found. Query: {query}"
            return self._build_response(
                tracer=tracer,
                response_gen=self._mock_stream(answer),
                sources=source_details,
                confidence_score=confidence_score,
                experiment_id=experiment_id,
                variant_id=variant_id,
            )

        if config.self_rag_enabled:
            with tracer.step(
                "generate",
                prompt_size=len(context_text),
                history_size=len(history),
                mode="self_rag_non_stream",
                input_tokens=self._estimate_tokens(context_text) + self._estimate_tokens(query),
            ):
                messages = self._build_messages(query, history, context_text)
                response = self.llm_client.chat.completions.create(
                    model=config.llm_model,
                    messages=messages,
                    stream=False,
                )
                answer = self._completion_text(response)
            with tracer.step(
                "self_rag_answer_check",
                answer_size=len(answer),
                output_tokens=self._estimate_tokens(answer),
            ):
                if not self.self_rag.critique_answer(query, answer, hits):
                    retry_response = self.llm_client.chat.completions.create(
                        model=config.llm_model,
                        messages=messages,
                        stream=False,
                    )
                    answer = self._completion_text(retry_response)
            return self._build_response(
                tracer=tracer,
                response_gen=self._mock_stream(answer),
                sources=source_details,
                confidence_score=confidence_score,
                experiment_id=experiment_id,
                variant_id=variant_id,
                metadata={
                    "self_rag_answer_checked": True,
                    "confidence_details": confidence_assessment.get("score_details", {}),
                },
            )

        with tracer.step(
            "generate",
            prompt_size=len(context_text),
            history_size=len(history),
            input_tokens=self._estimate_tokens(context_text) + self._estimate_tokens(query),
        ):
            messages = self._build_messages(query, history, context_text)
            response = self.llm_client.chat.completions.create(
                model=config.llm_model,
                messages=messages,
                stream=True,
            )

        return self._build_response(
            tracer=tracer,
            response_gen=response,
            sources=source_details,
            confidence_score=confidence_score,
            experiment_id=experiment_id,
            variant_id=variant_id,
            metadata={"confidence_details": confidence_assessment.get("score_details", {})},
        )

    def generate_answer_text(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None,
        *,
        session_id: str = "default",
        overrides: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None,
        variant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = self.generate_response(
            query,
            history=history,
            session_id=session_id,
            overrides=overrides,
            experiment_id=experiment_id,
            variant_id=variant_id,
        )
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
            "metadata": result.metadata,
        }

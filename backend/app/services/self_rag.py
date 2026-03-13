import re
from typing import Any, Dict, List, Optional

from ..config import config
from .llm_utils import complete_text, is_mock_backend


class SelfRAGController:
    def __init__(self, rag_service=None):
        self.rag_service = rag_service
        self.mock_llm = is_mock_backend()

    @staticmethod
    def _critique_model() -> Optional[str]:
        return config.self_rag_critique_model or config.llm_model

    @staticmethod
    def _build_context_snippet(hits: List[Dict[str, Any]], limit: int = 3) -> str:
        return "\n\n".join(str(hit.get("payload", {}).get("chunk_text", ""))[:700] for hit in hits[:limit])

    @staticmethod
    def _heuristic_skip_retrieval(query: str) -> bool:
        query = (query or "").strip()
        if not query:
            return False
        if re.fullmatch(r"\d+\s*[\+\-\*\/]\s*\d+", query):
            return True
        if re.fullmatch(r"what is \d+\s*[\+\-\*\/]\s*\d+\??", query.lower()):
            return True
        return query.lower() in {"hi", "hello", "你好", "谢谢", "thanks", "what is 2+2", "2+2"}

    def should_skip_retrieval(self, query: str) -> bool:
        if self.mock_llm:
            return self._heuristic_skip_retrieval(query)

        verdict = complete_text(
            [
                {
                    "role": "user",
                    "content": (
                        "Reply RETRIEVE if the query needs knowledge-base retrieval. "
                        "Reply DIRECT if it can be answered directly.\n\n"
                        f"Query: {query}"
                    ),
                }
            ],
            model=self._critique_model(),
        ).upper()
        if "DIRECT" in verdict and "RETRIEVE" not in verdict:
            return True
        return self._heuristic_skip_retrieval(query)

    def critique_hits(self, query: str, hits: List[Dict[str, Any]], confidence_score: float) -> bool:
        if not hits:
            return False
        if confidence_score >= max(config.low_confidence_threshold, 0.2):
            return True
        if not self.mock_llm:
            context = self._build_context_snippet(hits)
            verdict = complete_text(
                [
                    {
                        "role": "user",
                        "content": (
                            "Reply SUFFICIENT if the retrieved context is relevant and enough to answer the query. "
                            "Reply INSUFFICIENT otherwise.\n\n"
                            f"Query: {query}\n\nContext:\n{context}"
                        ),
                    }
                ],
                model=self._critique_model(),
            ).upper()
            if "SUFFICIENT" in verdict and "INSUFFICIENT" not in verdict:
                return True
        query_tokens = set((query or "").lower().split())
        for hit in hits:
            chunk_text = str(hit.get("payload", {}).get("chunk_text", "")).lower()
            if query_tokens and any(token in chunk_text for token in query_tokens):
                return True
        return False

    @staticmethod
    def should_retry(hits: List[Dict[str, Any]], confidence_score: float, attempt: int) -> bool:
        return attempt < max(config.self_rag_max_retries, 0) and (not hits or confidence_score < 0.3)

    def critique_answer(self, query: str, answer: str, hits: List[Dict[str, Any]]) -> bool:
        if self.mock_llm:
            return True
        verdict = complete_text(
            [
                {
                    "role": "user",
                    "content": (
                        "Reply PASS if the answer is supported by the provided context. "
                        "Reply FAIL if it hallucinates or is unsupported.\n\n"
                        f"Query: {query}\n\nContext:\n{self._build_context_snippet(hits)}\n\nAnswer:\n{answer}"
                    ),
                }
            ],
            model=self._critique_model(),
        ).upper()
        return "PASS" in verdict and "FAIL" not in verdict

    def generate_response(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None,
        session_id: str = "default",
        **kwargs,
    ):
        if not self.rag_service:
            raise RuntimeError("SelfRAGController.generate_response requires an attached rag_service")
        return self.rag_service.generate_response(query, history=history, session_id=session_id, **kwargs)

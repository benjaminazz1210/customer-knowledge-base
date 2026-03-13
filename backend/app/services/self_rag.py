import re
from typing import Any, Dict, List, Optional

from ..config import config


class SelfRAGController:
    def __init__(self, rag_service=None):
        self.rag_service = rag_service

    @staticmethod
    def should_skip_retrieval(query: str) -> bool:
        query = (query or "").strip()
        if not query:
            return False
        if re.fullmatch(r"\d+\s*[\+\-\*\/]\s*\d+", query):
            return True
        if re.fullmatch(r"what is \d+\s*[\+\-\*\/]\s*\d+\??", query.lower()):
            return True
        return query.lower() in {"hi", "hello", "你好", "谢谢", "thanks", "what is 2+2", "2+2"}

    @staticmethod
    def critique_hits(query: str, hits: List[Dict[str, Any]], confidence_score: float) -> bool:
        if not hits:
            return False
        if confidence_score >= max(config.low_confidence_threshold, 0.2):
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

    def generate_response(self, query: str, history: Optional[List[Dict[str, Any]]] = None, session_id: str = "default"):
        if not self.rag_service:
            raise RuntimeError("SelfRAGController.generate_response requires an attached rag_service")
        return self.rag_service.generate_response(query, history=history, session_id=session_id)

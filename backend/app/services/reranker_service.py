import logging
import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import requests

from ..config import config

logger = logging.getLogger("nexusai.reranker")


class RerankerService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RerankerService, cls).__new__(cls)
            cls._instance._init_once()
        return cls._instance

    def _init_once(self):
        self.backend = (config.reranker_backend or "local").strip().lower()
        self.model = None

    def _ensure_local_model(self):
        if self.model is not None:
            return
        from sentence_transformers import CrossEncoder  # pragma: no cover - optional heavy dependency

        self.model = CrossEncoder(config.reranker_model)

    @staticmethod
    def _as_hit(item: Any, index: int) -> Dict[str, Any]:
        if isinstance(item, dict):
            payload = item.get("payload", {}) or {}
            payload.setdefault("chunk_index", index)
            return {"payload": payload, "score": float(item.get("score", 0.0))}
        text, score = item
        return {"payload": {"chunk_text": str(text), "chunk_index": index}, "score": float(score)}

    @staticmethod
    def _token_overlap_score(query: str, chunk_text: str, base_score: float) -> float:
        query_tokens = set((query or "").lower().split())
        chunk_tokens = set((chunk_text or "").lower().split())
        overlap = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
        length_penalty = min(len(chunk_tokens) / 50.0, 1.0)
        return round((0.65 * overlap) + (0.25 * base_score) + (0.10 * length_penalty), 6)

    def rerank_tuples(self, query: str, items: Sequence[Tuple[str, float]]) -> List[Tuple[str, float]]:
        hits = [self._as_hit(item, idx) for idx, item in enumerate(items)]
        reranked_hits = self.rerank_hits(query, hits)
        return [(hit.get("payload", {}).get("chunk_text", ""), float(hit.get("score", 0.0))) for hit in reranked_hits]

    def rerank(self, query: str, items, top_k: int = None, **kwargs):
        if not items:
            return []
        first = items[0]
        if isinstance(first, tuple):
            if getattr(self, "backend", config.reranker_backend) == "mock":
                hits = [self._as_hit(item, idx) for idx, item in enumerate(items)]
                reranked_hits = self._rerank_mock(query, hits)
                if top_k is not None:
                    reranked_hits = reranked_hits[:top_k]
                return [
                    (hit.get("payload", {}).get("chunk_text", ""), float(hit.get("score", 0.0)))
                    for hit in reranked_hits
                ]
            result = self.rerank_tuples(query, items)
            return result[:top_k] if top_k is not None else result
        result = self.rerank_hits(query, items)
        return result[:top_k] if top_k is not None else result

    rerank_candidates = rerank

    def rerank_hits(self, query: str, hits: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized_hits = [self._as_hit(item, idx) for idx, item in enumerate(hits)]
        if not config.reranker_enabled or not normalized_hits:
            return normalized_hits[: config.reranker_top_k]

        backend = self.backend
        if backend == "api":
            return self._rerank_api(query, normalized_hits)
        if backend == "local":
            return self._rerank_local(query, normalized_hits)
        return self._rerank_mock(query, normalized_hits)

    def _rerank_local(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            self._ensure_local_model()
            scores = self.model.predict(
                [[query, str(hit.get("payload", {}).get("chunk_text", ""))] for hit in hits]
            )
            reranked = []
            for hit, score in zip(hits, scores):
                item = dict(hit)
                item["score"] = float(score)
                reranked.append(item)
            reranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
            return reranked[: config.reranker_top_k]
        except Exception as exc:
            logger.warning("⚠️ Local reranker unavailable, falling back to mock scoring: %s", exc)
            return self._rerank_mock(query, hits)

    def _rerank_api(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not config.reranker_api_url:
            return self._rerank_mock(query, hits)
        payload = {
            "model": config.reranker_model,
            "query": query,
            "documents": [str(hit.get("payload", {}).get("chunk_text", "")) for hit in hits],
            "top_n": config.reranker_top_k,
        }
        headers = {}
        if config.reranker_api_key:
            headers["Authorization"] = f"Bearer {config.reranker_api_key}"
        try:
            response = requests.post(config.reranker_api_url, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            scored: List[Dict[str, Any]] = []
            for row in results:
                index = int(row.get("index", 0))
                hit = dict(hits[index])
                hit["score"] = float(row.get("relevance_score", row.get("score", 0.0)))
                scored.append(hit)
            scored.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
            return scored[: config.reranker_top_k]
        except Exception as exc:
            logger.warning("⚠️ API reranker unavailable, falling back to mock scoring: %s", exc)
            return self._rerank_mock(query, hits)

    def _rerank_mock(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        reranked = []
        for hit in hits:
            chunk_text = str(hit.get("payload", {}).get("chunk_text", ""))
            score = self._token_overlap_score(query, chunk_text, float(hit.get("score", 0.0)))
            item = dict(hit)
            item["score"] = 1.0 / (1.0 + math.exp(-5.0 * (score - 0.5)))
            reranked.append(item)
        reranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return reranked[: config.reranker_top_k]

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
        self.model_name = ""

    def _ensure_local_model(self, model_name: str):
        if self.model is not None and self.model_name == model_name:
            return
        from sentence_transformers import CrossEncoder  # pragma: no cover - optional heavy dependency

        self.model = CrossEncoder(model_name)
        self.model_name = model_name

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

    def rerank_tuples(self, query: str, items: Sequence[Tuple[str, float]], **kwargs) -> List[Tuple[str, float]]:
        hits = [self._as_hit(item, idx) for idx, item in enumerate(items)]
        reranked_hits = self.rerank_hits(query, hits, **kwargs)
        return [(hit.get("payload", {}).get("chunk_text", ""), float(hit.get("score", 0.0))) for hit in reranked_hits]

    def rerank(self, query: str, items, top_k: int = None, **kwargs):
        if not items:
            return []
        first = items[0]
        if isinstance(first, tuple):
            backend = (kwargs.get("backend") or getattr(self, "backend", config.reranker_backend)).strip().lower()
            if backend == "mock":
                hits = [self._as_hit(item, idx) for idx, item in enumerate(items)]
                reranked_hits = self._rerank_mock(query, hits)
                if top_k is not None:
                    reranked_hits = reranked_hits[:top_k]
                return [
                    (hit.get("payload", {}).get("chunk_text", ""), float(hit.get("score", 0.0)))
                    for hit in reranked_hits
                ]
            result = self.rerank_tuples(query, items, **kwargs)
            return result[:top_k] if top_k is not None else result
        result = self.rerank_hits(query, items, **kwargs)
        return result[:top_k] if top_k is not None else result

    rerank_candidates = rerank

    def rerank_hits(
        self,
        query: str,
        hits: Iterable[Dict[str, Any]],
        *,
        enabled: bool = None,
        backend: str = None,
        top_k: int = None,
        model_name: str = None,
        api_url: str = None,
        api_key: str = None,
    ) -> List[Dict[str, Any]]:
        normalized_hits = [self._as_hit(item, idx) for idx, item in enumerate(hits)]
        if not normalized_hits:
            return []

        enabled = config.reranker_enabled if enabled is None else bool(enabled)
        limit = config.reranker_top_k if top_k is None else max(int(top_k), 0)
        if not enabled:
            return normalized_hits[:limit]

        backend = (backend or self.backend).strip().lower()
        model_name = model_name or config.reranker_model
        api_url = api_url if api_url is not None else config.reranker_api_url
        api_key = api_key if api_key is not None else config.reranker_api_key
        if backend == "api":
            reranked = self._rerank_api(query, normalized_hits, top_k=limit, model_name=model_name, api_url=api_url, api_key=api_key)
        elif backend == "local":
            reranked = self._rerank_local(query, normalized_hits, model_name=model_name)
        else:
            reranked = self._rerank_mock(query, normalized_hits)

        return reranked[:limit]

    def _rerank_local(self, query: str, hits: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
        try:
            self._ensure_local_model(model_name)
            scores = self.model.predict(
                [[query, str(hit.get("payload", {}).get("chunk_text", ""))] for hit in hits]
            )
            reranked = []
            for hit, score in zip(hits, scores):
                item = dict(hit)
                item["score"] = float(score)
                reranked.append(item)
            reranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
            return reranked
        except Exception as exc:
            logger.warning("⚠️ Local reranker unavailable, falling back to mock scoring: %s", exc)
            return self._rerank_mock(query, hits)

    def _rerank_api(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        *,
        top_k: int,
        model_name: str,
        api_url: str,
        api_key: str,
    ) -> List[Dict[str, Any]]:
        if not api_url:
            return self._rerank_mock(query, hits)
        payload = {
            "model": model_name,
            "query": query,
            "documents": [str(hit.get("payload", {}).get("chunk_text", "")) for hit in hits],
            "top_n": top_k,
        }
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=15)
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
            return scored
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
        return reranked

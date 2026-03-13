import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..config import config
from .llm_utils import complete_text


@dataclass
class TransformedQuery:
    strategy: str
    original_query: str
    search_queries: List[str]
    embedding_query: str
    rewritten_query: str = ""
    hypothetical_answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryTransformer:
    def __init__(
        self,
        *,
        enabled: bool = None,
        strategy: str = None,
        model: str = None,
        multi_query_count: int = None,
    ):
        self.enabled = config.query_transform_enabled if enabled is None else bool(enabled)
        self.strategy = (strategy or config.query_transform_strategy or "rewrite").strip().lower()
        self.model = model or config.query_transform_model
        self.multi_query_count = int(config.multi_query_count if multi_query_count is None else multi_query_count)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip())

    def transform(self, query: str) -> TransformedQuery:
        query = self._normalize(query)
        strategy = self.strategy
        if not self.enabled or strategy in ("", "none"):
            return TransformedQuery(strategy="none", original_query=query, search_queries=[query], embedding_query=query)
        if strategy == "decompose":
            return self._decompose(query)
        if strategy == "hyde":
            return self._hyde(query)
        if strategy == "multi_query":
            return self._multi_query(query)
        return self._rewrite(query)

    def _rewrite(self, query: str) -> TransformedQuery:
        prompt = (
            "Rewrite the user query into a concise search-friendly query. Return the rewritten query only.\n\n"
            f"Query: {query}"
        )
        rewritten = self._normalize(
            complete_text([{"role": "user", "content": prompt}], model=self.model)
        )
        if not rewritten:
            rewritten = query
        return TransformedQuery(
            strategy="rewrite",
            original_query=query,
            search_queries=[rewritten],
            embedding_query=rewritten,
            rewritten_query=rewritten,
        )

    def _decompose(self, query: str) -> TransformedQuery:
        fragments = re.split(r"[?？]|(?:\s+and\s+)|(?:\s+&\s+)|[，,；;、]|以及|并且|和", query)
        sub_queries = [self._normalize(fragment) for fragment in fragments if self._normalize(fragment)]
        if len(sub_queries) <= 1:
            sub_queries = [query]
        return TransformedQuery(
            strategy="decompose",
            original_query=query,
            search_queries=sub_queries,
            embedding_query=query,
            metadata={"sub_queries": sub_queries},
        )

    def _hyde(self, query: str) -> TransformedQuery:
        prompt = (
            "Write a short hypothetical answer that could appear in a knowledge-base article answering this query.\n\n"
            f"Query: {query}"
        )
        hypothetical = self._normalize(
            complete_text([{"role": "user", "content": prompt}], model=self.model)
        )
        if not hypothetical:
            hypothetical = f"Knowledge base answer for: {query}"
        return TransformedQuery(
            strategy="hyde",
            original_query=query,
            search_queries=[query],
            embedding_query=hypothetical,
            hypothetical_answer=hypothetical,
        )

    def _multi_query(self, query: str) -> TransformedQuery:
        variations = [query]
        templates = [
            f"{query} 详细说明",
            f"{query} implementation details",
            f"{query} FAQ",
            f"{query} best practices",
        ]
        for candidate in templates:
            normalized = self._normalize(candidate)
            if normalized not in variations:
                variations.append(normalized)
            if len(variations) >= max(self.multi_query_count, 1):
                break
        return TransformedQuery(
            strategy="multi_query",
            original_query=query,
            search_queries=variations,
            embedding_query=query,
            metadata={"variations": variations},
        )

    def multi_query(self, query: str) -> Dict[str, Any]:
        transformed = self._multi_query(query)
        return {
            "queries": transformed.search_queries,
            "metadata": transformed.metadata,
        }

    @staticmethod
    def reciprocal_rank_fusion(rankings: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for result_list in rankings:
            for rank, item in enumerate(result_list, start=1):
                payload = item.get("payload", {}) or {}
                key = "{}::{}".format(payload.get("source_file"), payload.get("chunk_index", payload.get("parent_id", rank)))
                entry = merged.setdefault(
                    key,
                    {
                        "payload": payload,
                        "score": 0.0,
                        "vector_score": item.get("vector_score", 0.0),
                        "keyword_score": item.get("keyword_score", 0.0),
                    },
                )
                entry["score"] += 1.0 / (k + rank)
                entry["vector_score"] = max(float(entry.get("vector_score", 0.0)), float(item.get("vector_score", 0.0)))
                entry["keyword_score"] = max(float(entry.get("keyword_score", 0.0)), float(item.get("keyword_score", 0.0)))

        fused = list(merged.values())
        fused.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return fused

    def merge_results(self, ranked_lists: List[List[Dict[str, Any]]], limit: int = 5) -> List[Dict[str, Any]]:
        return self.reciprocal_rank_fusion(ranked_lists)[:limit]

    merge_multi_query_results = merge_results

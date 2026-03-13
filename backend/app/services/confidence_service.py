from typing import Dict, List

from ..config import config
from .llm_utils import complete_text


class LowConfidenceService:
    @staticmethod
    def score_hits(hits: List[Dict]) -> float:
        if not hits:
            return 0.0
        top_hits = hits[: max(int(config.reranker_top_k), 1)]
        return sum(float(hit.get("score", 0.0)) for hit in top_hits) / max(len(top_hits), 1)

    @staticmethod
    def score_values(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(float(value) for value in values) / max(len(values), 1)

    def evaluate(self, query_or_scores, hits: List[Dict] = None) -> Dict[str, float]:
        if hits is None and isinstance(query_or_scores, list):
            score = self.score_values(query_or_scores)
            threshold = getattr(self, "threshold", config.low_confidence_threshold)
            return {
                "is_low_confidence": score < float(threshold),
                "confidence_score": score,
            }

        query = query_or_scores
        hits = hits or []
        score = self.score_hits(hits)
        if not config.low_confidence_enabled:
            return {"is_low_confidence": False, "confidence_score": score}

        if config.low_confidence_strategy == "llm_judge":
            context = "\n\n".join(
                str(hit.get("payload", {}).get("chunk_text", ""))[:600] for hit in hits[:3]
            )
            verdict = complete_text(
                [
                    {
                        "role": "user",
                        "content": (
                            "Reply YES if the provided context is sufficient to answer the query accurately, "
                            "otherwise reply NO.\n\n"
                            f"Query: {query}\n\nContext:\n{context}"
                        ),
                    }
                ]
            ).strip().upper()
            return {"is_low_confidence": "NO" in verdict, "confidence_score": score}

        return {
            "is_low_confidence": score < float(config.low_confidence_threshold),
            "confidence_score": score,
        }

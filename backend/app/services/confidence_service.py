import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import redis

from ..config import config
from .llm_utils import complete_text

logger = logging.getLogger("nexusai.confidence")


class LowConfidenceService:
    def __init__(self, redis_client=None, event_log_path: Optional[str] = None, state_store=None):
        self.client = redis_client
        self.state_store = state_store
        self._memory: Dict[str, Dict[str, Any]] = {}
        self.event_log_path = event_log_path or config.low_confidence_log_path

        if self.state_store is not None:
            return
        if self.client is not None:
            return

        host = str(os.getenv("REDIS_HOST") or config.redis_host)
        port = int(os.getenv("REDIS_PORT") or config.redis_port)
        db = int(os.getenv("REDIS_DB") or config.redis_db)
        try:
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.client.ping()
        except Exception as exc:
            logger.warning("LowConfidenceService fallback to file/memory store: %s", exc)
            self.client = None

    @staticmethod
    def _normalize_score(score: Any) -> float:
        try:
            value = float(score)
        except Exception:
            return 0.0
        return max(0.0, min(value, 1.0))

    @classmethod
    def _top_scores(cls, hits: List[Dict]) -> List[float]:
        top_hits = hits[: max(int(config.reranker_top_k), 1)]
        return [cls._normalize_score(hit.get("score", 0.0)) for hit in top_hits]

    @classmethod
    def score_hits(cls, hits: List[Dict]) -> float:
        scores = cls._top_scores(hits)
        if not scores:
            return 0.0

        base_average = sum(scores) / len(scores)
        coverage = min(len(scores) / max(int(config.reranker_top_k), 1), 1.0)
        peak = max(scores)

        if len(scores) == 1:
            blended = (0.75 * peak) + (0.25 * base_average)
        elif len(scores) == 2:
            blended = (0.65 * peak) + (0.35 * base_average)
        else:
            blended = base_average

        return round(blended * (0.5 + (0.5 * coverage)), 6)

    @classmethod
    def score_values(cls, values: List[float]) -> float:
        if not values:
            return 0.0
        normalized = [cls._normalize_score(value) for value in values]
        return round(sum(normalized) / max(len(normalized), 1), 6)

    def _ttl_seconds(self) -> int:
        return max(int(config.low_confidence_ttl_days), 1) * 24 * 3600

    @staticmethod
    def _event_key(event_id: str) -> str:
        return f"low_confidence:event:{event_id}"

    def _set_event(self, key: str, payload: Dict[str, Any]) -> None:
        if self.state_store is not None:
            self.state_store.set_json(key, payload, ttl_seconds=self._ttl_seconds())
            return
        if self.client:
            raw = json.dumps(payload, ensure_ascii=False)
            setex = getattr(self.client, "setex", None)
            if callable(setex):
                setex(key, self._ttl_seconds(), raw)
            else:
                self.client.set(key, raw)
            return
        self._memory[key] = payload

    def list_events(self) -> List[Dict[str, Any]]:
        if self.state_store is not None:
            events = self.state_store.list_prefix("low_confidence:event:")
            return sorted(events, key=lambda item: float(item.get("timestamp", 0.0)))
        if self.client:
            events: List[Dict[str, Any]] = []
            for key in self.client.scan_iter(match="low_confidence:event:*"):
                raw = self.client.get(key)
                if raw:
                    events.append(json.loads(raw))
            return sorted(events, key=lambda item: float(item.get("timestamp", 0.0)))
        return sorted(self._memory.values(), key=lambda item: float(item.get("timestamp", 0.0)))

    def log_event(
        self,
        *,
        query: str,
        hits: List[Dict[str, Any]],
        assessment: Dict[str, Any],
        trace_id: str = "",
        session_id: str = "default",
    ) -> Dict[str, Any]:
        event = {
            "event_id": str(uuid.uuid4()),
            "query": query,
            "trace_id": trace_id,
            "session_id": session_id,
            "timestamp": time.time(),
            "confidence_score": float(assessment.get("confidence_score", 0.0)),
            "top_scores": list(assessment.get("top_scores", [])),
            "hit_count": int(assessment.get("hit_count", len(hits))),
            "threshold": float(assessment.get("threshold", config.low_confidence_threshold)),
            "strategy": str(assessment.get("strategy", config.low_confidence_strategy)),
            "sources": [
                {
                    "source_file": (hit.get("payload", {}) or {}).get("source_file"),
                    "chunk_index": (hit.get("payload", {}) or {}).get("chunk_index"),
                    "score": float(hit.get("score", 0.0)),
                }
                for hit in hits[: max(int(config.reranker_top_k), 3)]
            ],
        }
        self._set_event(self._event_key(event["event_id"]), event)

        if self.event_log_path:
            try:
                directory = os.path.dirname(self.event_log_path)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                with open(self.event_log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            except Exception as exc:
                logger.info("Failed to append low-confidence event log file: %s", exc)
        return event

    def evaluate(self, query_or_scores, hits: List[Dict] = None) -> Dict[str, Any]:
        if hits is None and isinstance(query_or_scores, list):
            score = self.score_values(query_or_scores)
            threshold = float(getattr(self, "threshold", config.low_confidence_threshold))
            scores = [self._normalize_score(value) for value in query_or_scores]
            return {
                "is_low_confidence": score < threshold,
                "confidence_score": score,
                "top_scores": scores,
                "hit_count": len(query_or_scores),
                "threshold": threshold,
                "strategy": config.low_confidence_strategy,
                "score_details": {
                    "scores": scores,
                    "hit_count": len(query_or_scores),
                    "threshold": threshold,
                    "strategy": config.low_confidence_strategy,
                },
            }

        query = str(query_or_scores or "")
        hits = hits or []
        top_scores = self._top_scores(hits)
        score = self.score_hits(hits)
        threshold = float(config.low_confidence_threshold)
        result: Dict[str, Any] = {
            "confidence_score": score,
            "top_scores": top_scores,
            "hit_count": len(hits),
            "threshold": threshold,
            "strategy": config.low_confidence_strategy,
            "score_details": {
                "scores": top_scores,
                "hit_count": len(hits),
                "threshold": threshold,
                "strategy": config.low_confidence_strategy,
            },
        }
        if not config.low_confidence_enabled:
            result["is_low_confidence"] = False
            return result

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
            result["is_low_confidence"] = "NO" in verdict
            return result

        result["is_low_confidence"] = score < threshold
        return result

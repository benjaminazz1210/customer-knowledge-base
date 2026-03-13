import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import redis

from ..config import config

logger = logging.getLogger("nexusai.feedback")


class FeedbackService:
    def __init__(self, redis_client=None, ttl_seconds: Optional[int] = None):
        self.client = redis_client
        self._memory = {}
        self._ttl_override = ttl_seconds

        if self.client is not None:
            return

        host = str(os.getenv("REDIS_HOST") or config.redis_host)
        port = int(os.getenv("REDIS_PORT") or config.redis_port)
        db = int(os.getenv("REDIS_DB") or config.redis_db)
        try:
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.client.ping()
        except Exception as exc:
            logger.warning("FeedbackService fallback to in-memory store: %s", exc)
            self.client = None

    def _ttl_seconds(self) -> int:
        if self._ttl_override is not None:
            return int(self._ttl_override)
        return max(int(config.feedback_ttl_days), 1) * 24 * 3600

    @staticmethod
    def _artifact_key(trace_id: str) -> str:
        return f"feedback:artifact:{trace_id}"

    @staticmethod
    def _item_key(feedback_id: str) -> str:
        return f"feedback:item:{feedback_id}"

    @staticmethod
    def _session_key(session_id: str) -> str:
        return f"feedback:session:{session_id}"

    def _set_json(self, key: str, payload: Dict[str, Any]) -> None:
        if self.client:
            raw = json.dumps(payload, ensure_ascii=False)
            setex = getattr(self.client, "setex", None)
            if callable(setex):
                setex(key, self._ttl_seconds(), raw)
            else:
                self.client.set(key, raw)
                expire = getattr(self.client, "expire", None)
                if callable(expire):
                    expire(key, self._ttl_seconds())
            return
        self._memory[key] = payload

    def _get_json(self, key: str) -> Optional[Dict[str, Any]]:
        if self.client:
            raw = self.client.get(key)
            return json.loads(raw) if raw else None
        return self._memory.get(key)

    def save_chat_artifact(self, trace_id: str, payload: Dict[str, Any]) -> None:
        record = dict(payload)
        record.setdefault("trace_id", trace_id)
        record.setdefault("timestamp", time.time())
        self._set_json(self._artifact_key(trace_id), record)

    def register_turn(
        self,
        *,
        trace_id: str,
        session_id: str,
        query: str,
        answer: str,
        sources: List[Dict[str, Any]],
        confidence_score: float,
    ) -> Dict[str, Any]:
        artifact = {
            "trace_id": trace_id,
            "session_id": session_id,
            "query": query,
            "answer": answer,
            "sources": sources,
            "confidence_score": confidence_score,
            "timestamp": time.time(),
        }
        self.save_chat_artifact(trace_id, artifact)
        self._set_json(self._session_key(session_id), artifact)
        return artifact

    def capture_implicit_feedback(self, session_id: str, new_query: str) -> Optional[Dict[str, Any]]:
        previous = self._get_json(self._session_key(session_id))
        if not isinstance(previous, dict):
            return None
        old_tokens = set(str(previous.get("query", "")).lower().split())
        new_tokens = set((new_query or "").lower().split())
        if not old_tokens or not new_tokens:
            return None
        overlap = len(old_tokens & new_tokens) / max(len(old_tokens | new_tokens), 1)
        if overlap < 0.45:
            return None
        return self.save_feedback(
            previous.get("trace_id", str(uuid.uuid4())),
            {
                "message_id": previous.get("trace_id"),
                "rating": "implicit_negative",
                "comment": "follow_up_rephrase_detected",
                "query": previous.get("query", ""),
                "answer": previous.get("answer", ""),
                "sources": previous.get("sources", []),
                "session_id": session_id,
            },
        )

    def add_feedback(self, trace_id: str, message_id: str, rating: str, comment: str = "") -> Dict[str, Any]:
        return self.submit_feedback(trace_id=trace_id, rating=rating, comment=comment, message_id=message_id)

    def submit_feedback(
        self,
        *,
        trace_id: str,
        rating: str,
        comment: str = "",
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        artifact = self.get_artifact(trace_id) or {}
        payload = {
            "trace_id": trace_id,
            "message_id": message_id or trace_id,
            "rating": rating,
            "comment": comment,
            "query": artifact.get("query", ""),
            "answer": artifact.get("answer", ""),
            "sources": artifact.get("sources", []),
            "confidence_score": artifact.get("confidence_score"),
            "timestamp": time.time(),
            "session_id": artifact.get("session_id", "default"),
        }
        feedback_id = str(uuid.uuid4())
        self._set_json(self._item_key(feedback_id), payload)
        return payload

    def save_feedback(self, trace_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        artifact = self.get_artifact(trace_id) or {}
        combined = dict(payload)
        combined.setdefault("trace_id", trace_id)
        combined.setdefault("timestamp", time.time())
        combined.setdefault("query", artifact.get("query", ""))
        combined.setdefault("answer", artifact.get("answer", ""))
        combined.setdefault("sources", artifact.get("sources", []))
        combined.setdefault("artifact", artifact)
        feedback_id = combined.setdefault("feedback_id", str(uuid.uuid4()))
        self._set_json(self._item_key(feedback_id), combined)
        return combined

    def get_artifact(self, trace_id: str) -> Optional[Dict[str, Any]]:
        return self._get_json(self._artifact_key(trace_id))

    def get_feedback(self, trace_id: str) -> Optional[Dict[str, Any]]:
        exported = self.export_feedback()
        for item in exported:
            if item.get("trace_id") == trace_id:
                return item
        return None

    def export_feedback(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if self.client:
            for key in self.client.scan_iter(match="feedback:item:*"):
                raw = self.client.get(key)
                if raw:
                    items.append(json.loads(raw))
        else:
            items.extend(value for key, value in self._memory.items() if key.startswith("feedback:item:"))
        return sorted(items, key=lambda item: item.get("timestamp", 0.0))

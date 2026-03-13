import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import redis

from ..config import config

logger = logging.getLogger("nexusai.state")


class StateStore:
    _instance = None
    _memory_store: Dict[str, Any] = {}
    _memory_expiry: Dict[str, float] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(StateStore, cls).__new__(cls)
            cls._instance._init_once()
        return cls._instance

    def _init_once(self):
        host = str(os.getenv("REDIS_HOST") or config.redis_host)
        port = int(os.getenv("REDIS_PORT") or config.redis_port)
        db = int(os.getenv("REDIS_DB") or config.redis_db)
        self.client = None
        try:
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.client.ping()
            logger.info("✅ StateStore connected to Redis.")
        except Exception as exc:
            logger.warning("⚠️ StateStore using in-memory fallback: %s", exc)
            self.client = None

    def _cleanup_memory(self):
        now = time.time()
        expired = [key for key, until in self._memory_expiry.items() if until <= now]
        for key in expired:
            self._memory_store.pop(key, None)
            self._memory_expiry.pop(key, None)

    @staticmethod
    def _serialize(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _deserialize(value: Optional[str]) -> Any:
        if value in (None, ""):
            return None
        return json.loads(value)

    def set_json(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        if self.client:
            if ttl_seconds:
                self.client.setex(key, ttl_seconds, self._serialize(value))
            else:
                self.client.set(key, self._serialize(value))
            return

        self._cleanup_memory()
        self._memory_store[key] = value
        if ttl_seconds:
            self._memory_expiry[key] = time.time() + ttl_seconds
        else:
            self._memory_expiry.pop(key, None)

    def get_json(self, key: str, default: Any = None) -> Any:
        if self.client:
            raw = self.client.get(key)
            return default if raw is None else self._deserialize(raw)

        self._cleanup_memory()
        return self._memory_store.get(key, default)

    def delete(self, key: str) -> None:
        if self.client:
            self.client.delete(key)
            return
        self._memory_store.pop(key, None)
        self._memory_expiry.pop(key, None)

    def append_json_list(self, key: str, item: Any, ttl_seconds: Optional[int] = None) -> None:
        current = self.get_json(key, default=[])
        if not isinstance(current, list):
            current = []
        current.append(item)
        self.set_json(key, current, ttl_seconds=ttl_seconds)

    def list_prefix(self, prefix: str) -> List[Any]:
        if self.client:
            values: List[Any] = []
            for key in self.client.scan_iter(match=f"{prefix}*"):
                raw = self.client.get(key)
                if raw is not None:
                    values.append(self._deserialize(raw))
            return values

        self._cleanup_memory()
        return [value for key, value in self._memory_store.items() if key.startswith(prefix)]

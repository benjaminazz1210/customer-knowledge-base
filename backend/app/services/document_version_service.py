import json
import logging
import os
import time
import uuid
from hashlib import sha256
from typing import Any, Dict, List, Optional

import redis

from ..config import config

logger = logging.getLogger("nexusai.document_versions")


class DocumentVersionService:
    def __init__(self, redis_client=None):
        self.client = redis_client
        self._memory = {}
        if self.client is not None:
            return

        host = str(os.getenv("REDIS_HOST") or config.redis_host)
        port = int(os.getenv("REDIS_PORT") or config.redis_port)
        db = int(os.getenv("REDIS_DB") or config.redis_db)
        try:
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.client.ping()
        except Exception as exc:
            logger.warning("DocumentVersionService fallback to in-memory store: %s", exc)
            self.client = None

    @staticmethod
    def compute_content_hash(content: bytes) -> str:
        return sha256(content or b"").hexdigest()

    compute_hash = compute_content_hash

    def _key(self, filename: str) -> str:
        return f"document:versions:{filename}"

    def _set_versions(self, filename: str, versions: List[Dict[str, Any]]) -> None:
        key = self._key(filename)
        if self.client:
            self.client.set(key, json.dumps(versions, ensure_ascii=False))
            return
        self._memory[key] = versions

    def get_versions(self, filename: str) -> List[Dict[str, Any]]:
        key = self._key(filename)
        if self.client:
            raw = self.client.get(key)
            return json.loads(raw) if raw else []
        return list(self._memory.get(key, []))

    def latest(self, filename: str) -> Optional[Dict[str, Any]]:
        versions = self.get_versions(filename)
        return versions[-1] if versions else None

    def latest_hash(self, filename: str) -> str:
        latest = self.latest(filename)
        if not latest:
            return ""
        return str(latest.get("content_hash") or latest.get("hash") or "")

    def is_unchanged(self, filename: str, content_hash: str) -> bool:
        return self.latest_hash(filename) == content_hash

    def record_version(
        self,
        filename: str,
        content_hash: str,
        chunks: List[Any],
        timestamp: Optional[Any] = None,
        raw_content: Optional[str] = None,
    ) -> Dict[str, Any]:
        versions = self.get_versions(filename)
        record = {
            "version_id": str(uuid.uuid4()),
            "filename": filename,
            "content_hash": content_hash,
            "hash": content_hash,
            "timestamp": timestamp if timestamp is not None else time.time(),
            "chunks": chunks,
            "chunk_ids": chunks,
            "raw_content": raw_content or "",
        }
        versions.append(record)
        self._set_versions(filename, versions)
        return record

    def get_version(self, filename: str, version_id: str) -> Optional[Dict[str, Any]]:
        for item in self.get_versions(filename):
            if item.get("version_id") == version_id:
                return item
        return None

    def rollback(self, filename: str, version_id: str) -> Dict[str, Any]:
        version = self.get_version(filename, version_id)
        if version is None:
            raise ValueError(f"Version {version_id} not found for {filename}")
        return version

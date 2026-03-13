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

    @classmethod
    def compute_chunk_hash(cls, chunk_text: str) -> str:
        return cls.compute_content_hash((chunk_text or "").encode("utf-8"))

    @staticmethod
    def generate_version_id() -> str:
        return str(uuid.uuid4())

    def _key(self, filename: str) -> str:
        return f"document:versions:{filename}"

    @staticmethod
    def _payload(chunk: Any) -> Dict[str, Any]:
        if isinstance(chunk, dict) and "payload" in chunk:
            return dict(chunk.get("payload", {}) or {})
        if isinstance(chunk, dict):
            payload = {key: value for key, value in chunk.items() if key != "metadata"}
            metadata = chunk.get("metadata", {}) or {}
            for key, value in metadata.items():
                payload.setdefault(key, value)
            return payload
        return dict(chunk or {})

    @classmethod
    def chunk_identity(cls, chunk: Any, fallback_index: int = 0) -> str:
        payload = cls._payload(chunk)
        delta_key = payload.get("delta_key")
        if delta_key:
            return str(delta_key)
        chunk_hash = payload.get("chunk_hash")
        if chunk_hash:
            return str(chunk_hash)
        chunk_index = payload.get("chunk_index", fallback_index)
        return f"chunk-index:{chunk_index}"

    @classmethod
    def index_chunks(cls, chunks: List[Any]) -> Dict[str, Dict[str, Any]]:
        indexed: Dict[str, Dict[str, Any]] = {}
        for idx, chunk in enumerate(chunks or []):
            payload = cls._payload(chunk)
            indexed[cls.chunk_identity(payload, fallback_index=idx)] = payload
        return indexed

    @classmethod
    def diff_chunks(cls, old_chunks: List[Any], new_chunks: List[Any]) -> Dict[str, Any]:
        old_index = cls.index_chunks(old_chunks)
        new_index = cls.index_chunks(new_chunks)
        added: List[Dict[str, Any]] = []
        unchanged: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(new_chunks or []):
            payload = cls._payload(chunk)
            identity = cls.chunk_identity(payload, fallback_index=idx)
            if old_index.get(identity) is None:
                added.append(payload)
                continue
            unchanged.append(payload)

        deleted = [identity for identity in old_index.keys() if identity not in new_index]
        return {
            "added": added,
            "updated": [],
            "deleted": deleted,
            "unchanged": unchanged,
        }

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
        *,
        version_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        versions = self.get_versions(filename)
        record = {
            "version_id": version_id or self.generate_version_id(),
            "filename": filename,
            "content_hash": content_hash,
            "hash": content_hash,
            "timestamp": timestamp if timestamp is not None else time.time(),
            "chunks": chunks,
            "chunk_ids": chunks,
            "raw_content": raw_content or "",
            "metadata": metadata or {},
            "embeddings": embeddings or [],
        }
        versions.append(record)
        self._set_versions(filename, versions)
        return record

    def get_version(self, filename: str, version_id: str) -> Optional[Dict[str, Any]]:
        for item in self.get_versions(filename):
            if item.get("version_id") == version_id:
                return item
        return None

    def activate_version(
        self,
        filename: str,
        version_id: str,
        *,
        activated_by: str = "rollback",
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        version = self.get_version(filename, version_id)
        if version is None:
            raise ValueError(f"Version {version_id} not found for {filename}")
        activated_record = {
            "version_id": self.generate_version_id(),
            "filename": filename,
            "content_hash": version.get("content_hash"),
            "hash": version.get("content_hash"),
            "timestamp": time.time() if timestamp is None else timestamp,
            "chunks": version.get("chunks", []),
            "chunk_ids": version.get("chunks", []),
            "raw_content": version.get("raw_content", ""),
            "metadata": {
                **(version.get("metadata", {}) or {}),
                "activated_from_version_id": version_id,
                "activation_reason": activated_by,
            },
            "embeddings": version.get("embeddings", []),
        }
        versions = self.get_versions(filename)
        versions.append(activated_record)
        self._set_versions(filename, versions)
        return activated_record

    def rollback(self, filename: str, version_id: str) -> Dict[str, Any]:
        version = self.get_version(filename, version_id)
        if version is None:
            raise ValueError(f"Version {version_id} not found for {filename}")
        return {
            "filename": filename,
            "version_id": version_id,
            "chunks": version.get("chunks", []),
            "embeddings": version.get("embeddings", []),
            "content_hash": version.get("content_hash"),
            "metadata": version.get("metadata", {}),
        }

import re
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from ..config import config


class VectorStore:
    def __init__(self):
        self.supports_text_index = False
        self.available = False
        self._memory_points: List[Dict[str, Any]] = []
        self.client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port, check_compatibility=False)
        try:
            self._ensure_collection()
            self.available = True
        except Exception as exc:
            print(f"⚠️ Qdrant unavailable, using in-memory vector store fallback: {exc}")

    def _is_available(self) -> bool:
        return bool(getattr(self, "available", True))

    def _memory_bucket(self) -> List[Dict[str, Any]]:
        bucket = getattr(self, "_memory_points", None)
        if bucket is None:
            bucket = []
            self._memory_points = bucket
        return bucket

    def _ensure_collection(self):
        try:
            collection_info = self.client.get_collection(config.collection_name)
            current_dim = collection_info.config.params.vectors.size
            if current_dim != config.vector_dimension:
                print(
                    f"⚠️ Vector dimension mismatch: {current_dim} vs {config.vector_dimension}. Recreating collection..."
                )
                self.client.delete_collection(config.collection_name)
                self._create_collection()
                return
        except Exception:
            self._create_collection()
            return

        self._ensure_payload_indexes()

    def _create_collection(self, collection_name: Optional[str] = None):
        name = collection_name or config.collection_name
        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=config.vector_dimension,
                distance=models.Distance.COSINE,
            ),
        )
        self._ensure_payload_indexes(collection_name=name)

    def _ensure_payload_indexes(self, collection_name: Optional[str] = None):
        name = collection_name or config.collection_name
        for field_name in ("source_file", "parent_id", "version_id", "content_hash"):
            try:
                self.client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception:
                pass
        try:
            self.client.create_payload_index(
                collection_name=name,
                field_name="chunk_text",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.MULTILINGUAL,
                    min_token_len=2,
                    lowercase=True,
                ),
                wait=True,
            )
            self.supports_text_index = True
        except Exception:
            self.supports_text_index = False

    @staticmethod
    def _build_point_id(filename: str, chunk_index: int) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"nexusai://{filename}/{chunk_index}"))

    @staticmethod
    def _payload_key(payload: Dict[str, Any], fallback: Any = None) -> str:
        source_file = payload.get("source_file")
        parent_id = payload.get("parent_id")
        chunk_index = payload.get("chunk_index")
        if source_file is not None and parent_id:
            return f"{source_file}::{parent_id}"
        if source_file is not None and chunk_index is not None:
            return f"{source_file}::{chunk_index}"
        return str(fallback)

    @staticmethod
    def _expand_hit_to_parent(hit: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(hit.get("payload", {}) or {})
        parent_text = payload.get("parent_text") or payload.get("parent_chunk_text")
        if parent_text:
            payload["chunk_text"] = parent_text
            payload["chunk_role"] = "parent_context"
        return {
            "payload": payload,
            "score": float(hit.get("score", 0.0)),
            "vector_score": float(hit.get("vector_score", hit.get("score", 0.0))),
            "keyword_score": float(hit.get("keyword_score", 0.0)),
        }

    def _dedupe_expanded_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: Dict[str, Dict[str, Any]] = {}
        for hit in hits:
            expanded = self._expand_hit_to_parent(hit)
            payload = expanded.get("payload", {}) or {}
            key = self._payload_key(payload, fallback=id(expanded))
            previous = deduped.get(key)
            if previous is None or float(expanded.get("score", 0.0)) > float(previous.get("score", 0.0)):
                deduped[key] = expanded
        ranked = list(deduped.values())
        ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return ranked

    def _expand_hits_to_parents(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._dedupe_expanded_hits(hits)

    def upsert_chunks(self, filename: str, chunks: List[Any], embeddings: List[List[float]]):
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            if isinstance(chunk, dict):
                chunk_text = str(chunk.get("chunk_text", ""))
                metadata = chunk.get("metadata", {}) or {}
            else:
                chunk_text = str(chunk)
                metadata = {}

            payload = {
                "source_file": filename,
                "chunk_text": chunk_text,
                "chunk_index": i,
            }
            if isinstance(chunk, dict) and chunk.get("child_id"):
                payload["child_id"] = chunk.get("child_id")
            for key, value in metadata.items():
                if value is not None:
                    payload[key] = value

            points.append(
                models.PointStruct(
                    id=self._build_point_id(filename, i),
                    vector=vector,
                    payload=payload,
                )
            )

        if not points:
            return

        if not getattr(self, "available", True):
            self._memory_points = [
                point for point in getattr(self, "_memory_points", []) if point["payload"].get("source_file") != filename
            ]
            self._memory_points.extend(
                [{"id": point.id, "payload": point.payload, "vector": point.vector} for point in points]
            )
            return

        self.client.upsert(collection_name=config.collection_name, points=points, wait=True)

    def replace_file_chunks(self, filename: str, chunks: List[Any], embeddings: List[List[float]]):
        self.delete_by_file(filename)
        self.upsert_chunks(filename, chunks, embeddings)

    def search(self, query_vector: List[float], limit: int = 5, expand_to_parent: bool = False) -> List[Dict[str, Any]]:
        if not getattr(self, "available", True):
            hits = [{"payload": point["payload"], "score": 0.5} for point in getattr(self, "_memory_points", [])[:limit]]
            return self._dedupe_expanded_hits(hits) if expand_to_parent else hits
        results = self.client.query_points(
            collection_name=config.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        hits = [{"payload": hit.payload, "score": hit.score} for hit in results.points]
        if expand_to_parent:
            hits = self._dedupe_expanded_hits(hits)
        return hits

    def _scroll_chunks(
        self,
        limit_per_page: int = 256,
        with_payload: Optional[List[str]] = None,
        with_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        if not getattr(self, "available", True):
            rows = [{"id": point["id"], "payload": point["payload"]} for point in getattr(self, "_memory_points", [])]
            if with_vectors:
                for row, point in zip(rows, getattr(self, "_memory_points", [])):
                    row["vector"] = point.get("vector")
            return rows
        points: List[Dict[str, Any]] = []
        offset = None
        while True:
            scroll_result = self.client.scroll(
                collection_name=config.collection_name,
                with_payload=with_payload or True,
                with_vectors=with_vectors,
                limit=limit_per_page,
                offset=offset,
            )
            page_points, offset = scroll_result
            for point in page_points:
                row = {"id": point.id, "payload": point.payload}
                if with_vectors:
                    row["vector"] = point.vector
                points.append(row)
            if offset is None:
                break
        return points

    def get_file_chunks(self, filename: str, include_vectors: bool = False) -> List[Dict[str, Any]]:
        rows = self._scroll_chunks(
            with_payload=[
                "source_file",
                "chunk_text",
                "chunk_index",
                "parent_id",
                "children_ids",
                "content_hash",
                "version_id",
                "heading_path",
                "heading_level",
                "section_type",
                "chunk_hash",
            ],
            with_vectors=include_vectors,
        )
        results = [row for row in rows if str((row.get("payload") or {}).get("source_file", "")) == filename]
        results.sort(key=lambda item: int((item.get("payload") or {}).get("chunk_index", 0)))
        return results

    @staticmethod
    def _tokenize_query(query_text: str) -> List[str]:
        if not query_text:
            return []
        lowered = query_text.lower()
        coarse = re.split(r"[\s,，。！？；:：、\n\t]+", lowered)
        tokens = [token.strip() for token in coarse if len(token.strip()) >= 2]
        return tokens or [lowered]

    def keyword_search(self, query_text: str, limit: int = 8) -> List[Dict[str, Any]]:
        tokens = self._tokenize_query(query_text)
        if not tokens:
            return []
        if self.supports_text_index:
            return self._keyword_search_with_index(tokens, limit)
        return self._keyword_search_by_scrolling(tokens, limit)

    def _keyword_search_with_index(self, tokens: List[str], limit: int) -> List[Dict[str, Any]]:
        unique_tokens = list(dict.fromkeys(tokens))[:8]
        per_token_limit = max(limit * 4, 20)
        ranked: Dict[str, Dict[str, Any]] = {}

        for token in unique_tokens:
            page_points, _ = self.client.scroll(
                collection_name=config.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="chunk_text",
                            match=models.MatchText(text=token),
                        )
                    ]
                ),
                with_payload=[
                    "source_file",
                    "chunk_text",
                    "chunk_index",
                    "parent_id",
                    "parent_text",
                    "children_ids",
                ],
                limit=per_token_limit,
            )
            for point in page_points:
                payload = point.payload or {}
                key = self._payload_key(payload, fallback=point.id)
                entry = ranked.setdefault(
                    key,
                    {"id": point.id, "payload": payload, "matched_tokens": set()},
                )
                entry["matched_tokens"].add(token)

        results: List[Dict[str, Any]] = []
        token_count = max(len(unique_tokens), 1)
        for entry in ranked.values():
            results.append(
                {
                    "id": entry["id"],
                    "payload": entry["payload"],
                    "score": len(entry["matched_tokens"]) / token_count,
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]

    def _keyword_search_by_scrolling(self, tokens: List[str], limit: int) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        for point in self._scroll_chunks(
            with_payload=["source_file", "chunk_text", "chunk_index", "parent_id", "parent_text", "children_ids"]
        ):
            payload = point.get("payload", {}) or {}
            chunk_text = str(payload.get("chunk_text", "")).lower()
            if not chunk_text:
                continue
            matched = sum(1 for token in tokens if token in chunk_text)
            if matched <= 0:
                continue
            keyword_score = matched / max(len(tokens), 1)
            ranked.append({"id": point.get("id"), "payload": payload, "score": keyword_score})
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:limit]

    def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        limit: int = 8,
        alpha: float = 0.7,
        expand_to_parent: bool = False,
    ) -> List[Dict[str, Any]]:
        alpha = min(1.0, max(0.0, alpha))
        expanded_limit = max(limit * 3, 20)

        vector_hits = self.search(query_vector, limit=expanded_limit, expand_to_parent=False)
        keyword_hits = self.keyword_search(query_text, limit=expanded_limit)

        merged: Dict[str, Dict[str, Any]] = {}

        for hit in vector_hits:
            payload = hit.get("payload", {}) or {}
            key = self._payload_key(payload)
            vector_score_raw = float(hit.get("score", 0.0))
            vector_score_norm = (vector_score_raw + 1.0) / 2.0
            merged[key] = {
                "payload": payload,
                "vector_score": vector_score_norm,
                "keyword_score": 0.0,
            }

        for hit in keyword_hits:
            payload = hit.get("payload", {}) or {}
            key = self._payload_key(payload, fallback=hit.get("id"))
            entry = merged.setdefault(
                key,
                {"payload": payload, "vector_score": 0.0, "keyword_score": 0.0},
            )
            entry["keyword_score"] = max(entry["keyword_score"], float(hit.get("score", 0.0)))

        ranked: List[Dict[str, Any]] = []
        for entry in merged.values():
            blended = alpha * entry["vector_score"] + (1.0 - alpha) * entry["keyword_score"]
            ranked.append(
                {
                    "payload": entry["payload"],
                    "score": blended,
                    "vector_score": entry["vector_score"],
                    "keyword_score": entry["keyword_score"],
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        ranked = ranked[:limit]
        if expand_to_parent:
            ranked = self._dedupe_expanded_hits(ranked)
        return ranked

    def delete_by_file(self, filename: str):
        if not getattr(self, "available", True):
            self._memory_points = [
                point for point in getattr(self, "_memory_points", []) if point["payload"].get("source_file") != filename
            ]
            return
        self.client.delete(
            collection_name=config.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_file",
                            match=models.MatchValue(value=filename),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def get_all_files(self) -> List[str]:
        if not getattr(self, "available", True):
            return sorted(
                {
                    str(point["payload"].get("source_file"))
                    for point in getattr(self, "_memory_points", [])
                    if point["payload"].get("source_file")
                }
            )
        try:
            response = self.client.facet(
                collection_name=config.collection_name,
                key="source_file",
                limit=10_000,
                exact=True,
            )
            return sorted(
                str(hit.value)
                for hit in response.hits
                if hit.value and int(getattr(hit, "count", 0) or 0) > 0
            )
        except Exception:
            pass

        files = set()
        offset = None
        while True:
            scroll_result = self.client.scroll(
                collection_name=config.collection_name,
                with_payload=["source_file"],
                limit=100,
                offset=offset,
            )
            for point in scroll_result[0]:
                files.add(point.payload["source_file"])
            offset = scroll_result[1]
            if offset is None:
                break
        return sorted(files)

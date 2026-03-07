import re
import uuid
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http import models

from ..config import config


class VectorStore:
    def __init__(self):
        self.supports_text_index = False
        self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            collection_info = self.client.get_collection(config.COLLECTION_NAME)
            current_dim = collection_info.config.params.vectors.size
            if current_dim != config.VECTOR_DIMENSION:
                print(f"⚠️ Vector dimension mismatch: {current_dim} vs {config.VECTOR_DIMENSION}. Recreating collection...")
                self.client.delete_collection(config.COLLECTION_NAME)
                self._create_collection()
                return
        except Exception:
            self._create_collection()
            return

        self._ensure_payload_indexes()

    def _create_collection(self):
        self.client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=config.VECTOR_DIMENSION,
                distance=models.Distance.COSINE,
            ),
        )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self):
        self.client.create_payload_index(
            collection_name=config.COLLECTION_NAME,
            field_name="source_file",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        try:
            self.client.create_payload_index(
                collection_name=config.COLLECTION_NAME,
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
        chunk_index = payload.get("chunk_index")
        if source_file is not None and chunk_index is not None:
            return f"{source_file}::{chunk_index}"
        return str(fallback)

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

        self.client.upsert(
            collection_name=config.COLLECTION_NAME,
            points=points,
            wait=True,
        )

    def replace_file_chunks(self, filename: str, chunks: List[Any], embeddings: List[List[float]]):
        self.delete_by_file(filename)
        self.upsert_chunks(filename, chunks, embeddings)

    def search(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        results = self.client.query_points(
            collection_name=config.COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return [{"payload": hit.payload, "score": hit.score} for hit in results.points]

    def _scroll_chunks(self, limit_per_page: int = 256) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        offset = None
        while True:
            scroll_result = self.client.scroll(
                collection_name=config.COLLECTION_NAME,
                with_payload=["source_file", "chunk_text", "chunk_index"],
                limit=limit_per_page,
                offset=offset,
            )
            page_points, offset = scroll_result
            for point in page_points:
                points.append({"id": point.id, "payload": point.payload})
            if offset is None:
                break
        return points

    @staticmethod
    def _tokenize_query(query_text: str) -> List[str]:
        if not query_text:
            return []
        lowered = query_text.lower()
        coarse = re.split(r"[\s,，。！？；:：、\n\t]+", lowered)
        tokens = [token.strip() for token in coarse if len(token.strip()) >= 2]
        if tokens:
            return tokens
        return [lowered]

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
                collection_name=config.COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="chunk_text",
                            match=models.MatchText(text=token),
                        )
                    ]
                ),
                with_payload=["source_file", "chunk_text", "chunk_index"],
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
        for point in self._scroll_chunks():
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
    ) -> List[Dict[str, Any]]:
        alpha = min(1.0, max(0.0, alpha))
        expanded_limit = max(limit * 3, 20)

        vector_hits = self.search(query_vector, limit=expanded_limit)
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
        return ranked[:limit]

    def delete_by_file(self, filename: str):
        self.client.delete(
            collection_name=config.COLLECTION_NAME,
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
        try:
            response = self.client.facet(
                collection_name=config.COLLECTION_NAME,
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
                collection_name=config.COLLECTION_NAME,
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

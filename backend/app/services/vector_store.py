from qdrant_client import QdrantClient
from qdrant_client.http import models
from ..config import config
from typing import List, Dict, Any
import re

class VectorStore:
    def __init__(self):
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
        except Exception:
            # Collection likely doesn't exist
            self._create_collection()

    def _create_collection(self):
        self.client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=config.VECTOR_DIMENSION,
                distance=models.Distance.COSINE
            )
        )

    def upsert_chunks(self, filename: str, chunks: List[Any], embeddings: List[List[float]]):
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
            for k, v in metadata.items():
                if v is not None:
                    payload[k] = v

            points.append(models.PointStruct(
                id=abs(hash(f"{filename}_{i}")),
                vector=vector,
                payload=payload
            ))
        
        self.client.upsert(
            collection_name=config.COLLECTION_NAME,
            points=points
        )

    def search(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        results = self.client.query_points(
            collection_name=config.COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            with_payload=True
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
            for p in page_points:
                points.append(
                    {
                        "id": p.id,
                        "payload": p.payload,
                    }
                )
            if offset is None:
                break
        return points

    @staticmethod
    def _tokenize_query(query_text: str) -> List[str]:
        if not query_text:
            return []
        lowered = query_text.lower()
        coarse = re.split(r"[\s,，。！？；:：、\n\t]+", lowered)
        tokens = [t.strip() for t in coarse if len(t.strip()) >= 2]
        if tokens:
            return tokens
        return [lowered]

    def keyword_search(self, query_text: str, limit: int = 8) -> List[Dict[str, Any]]:
        tokens = self._tokenize_query(query_text)
        if not tokens:
            return []

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
            ranked.append(
                {
                    "id": point.get("id"),
                    "payload": payload,
                    "score": keyword_score,
                }
            )
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:limit]

    def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        limit: int = 8,
        alpha: float = 0.7,
    ) -> List[Dict[str, Any]]:
        # alpha: vector weight, (1-alpha): keyword weight
        alpha = min(1.0, max(0.0, alpha))
        expanded_limit = max(limit * 3, 20)

        vector_hits = self.search(query_vector, limit=expanded_limit)
        keyword_hits = self.keyword_search(query_text, limit=expanded_limit)

        merged: Dict[str, Dict[str, Any]] = {}

        for hit in vector_hits:
            payload = hit.get("payload", {}) or {}
            key = f"{payload.get('source_file')}::{payload.get('chunk_index')}"
            vector_score_raw = float(hit.get("score", 0.0))
            vector_score_norm = (vector_score_raw + 1.0) / 2.0  # cosine score in [-1,1]
            merged[key] = {
                "payload": payload,
                "vector_score": vector_score_norm,
                "keyword_score": 0.0,
            }

        for hit in keyword_hits:
            payload = hit.get("payload", {}) or {}
            key = f"{payload.get('source_file')}::{payload.get('chunk_index')}"
            entry = merged.setdefault(
                key,
                {
                    "payload": payload,
                    "vector_score": 0.0,
                    "keyword_score": 0.0,
                },
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
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:limit]

    def delete_by_file(self, filename: str):
        self.client.delete(
            collection_name=config.COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_file",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                )
            )
        )

    def get_all_files(self) -> List[str]:
        # Using scroll to find all unique source_file in payloads
        # In a real app, you might want a separate SQL DB for this, but for MVP we query Qdrant
        # This is a bit expensive but works for small/medium datasets
        files = set()
        offset = None
        while True:
            scroll_result = self.client.scroll(
                collection_name=config.COLLECTION_NAME,
                with_payload=["source_file"],
                limit=100,
                offset=offset
            )
            for point in scroll_result[0]:
                files.add(point.payload["source_file"])
            offset = scroll_result[1]
            if offset is None:
                break
        return list(files)

from qdrant_client import QdrantClient
from qdrant_client.http import models
from ..config import config
from typing import List, Dict, Any

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

    def upsert_chunks(self, filename: str, chunks: List[str], embeddings: List[List[float]]):
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            points.append(models.PointStruct(
                id=abs(hash(f"{filename}_{i}")),
                vector=vector,
                payload={
                    "source_file": filename,
                    "chunk_text": chunk,
                    "chunk_index": i
                }
            ))
        
        self.client.upsert(
            collection_name=config.COLLECTION_NAME,
            points=points
        )

    def search(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        results = self.client.search(
            collection_name=config.COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        return [{"payload": hit.payload, "score": hit.score} for hit in results]

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

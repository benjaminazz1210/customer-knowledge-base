import time

from qdrant_client.http import models

from ..config import config
from ..services.embedding_service import EmbeddingService
from ..services.text_chunker import TextChunker
from ..services.vector_store import VectorStore


def main():
    vector_store = VectorStore()
    embedding_service = EmbeddingService()
    chunker = TextChunker()
    target_collection = f"{config.collection_name}_reindex_{int(time.time())}"

    if not vector_store.client.collection_exists(target_collection):
        vector_store._create_collection(collection_name=target_collection)

    files = vector_store.get_all_files()
    for filename in files:
        file_chunks = vector_store.get_file_chunks(filename)
        full_text = "\n\n".join((row.get("payload") or {}).get("chunk_text", "") for row in file_chunks)
        rebuilt_chunks = chunker.chunk_document([], full_text)
        normalized_chunks = []
        for chunk in rebuilt_chunks:
            if isinstance(chunk, dict):
                normalized_chunks.append(chunk)
            else:
                normalized_chunks.append({"chunk_text": str(chunk), "metadata": {}})
        embeddings = embedding_service.get_embeddings([chunk["chunk_text"] for chunk in normalized_chunks])
        points = []
        for idx, (chunk, vector) in enumerate(zip(normalized_chunks, embeddings)):
            payload = {"source_file": filename, "chunk_text": chunk["chunk_text"], "chunk_index": idx}
            payload.update(chunk.get("metadata", {}))
            points.append(
                models.PointStruct(
                    id=VectorStore._build_point_id(filename, idx),
                    vector=vector,
                    payload=payload,
                )
            )
        if points:
            vector_store.client.upsert(collection_name=target_collection, points=points, wait=True)

    try:
        vector_store.client.update_collection_aliases(
            change_aliases_operations=[
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=config.collection_name)),
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=target_collection,
                        alias_name=config.collection_name,
                    )
                ),
            ]
        )
    except Exception:
        pass

    print(f"Reindexed {len(files)} files into collection {target_collection}")


if __name__ == "__main__":
    main()

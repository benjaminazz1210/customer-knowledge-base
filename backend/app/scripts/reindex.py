import argparse
import time
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client.http import models

from ..config import config
from ..services.document_parser import StructuredSection
from ..services.embedding_service import EmbeddingService
from ..services.text_chunker import TextChunker
from ..services.vector_store import VectorStore


METADATA_FIELDS = (
    "parent_id",
    "children_ids",
    "content_hash",
    "version_id",
    "chunk_hash",
    "delta_key",
    "heading_path",
    "heading_level",
    "section_type",
    "page",
    "slide",
    "chunk_role",
    "parent_text",
    "parent_chunk_text",
)


def _payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return dict((row or {}).get("payload", {}) or {})


def _normalize_heading_path(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if value:
        return [str(value)]
    return ["文档正文"]


def _preserve_rows(file_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in sorted(file_rows, key=lambda item: int(_payload(item).get("chunk_index", 0))):
        payload = _payload(row)
        metadata = {key: payload[key] for key in METADATA_FIELDS if key in payload}
        normalized.append(
            {
                "chunk_text": str(payload.get("chunk_text", "")),
                "metadata": metadata,
            }
        )
    return normalized


def _select_source_rows(file_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    parent_rows = [row for row in file_rows if _payload(row).get("chunk_role") == "parent"]
    if parent_rows:
        return parent_rows

    non_child_rows = [row for row in file_rows if _payload(row).get("chunk_role") != "child"]
    if non_child_rows:
        return non_child_rows
    return file_rows


def _rows_to_sections(file_rows: List[Dict[str, Any]]) -> Tuple[List[StructuredSection], str]:
    sections: List[StructuredSection] = []
    ordered_rows = sorted(_select_source_rows(file_rows), key=lambda item: int(_payload(item).get("chunk_index", 0)))
    for row in ordered_rows:
        payload = _payload(row)
        text = str(payload.get("chunk_text", "")).strip()
        if not text:
            continue
        sections.append(
            StructuredSection(
                heading_path=_normalize_heading_path(payload.get("heading_path")),
                heading_level=int(payload.get("heading_level", 1) or 1),
                content=text,
                section_type=str(payload.get("section_type", "text") or "text"),
                page=payload.get("page"),
                slide=payload.get("slide"),
            )
        )
    full_text = "\n\n".join(section.content for section in sections)
    return sections, full_text


def rebuild_chunks(file_rows: List[Dict[str, Any]], chunker: Optional[TextChunker] = None) -> List[Dict[str, Any]]:
    chunker = chunker or TextChunker()
    strategy = (config.chunking_strategy or "fixed").strip().lower()
    preserved_rows = _preserve_rows(file_rows)
    if strategy == "fixed":
        return preserved_rows

    sections, full_text = _rows_to_sections(file_rows)
    if not sections and not full_text:
        return preserved_rows

    rebuilt = chunker.chunk_document(sections, full_text)
    if not rebuilt:
        return preserved_rows

    shared_metadata: Dict[str, Any] = {}
    for row in file_rows:
        payload = _payload(row)
        for field in ("content_hash", "version_id"):
            if payload.get(field) is not None:
                shared_metadata[field] = payload.get(field)

    normalized: List[Dict[str, Any]] = []
    for chunk in rebuilt:
        if isinstance(chunk, dict):
            metadata = dict(chunk.get("metadata", {}))
            for key, value in shared_metadata.items():
                metadata.setdefault(key, value)
            normalized.append(
                {
                    "chunk_text": str(chunk.get("chunk_text", "")),
                    "metadata": metadata,
                    **({"child_id": chunk.get("child_id")} if chunk.get("child_id") else {}),
                }
            )
            continue
        normalized.append({"chunk_text": str(chunk), "metadata": dict(shared_metadata)})
    return normalized


def _collection_exists(client: Any, collection_name: str) -> bool:
    if hasattr(client, "collection_exists"):
        return bool(client.collection_exists(collection_name))
    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False


def _alias_names(client: Any) -> List[str]:
    try:
        aliases_response = client.get_aliases()
    except Exception:
        return []
    aliases = getattr(aliases_response, "aliases", aliases_response) or []
    return [str(getattr(alias, "alias_name", "")) for alias in aliases if getattr(alias, "alias_name", None)]


def swap_alias(client: Any, alias_name: str, target_collection: str) -> List[Any]:
    operations: List[Any] = []
    if alias_name in _alias_names(client):
        operations.append(
            models.DeleteAliasOperation(
                delete_alias=models.DeleteAlias(alias_name=alias_name),
            )
        )
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name=target_collection,
                alias_name=alias_name,
            )
        )
    )
    client.update_collection_aliases(change_aliases_operations=operations)
    return operations


def run_reindex(
    *,
    source_collection: Optional[str] = None,
    target_collection: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    vector_store = VectorStore()
    embedding_service = EmbeddingService()
    source_name = source_collection or config.collection_name
    target_name = target_collection or f"{source_name}_reindex_{int(time.time())}"

    if source_name != config.collection_name:
        raise ValueError("reindex currently supports the active collection alias only")

    if not _collection_exists(vector_store.client, target_name):
        vector_store._create_collection(collection_name=target_name)

    files = vector_store.get_all_files()
    points_written = 0
    file_summaries: List[Dict[str, Any]] = []

    for filename in files:
        file_rows = vector_store.get_file_chunks(filename)
        rebuilt_chunks = rebuild_chunks(file_rows, chunker=TextChunker())
        embeddings = embedding_service.get_embeddings([chunk["chunk_text"] for chunk in rebuilt_chunks])
        points = []
        for idx, (chunk, vector) in enumerate(zip(rebuilt_chunks, embeddings)):
            payload = {"source_file": filename, "chunk_text": chunk["chunk_text"], "chunk_index": idx}
            payload.update(chunk.get("metadata", {}))
            if chunk.get("child_id"):
                payload["child_id"] = chunk["child_id"]
            points.append(
                models.PointStruct(
                    id=VectorStore._point_id_for_payload(filename, payload),
                    vector=vector,
                    payload=payload,
                )
            )
        if points and not dry_run:
            vector_store.client.upsert(collection_name=target_name, points=points, wait=True)
        points_written += len(points)
        file_summaries.append(
            {
                "filename": filename,
                "chunks": len(rebuilt_chunks),
                "points_written": len(points),
            }
        )

    alias_swapped = False
    if not dry_run:
        swap_alias(vector_store.client, config.collection_name, target_name)
        alias_swapped = True

    return {
        "source_collection": source_name,
        "target_collection": target_name,
        "files_count": len(files),
        "points_written": points_written,
        "alias_swapped": alias_swapped,
        "dry_run": dry_run,
        "files": file_summaries,
    }


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Reindex the active Qdrant collection with the configured chunking strategy.")
    parser.add_argument("--source-collection", default="", help="Override source collection/alias.")
    parser.add_argument("--target-collection", default="", help="Override target collection name.")
    parser.add_argument("--dry-run", action="store_true", help="Build chunks and embeddings without alias swap or upsert.")
    args = parser.parse_args(argv)

    result = run_reindex(
        source_collection=args.source_collection or None,
        target_collection=args.target_collection or None,
        dry_run=args.dry_run,
    )
    print(
        "Reindexed {files_count} files into {target_collection} (points={points_written}, alias_swapped={alias_swapped}, dry_run={dry_run})".format(
            **result
        )
    )
    return result


if __name__ == "__main__":
    main()

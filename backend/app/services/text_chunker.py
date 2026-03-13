from typing import Any, Dict, List

from ..config import config
from .document_parser import StructuredSection


class TextChunker:
    @staticmethod
    def chunk(text: str, chunk_size: int = config.chunk_size, overlap: int = config.chunk_overlap) -> List[str]:
        if not text:
            return []

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
            if end >= len(text):
                break
        return chunks

    @staticmethod
    def _section_prefix(section: StructuredSection) -> str:
        heading_path = section.heading_path or ["文档正文"]
        heading_line = " > ".join([h for h in heading_path if h]) or "文档正文"
        return f"[结构路径] {heading_line}\n"

    @classmethod
    def chunk_structured(
        cls,
        sections: List[StructuredSection],
        chunk_size: int = config.chunk_size,
        overlap: int = config.chunk_overlap,
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        if not sections:
            return chunks

        for section_idx, section in enumerate(sections):
            content = (section.content or "").strip()
            if not content:
                continue

            prefix = cls._section_prefix(section)
            start = 0
            while start < len(content):
                end = start + chunk_size
                piece = content[start:end]
                if not piece.strip():
                    break

                chunks.append(
                    {
                        "chunk_text": f"{prefix}{piece}",
                        "metadata": {
                            "heading_path": section.heading_path or ["文档正文"],
                            "heading_level": int(section.heading_level or 1),
                            "section_type": section.section_type,
                            "section_index": section_idx,
                            "page": section.page,
                            "slide": section.slide,
                            "chunk_role": "standard",
                        },
                    }
                )
                start = end - overlap
                if end >= len(content):
                    break
        return chunks

    @classmethod
    def semantic_chunk_structured(
        cls,
        sections: List[StructuredSection],
        min_size: int = config.semantic_chunk_min_size,
        max_size: int = config.semantic_chunk_max_size,
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        if not sections:
            return chunks

        for section_idx, section in enumerate(sections):
            paragraphs = [line.strip() for line in (section.content or "").split("\n\n") if line.strip()]
            if not paragraphs:
                paragraphs = [section.content.strip()] if section.content.strip() else []

            buffer: List[str] = []
            current_size = 0
            for paragraph in paragraphs:
                para_len = len(paragraph)
                if buffer and current_size + para_len > max_size:
                    piece = "\n\n".join(buffer).strip()
                    if piece:
                        chunks.append(
                            {
                                "chunk_text": f"{cls._section_prefix(section)}{piece}",
                                "metadata": {
                                    "heading_path": section.heading_path or ["文档正文"],
                                    "heading_level": int(section.heading_level or 1),
                                    "section_type": section.section_type,
                                    "section_index": section_idx,
                                    "page": section.page,
                                    "slide": section.slide,
                                    "chunk_role": "semantic",
                                },
                            }
                        )
                    buffer = []
                    current_size = 0

                buffer.append(paragraph)
                current_size += para_len
                if current_size >= min_size:
                    continue

            if buffer:
                piece = "\n\n".join(buffer).strip()
                if piece:
                    chunks.append(
                        {
                            "chunk_text": f"{cls._section_prefix(section)}{piece}",
                            "metadata": {
                                "heading_path": section.heading_path or ["文档正文"],
                                "heading_level": int(section.heading_level or 1),
                                "section_type": section.section_type,
                                "section_index": section_idx,
                                "page": section.page,
                                "slide": section.slide,
                                "chunk_role": "semantic",
                            },
                        }
                    )
        return chunks

    @classmethod
    def parent_child_chunk_structured(
        cls,
        sections: List[StructuredSection],
        parent_size: int = config.parent_chunk_size,
        child_size: int = config.child_chunk_size,
        overlap: int = config.chunk_overlap,
    ) -> List[Dict[str, Any]]:
        semantic_chunks = cls.semantic_chunk_structured(
            sections,
            min_size=min(child_size, config.semantic_chunk_min_size),
            max_size=max(parent_size, child_size),
        )
        if not semantic_chunks:
            return []

        parent_chunks: List[Dict[str, Any]] = []
        parent_idx = 0
        current_parent: List[Dict[str, Any]] = []
        current_size = 0
        for chunk in semantic_chunks:
            text = chunk.get("chunk_text", "")
            if current_parent and current_size + len(text) > parent_size:
                parent_chunks.append({"parent_id": f"parent-{parent_idx}", "children": current_parent[:]})
                current_parent = []
                current_size = 0
                parent_idx += 1
            current_parent.append(chunk)
            current_size += len(text)
        if current_parent:
            parent_chunks.append({"parent_id": f"parent-{parent_idx}", "children": current_parent[:]})

        combined_chunks: List[Dict[str, Any]] = []
        for parent in parent_chunks:
            parent_id = parent["parent_id"]
            parent_text = "\n\n".join(child["chunk_text"] for child in parent["children"])
            children_ids: List[str] = []
            parent_metadata = {
                "chunk_role": "parent",
                "parent_id": parent_id,
                "children_ids": [],
                "parent_text": parent_text,
                "parent_chunk_text": parent_text,
            }
            for child_idx, child in enumerate(parent["children"]):
                base_text = child.get("chunk_text", "")
                child_texts = cls.chunk(base_text, chunk_size=child_size, overlap=overlap) or [base_text]
                for nested_idx, child_text in enumerate(child_texts):
                    child_id = f"{parent_id}-child-{child_idx}-{nested_idx}"
                    children_ids.append(child_id)
                    metadata = dict(child.get("metadata", {}))
                    metadata.update(
                        {
                            "chunk_role": "child",
                            "parent_id": parent_id,
                            "parent_text": parent_text,
                            "parent_chunk_text": parent_text,
                        }
                    )
                    combined_chunks.append({"chunk_text": child_text, "metadata": metadata, "child_id": child_id})

            parent_metadata["children_ids"] = children_ids
            combined_chunks.append({"chunk_text": parent_text, "metadata": parent_metadata})
            for chunk in combined_chunks:
                if chunk.get("metadata", {}).get("parent_id") == parent_id:
                    chunk["metadata"]["children_ids"] = children_ids
        return combined_chunks

    @classmethod
    def parent_child_chunks(cls, sections: List[StructuredSection]) -> List[Dict[str, Any]]:
        return cls.parent_child_chunk_structured(sections)

    @classmethod
    def chunk_document(cls, sections: List[StructuredSection], full_text: str = "") -> List[Any]:
        strategy = (config.chunking_strategy or "fixed").strip().lower()
        if strategy == "semantic":
            return cls.semantic_chunk_structured(sections)
        if strategy == "parent_child":
            return cls.parent_child_chunk_structured(sections)
        if sections:
            return cls.chunk_structured(sections)
        return cls.chunk(full_text)

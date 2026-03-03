from typing import List
from ..config import config
from .document_parser import StructuredSection

class TextChunker:
    @staticmethod
    def chunk(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> List[str]:
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
    def chunk_structured(
        sections: List[StructuredSection],
        chunk_size: int = config.CHUNK_SIZE,
        overlap: int = config.CHUNK_OVERLAP,
    ) -> List[dict]:
        chunks: List[dict] = []
        if not sections:
            return chunks

        for section_idx, section in enumerate(sections):
            content = (section.content or "").strip()
            if not content:
                continue

            heading_path = section.heading_path or ["文档正文"]
            heading_line = " > ".join([h for h in heading_path if h]) or "文档正文"
            prefix = f"[结构路径] {heading_line}\n"

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
                            "heading_path": heading_path,
                            "heading_level": int(section.heading_level or 1),
                            "section_type": section.section_type,
                            "section_index": section_idx,
                            "page": section.page,
                            "slide": section.slide,
                        },
                    }
                )
                start = end - overlap
                if end >= len(content):
                    break
        return chunks

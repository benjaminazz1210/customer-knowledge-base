import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.document_parser import DocumentParser
from app.services.text_chunker import TextChunker
from app.services.vector_store import VectorStore


class DocumentParserTests(unittest.TestCase):
    def test_parse_txt_md_and_unsupported_extension(self):
        self.assertEqual(DocumentParser.parse(b"Hello TXT", "test.txt"), "Hello TXT")
        self.assertEqual(DocumentParser.parse(b"# Hello MD", "test.md"), "# Hello MD")

        with self.assertRaisesRegex(ValueError, "Unsupported file format: .exe"):
            DocumentParser.parse(b"exe content", "test.exe")


class TextChunkerTests(unittest.TestCase):
    def test_chunk_overlap_and_edge_cases(self):
        self.assertEqual(TextChunker.chunk("123456789", chunk_size=5, overlap=2), ["12345", "45678", "789"])
        self.assertEqual(TextChunker.chunk("", chunk_size=5, overlap=2), [])
        self.assertEqual(TextChunker.chunk("12", chunk_size=5, overlap=2), ["12"])


class VectorStoreTests(unittest.TestCase):
    def make_store(self):
        store = VectorStore.__new__(VectorStore)
        store.client = MagicMock()
        store.supports_text_index = True
        return store

    def test_point_id_is_deterministic(self):
        point_id = VectorStore._build_point_id("demo.txt", 3)
        self.assertEqual(point_id, VectorStore._build_point_id("demo.txt", 3))
        self.assertNotEqual(point_id, VectorStore._build_point_id("demo.txt", 4))

    def test_upsert_chunks_uses_stable_ids_and_waits(self):
        store = self.make_store()
        store.upsert_chunks(
            "demo.txt",
            [{"chunk_text": "hello", "metadata": {"section_type": "body"}}],
            [[0.1, 0.2]],
        )

        kwargs = store.client.upsert.call_args.kwargs
        self.assertTrue(kwargs["wait"])
        self.assertEqual(kwargs["points"][0].id, VectorStore._build_point_id("demo.txt", 0))
        self.assertEqual(kwargs["points"][0].payload["source_file"], "demo.txt")

    def test_replace_file_chunks_deletes_before_upsert(self):
        store = self.make_store()
        store.delete_by_file = MagicMock()
        store.upsert_chunks = MagicMock()

        chunks = [{"chunk_text": "hello"}]
        embeddings = [[0.1, 0.2]]
        store.replace_file_chunks("demo.txt", chunks, embeddings)

        store.delete_by_file.assert_called_once_with("demo.txt")
        store.upsert_chunks.assert_called_once_with("demo.txt", chunks, embeddings)

    def test_keyword_search_uses_qdrant_text_filter(self):
        store = self.make_store()
        store._scroll_chunks = MagicMock(side_effect=AssertionError("scroll fallback should not be used"))
        record = SimpleNamespace(
            id="point-1",
            payload={"source_file": "demo.txt", "chunk_text": "alpha beta", "chunk_index": 0},
        )
        store.client.scroll.side_effect = [([record], None), ([record], None)]

        hits = store.keyword_search("alpha beta", limit=5)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["score"], 1.0)
        self.assertEqual(store.client.scroll.call_count, 2)
        first_filter = store.client.scroll.call_args_list[0].kwargs["scroll_filter"]
        self.assertEqual(first_filter.must[0].key, "chunk_text")
        self.assertEqual(first_filter.must[0].match.text, "alpha")

    def test_get_all_files_prefers_facet_index(self):
        store = self.make_store()
        store.client.facet.return_value = SimpleNamespace(
            hits=[
                SimpleNamespace(value="b.txt", count=1),
                SimpleNamespace(value="a.txt", count=2),
                SimpleNamespace(value="ghost.txt", count=0),
            ]
        )

        files = store.get_all_files()

        self.assertEqual(files, ["a.txt", "b.txt"])
        store.client.facet.assert_called_once()
        store.client.scroll.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)

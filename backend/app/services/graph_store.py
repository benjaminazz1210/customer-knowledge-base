import logging
import re
from collections import defaultdict
from typing import Dict, List, Set

from ..config import config

logger = logging.getLogger("nexusai.graph")


class GraphStore:
    def __init__(self):
        self._graph = defaultdict(set)
        self._available = True
        self._driver = None
        if not config.graph_rag_enabled:
            return
        try:
            from neo4j import GraphDatabase  # type: ignore
        except Exception as exc:
            logger.warning("neo4j driver unavailable, graph store uses in-memory fallback: %s", exc)
            return

        try:
            if config.neo4j_password:
                self._driver = GraphDatabase.driver(
                    config.neo4j_uri,
                    auth=(config.neo4j_user, config.neo4j_password),
                )
        except Exception as exc:
            logger.warning("Failed to connect Neo4j, graph store uses in-memory fallback: %s", exc)
            self._driver = None

    @staticmethod
    def extract_entities(text: str) -> List[str]:
        tokens = re.findall(r"[A-Z][a-zA-Z0-9_-]+|[\u4e00-\u9fff]{2,8}", text or "")
        stop_words = {"文档正文", "问题", "回答", "关于", "什么", "如何", "哪些"}
        entities = []
        for token in tokens:
            if token in stop_words:
                continue
            if token not in entities:
                entities.append(token)
        return entities[:20]

    def upsert_document(self, filename: str, chunks: List[Dict[str, str]]) -> None:
        if not config.graph_rag_enabled:
            return

        entities: Set[str] = set()
        for chunk in chunks:
            chunk_entities = self.extract_entities(chunk.get("chunk_text", ""))
            entities.update(chunk_entities)
            for left in chunk_entities:
                for right in chunk_entities:
                    if left == right:
                        continue
                    self._graph[left].add(right)
            if filename:
                for entity in chunk_entities:
                    self._graph[filename].add(entity)
                    self._graph[entity].add(filename)

        if self._driver is None:
            return

        try:
            with self._driver.session() as session:
                for entity in entities:
                    session.run("MERGE (:Entity {name: $name})", name=entity)
                for src, targets in self._graph.items():
                    for dst in targets:
                        session.run(
                            "MERGE (a:Entity {name: $src}) "
                            "MERGE (b:Entity {name: $dst}) "
                            "MERGE (a)-[:RELATED_TO]->(b)",
                            src=src,
                            dst=dst,
                        )
        except Exception as exc:
            logger.warning("Neo4j write failed, keeping in-memory graph only: %s", exc)

    def ingest_document(self, filename: str, chunks: List[Dict[str, str]]) -> None:
        self.upsert_document(filename, chunks)

    def query_context(self, query: str) -> List[Dict[str, object]]:
        if not config.graph_rag_enabled:
            return []

        entities = self.extract_entities(query)
        related = []
        seen = set()
        for entity in entities:
            neighbors = sorted(self._graph.get(entity, set()))
            for neighbor in neighbors[: config.graph_max_hops * 5]:
                key = (entity, neighbor)
                if key in seen:
                    continue
                seen.add(key)
                related.append(
                    {
                        "payload": {
                            "source_file": "__graph__",
                            "chunk_text": "%s 关联到 %s" % (entity, neighbor),
                            "chunk_index": len(related),
                            "graph_entity": entity,
                        },
                        "score": 0.4,
                        "vector_score": 0.0,
                        "keyword_score": 0.0,
                    }
                )
        return related

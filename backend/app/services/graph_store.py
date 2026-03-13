import json
import logging
import re
from collections import defaultdict
from typing import Dict, List, Set

from ..config import config
from .llm_utils import complete_text, is_mock_backend

logger = logging.getLogger("nexusai.graph")


class GraphStore:
    def __init__(self):
        self._graph = defaultdict(set)
        self._document_entities: Dict[str, Set[str]] = {}
        self._driver = None
        self._spacy_nlp = None
        self._spacy_unavailable = False
        self._mock_llm = is_mock_backend()
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
    def _dedupe_entities(tokens: List[str]) -> List[str]:
        stop_words = {"文档正文", "问题", "回答", "关于", "什么", "如何", "哪些"}
        entities: List[str] = []
        for token in tokens:
            normalized = str(token or "").strip()
            if not normalized or normalized in stop_words:
                continue
            if normalized not in entities:
                entities.append(normalized)
        return entities[:20]

    @classmethod
    def _extract_entities_regex(cls, text: str) -> List[str]:
        tokens = re.findall(r"[A-Z][a-zA-Z0-9_-]+|[\u4e00-\u9fff]{2,8}", text or "")
        return cls._dedupe_entities(tokens)

    def _extract_entities_llm(self, text: str) -> List[str]:
        prompt = (
            "Extract the most important named entities from the text, including people, organizations, products, "
            "and concepts. Return a JSON array of strings only.\n\n"
            f"Text:\n{text[:3000]}"
        )
        raw = complete_text(
            [{"role": "user", "content": prompt}],
            model=config.graph_entity_extraction_model or config.llm_model,
        ).strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return self._dedupe_entities([str(item) for item in parsed])
        except Exception:
            pass
        fallback = [item.strip(" -\n\t\"'") for item in re.split(r"[,;\n]+", raw) if item.strip()]
        return self._dedupe_entities(fallback)

    def _extract_entities_spacy(self, text: str) -> List[str]:
        if self._spacy_unavailable:
            return []
        if self._spacy_nlp is None:
            try:
                import spacy  # type: ignore

                self._spacy_nlp = spacy.load(config.graph_spacy_model)
            except Exception as exc:
                logger.info("spaCy entity extraction unavailable, falling back to regex: %s", exc)
                self._spacy_unavailable = True
                self._spacy_nlp = None
                return []
        doc = self._spacy_nlp(text)
        allowed_labels = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "NORP", "FAC"}
        return self._dedupe_entities([ent.text for ent in doc.ents if ent.label_ in allowed_labels])

    def extract_entities(self, text: str) -> List[str]:
        backend = (config.graph_entity_extraction_backend or "auto").strip().lower()
        if backend in ("auto", "llm") and config.graph_entity_extraction_model:
            entities = self._extract_entities_llm(text)
            if entities or backend == "llm":
                return entities
        if backend in ("auto", "spacy"):
            entities = self._extract_entities_spacy(text)
            if entities or backend == "spacy":
                return entities
        return self._extract_entities_regex(text)

    def extract_knowledge(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        backend = (config.graph_entity_extraction_backend or "auto").strip().lower()
        if backend == "llm":
            prompt = (
                "Extract entities and relationships from the text. "
                "Return a JSON object with keys 'entities' and 'relations'. "
                "'entities' should be a list of objects with 'name' and 'type'. "
                "'relations' should be a list of objects with 'source', 'target', and 'type'.\n\n"
                f"Text:\n{text[:3000]}"
            )
            raw = complete_text(
                [{"role": "user", "content": prompt}],
                model=config.graph_entity_extraction_model or config.llm_model,
            ).strip()
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    entities = parsed.get("entities", []) if isinstance(parsed.get("entities"), list) else []
                    relations = parsed.get("relations", []) if isinstance(parsed.get("relations"), list) else []
                    return {
                        "entities": [
                            {"name": str(item.get("name", "")).strip(), "type": str(item.get("type", "entity")).strip()}
                            for item in entities
                            if isinstance(item, dict) and str(item.get("name", "")).strip()
                        ],
                        "relations": [
                            {
                                "source": str(item.get("source", "")).strip(),
                                "target": str(item.get("target", "")).strip(),
                                "type": str(item.get("type", "RELATED_TO")).strip(),
                            }
                            for item in relations
                            if isinstance(item, dict)
                            and str(item.get("source", "")).strip()
                            and str(item.get("target", "")).strip()
                        ],
                    }
            except Exception:
                logger.info("Failed to parse LLM graph extraction output, falling back to heuristic extraction.")

        entities = [{"name": entity, "type": "entity"} for entity in self.extract_entities(text)]
        relations = []
        for index, entity in enumerate(entities):
            for other in entities[index + 1 :]:
                relations.append({"source": entity["name"], "target": other["name"], "type": "RELATED_TO"})
        return {"entities": entities, "relations": relations}

    def upsert_document(self, filename: str, chunks: List[Dict[str, str]]) -> None:
        if not config.graph_rag_enabled:
            return

        entities: Set[str] = set()
        for chunk in chunks:
            knowledge = self.extract_knowledge(chunk.get("chunk_text", ""))
            chunk_entities = [item["name"] for item in knowledge.get("entities", []) if item.get("name")]
            entities.update(chunk_entities)
            for left in chunk_entities:
                for right in chunk_entities:
                    if left == right:
                        continue
                    self._graph[left].add(right)
            for relation in knowledge.get("relations", []):
                source = str(relation.get("source", "")).strip()
                target = str(relation.get("target", "")).strip()
                if source and target and source != target:
                    self._graph[source].add(target)
            if filename:
                for entity in chunk_entities:
                    self._graph[filename].add(entity)
                    self._graph[entity].add(filename)
        if filename:
            self._document_entities[filename] = set(entities)

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

    def replace_document(self, filename: str, chunks: List[Dict[str, str]]) -> None:
        previous_entities = self._document_entities.pop(filename, set())
        if previous_entities:
            self._graph.pop(filename, None)
            for entity in previous_entities:
                self._graph[entity].discard(filename)
        self.upsert_document(filename, chunks)

    def query_context(self, query: str) -> List[Dict[str, object]]:
        if not config.graph_rag_enabled:
            return []

        entities = self.extract_entities(query)
        related = []
        seen = set()
        for entity in entities:
            frontier = [(entity, 0)]
            visited = {entity}
            while frontier:
                current, depth = frontier.pop(0)
                if depth >= max(int(config.graph_max_hops), 1):
                    continue
                neighbors = sorted(self._graph.get(current, set()))
                for neighbor in neighbors:
                    key = (current, neighbor)
                    if key not in seen:
                        seen.add(key)
                        related.append(
                            {
                                "payload": {
                                    "source_file": "__graph__",
                                    "chunk_text": "%s 关联到 %s" % (current, neighbor),
                                    "chunk_index": len(related),
                                    "graph_entity": entity,
                                },
                                "score": 0.4,
                                "vector_score": 0.0,
                                "keyword_score": 0.0,
                            }
                        )
                    if neighbor not in visited:
                        visited.add(neighbor)
                        frontier.append((neighbor, depth + 1))
        return related

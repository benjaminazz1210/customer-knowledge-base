import json
import logging
import re
from collections import defaultdict, deque
from typing import Any, DefaultDict, Dict, List, Set, Tuple

from ..config import config
from .llm_utils import complete_text, is_mock_backend

logger = logging.getLogger("nexusai.graph")


class GraphStore:
    def __init__(self):
        self._graph: DefaultDict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self._file_entities: DefaultDict[str, Set[str]] = defaultdict(set)
        self._file_edges: DefaultDict[str, Set[Tuple[str, str]]] = defaultdict(set)
        self._node_sources: DefaultDict[str, Set[Tuple[str, str]]] = defaultdict(set)
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

    def _extractor_backend(self) -> str:
        backend = str(config.graph_entity_extraction_backend or "auto").strip().lower()
        if backend != "auto":
            return backend
        if config.graph_entity_extraction_model:
            return "llm"
        if self._load_spacy() is not None:
            return "spacy"
        return "regex"

    def _load_spacy(self):
        if self._spacy_nlp is not None or self._spacy_unavailable:
            return self._spacy_nlp
        try:
            import spacy  # type: ignore

            self._spacy_nlp = spacy.load(config.graph_spacy_model)
        except Exception as exc:
            logger.info("spaCy entity extraction unavailable, falling back to regex: %s", exc)
            self._spacy_unavailable = True
            self._spacy_nlp = None
        return self._spacy_nlp

    @staticmethod
    def _dedupe_entities(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        stop_words = {"文档正文", "问题", "回答", "关于", "什么", "如何", "哪些"}
        deduped: List[Dict[str, str]] = []
        seen = set()
        for entity in entities:
            name = str(entity.get("name", "")).strip()
            entity_type = str(entity.get("type", "entity") or "entity").strip().lower()
            if not name or name in seen or name in stop_words:
                continue
            seen.add(name)
            deduped.append({"name": name, "type": entity_type})
        return deduped[:20]

    @classmethod
    def _extract_entities_regex(cls, text: str) -> List[Dict[str, str]]:
        tokens = re.findall(r"[A-Z][a-zA-Z0-9_-]+|[\u4e00-\u9fff]{2,12}", text or "")
        entities = []
        for token in tokens:
            entity_type = "concept" if re.search(r"[\u4e00-\u9fff]", token) else "named_entity"
            entities.append({"name": token, "type": entity_type})
        return cls._dedupe_entities(entities)

    def _extract_entities_llm(self, text: str) -> List[Dict[str, str]]:
        if self._mock_llm and not config.graph_entity_extraction_model:
            return []
        prompt = (
            "Extract the most important named entities from the text, including people, organizations, products, "
            "and concepts. Return a JSON array where each item has name and type.\n\n"
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
                entities = []
                for item in parsed:
                    if isinstance(item, dict):
                        entities.append(
                            {
                                "name": str(item.get("name", "")).strip(),
                                "type": str(item.get("type", "entity")).strip(),
                            }
                        )
                    else:
                        entities.append({"name": str(item).strip(), "type": "entity"})
                return self._dedupe_entities(entities)
        except Exception:
            pass
        fallback = [item.strip(" -\n\t\"'") for item in re.split(r"[,;\n]+", raw) if item.strip()]
        return self._dedupe_entities([{"name": item, "type": "entity"} for item in fallback])

    def _extract_entities_spacy(self, text: str) -> List[Dict[str, str]]:
        nlp = self._load_spacy()
        if nlp is None:
            return []
        doc = nlp(text)
        allowed_labels = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "NORP", "FAC"}
        return self._dedupe_entities(
            [{"name": ent.text, "type": ent.label_} for ent in doc.ents if ent.label_ in allowed_labels]
        )

    @staticmethod
    def _cooccurrence_relations(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        relations = []
        names = [entity["name"] for entity in entities if entity.get("name")]
        for left_idx, left in enumerate(names):
            for right in names[left_idx + 1 :]:
                if left == right:
                    continue
                relations.append({"source": left, "target": right, "type": "MENTIONED_WITH"})
        return relations

    def extract_entities(self, text: str) -> List[str]:
        return [entity["name"] for entity in self.extract_knowledge(text)["entities"]]

    def extract_knowledge(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        backend = self._extractor_backend()
        entities: List[Dict[str, str]]
        relations: List[Dict[str, str]] = []

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
                    parsed_entities = parsed.get("entities", []) if isinstance(parsed.get("entities"), list) else []
                    parsed_relations = parsed.get("relations", []) if isinstance(parsed.get("relations"), list) else []
                    entities = self._dedupe_entities(
                        [
                            {
                                "name": str(item.get("name", "")).strip(),
                                "type": str(item.get("type", "entity")).strip(),
                            }
                            for item in parsed_entities
                            if isinstance(item, dict)
                        ]
                    )
                    relations = [
                        {
                            "source": str(item.get("source", "")).strip(),
                            "target": str(item.get("target", "")).strip(),
                            "type": str(item.get("type", "RELATED_TO")).strip(),
                        }
                        for item in parsed_relations
                        if isinstance(item, dict)
                        and str(item.get("source", "")).strip()
                        and str(item.get("target", "")).strip()
                    ]
                    if entities or relations:
                        return {"entities": entities, "relations": relations or self._cooccurrence_relations(entities)}
            except Exception:
                logger.info("Failed to parse LLM graph extraction output, falling back to heuristic extraction.")
            entities = self._extract_entities_llm(text) or self._extract_entities_spacy(text) or self._extract_entities_regex(text)
        elif backend == "spacy":
            entities = self._extract_entities_spacy(text) or self._extract_entities_regex(text)
        else:
            entities = self._extract_entities_regex(text)

        return {"entities": entities, "relations": self._cooccurrence_relations(entities)}

    def _record_entity(self, filename: str, entity_name: str, entity_type: str):
        self._file_entities[filename].add(entity_name)
        self._node_sources[entity_name].add((filename, entity_type))

    def _record_edge(
        self,
        *,
        filename: str,
        source: str,
        target: str,
        relation_type: str,
        chunk_hash: str,
        version_id: str,
    ):
        edge = self._graph[source].setdefault(target, {"relation_types": set(), "sources": set()})
        edge["relation_types"].add(relation_type)
        edge["sources"].add((filename, chunk_hash, version_id))
        self._file_edges[filename].add((source, target))

    def delete_document(self, filename: str) -> None:
        for entity_name in list(self._file_entities.pop(filename, set())):
            remaining = {item for item in self._node_sources.get(entity_name, set()) if item[0] != filename}
            if remaining:
                self._node_sources[entity_name] = remaining
            else:
                self._node_sources.pop(entity_name, None)

        for source, target in list(self._file_edges.pop(filename, set())):
            edge = self._graph.get(source, {}).get(target)
            if edge is None:
                continue
            edge["sources"] = {item for item in edge.get("sources", set()) if item[0] != filename}
            if not edge["sources"]:
                self._graph[source].pop(target, None)
                if not self._graph[source]:
                    self._graph.pop(source, None)

        if self._driver is not None:
            try:
                with self._driver.session() as session:
                    session.run("MATCH ()-[r:RELATED_TO {source_file: $source_file}]-() DELETE r", source_file=filename)
            except Exception as exc:
                logger.warning("Neo4j delete failed for %s: %s", filename, exc)

    def upsert_document(self, filename: str, chunks: List[Dict[str, Any]]) -> None:
        if not config.graph_rag_enabled:
            return

        for chunk in chunks:
            chunk_text = str(chunk.get("chunk_text", ""))
            metadata = chunk.get("metadata", {}) or {}
            chunk_hash = str(metadata.get("chunk_hash", ""))
            version_id = str(metadata.get("version_id", ""))
            knowledge = self.extract_knowledge(chunk_text)

            for entity in knowledge["entities"]:
                self._record_entity(filename, entity["name"], entity["type"])

            for relation in knowledge["relations"]:
                source = str(relation.get("source", "")).strip()
                target = str(relation.get("target", "")).strip()
                relation_type = str(relation.get("type", "RELATED_TO")).strip() or "RELATED_TO"
                if not source or not target or source == target:
                    continue
                self._record_edge(
                    filename=filename,
                    source=source,
                    target=target,
                    relation_type=relation_type,
                    chunk_hash=chunk_hash,
                    version_id=version_id,
                )
                self._record_edge(
                    filename=filename,
                    source=target,
                    target=source,
                    relation_type=relation_type,
                    chunk_hash=chunk_hash,
                    version_id=version_id,
                )

            if self._driver is None:
                continue
            try:
                with self._driver.session() as session:
                    for entity in knowledge["entities"]:
                        session.run(
                            "MERGE (e:Entity {name: $name}) SET e.entity_type = $entity_type",
                            name=entity["name"],
                            entity_type=entity["type"],
                        )
                    for relation in knowledge["relations"]:
                        source = str(relation.get("source", "")).strip()
                        target = str(relation.get("target", "")).strip()
                        relation_type = str(relation.get("type", "RELATED_TO")).strip() or "RELATED_TO"
                        if not source or not target or source == target:
                            continue
                        session.run(
                            "MATCH (a:Entity {name: $source}), (b:Entity {name: $target}) "
                            "MERGE (a)-[:RELATED_TO {source_file: $source_file, chunk_hash: $chunk_hash, version_id: $version_id, relation_type: $relation_type}]->(b)",
                            source=source,
                            target=target,
                            source_file=filename,
                            chunk_hash=chunk_hash,
                            version_id=version_id,
                            relation_type=relation_type,
                        )
            except Exception as exc:
                logger.warning("Neo4j write failed, keeping in-memory graph only: %s", exc)

    def ingest_document(self, filename: str, chunks: List[Dict[str, Any]]) -> None:
        self.replace_document(filename, chunks)

    def replace_document(self, filename: str, chunks: List[Dict[str, Any]]) -> None:
        self.delete_document(filename)
        self.upsert_document(filename, chunks)

    def _query_neo4j(self, entities: List[str]) -> List[Dict[str, object]]:
        if self._driver is None or not entities:
            return []

        related: List[Dict[str, object]] = []
        max_hops = max(int(config.graph_max_hops), 1)
        seen = set()
        query = (
            f"MATCH p=(a:Entity {{name: $entity}})-[rels:RELATED_TO*1..{max_hops}]-(b:Entity) "
            "RETURN nodes(p) AS nodes, relationships(p) AS rels LIMIT $limit"
        )
        try:
            with self._driver.session() as session:
                for entity in entities:
                    rows = session.run(query, entity=entity, limit=max_hops * 5)
                    for row in rows:
                        nodes = row.get("nodes", [])
                        rels = row.get("rels", [])
                        for idx in range(min(len(rels), len(nodes) - 1)):
                            source = str(nodes[idx].get("name"))
                            target = str(nodes[idx + 1].get("name"))
                            relation_type = str(rels[idx].get("relation_type", "RELATED_TO"))
                            edge_key = (source, target, relation_type)
                            if edge_key in seen:
                                continue
                            seen.add(edge_key)
                            related.append(
                                {
                                    "payload": {
                                        "source_file": "__graph__",
                                        "chunk_text": f"{source} 通过 {relation_type} 关联到 {target}",
                                        "chunk_index": len(related),
                                        "graph_entity": entity,
                                    },
                                    "score": 0.4,
                                    "vector_score": 0.0,
                                    "keyword_score": 0.0,
                                }
                            )
        except Exception as exc:
            logger.warning("Neo4j query failed, falling back to in-memory graph: %s", exc)
            return []
        return related

    def _query_in_memory(self, entities: List[str]) -> List[Dict[str, object]]:
        related: List[Dict[str, object]] = []
        seen = set()
        max_hops = max(int(config.graph_max_hops), 1)
        for entity in entities:
            frontier = deque([(entity, 0)])
            visited = {entity}
            while frontier:
                current, depth = frontier.popleft()
                if depth >= max_hops:
                    continue
                for neighbor, edge in sorted(self._graph.get(current, {}).items()):
                    relation_type = sorted(edge.get("relation_types", {"RELATED_TO"}))[0]
                    edge_key = (current, neighbor, relation_type)
                    if edge_key not in seen:
                        seen.add(edge_key)
                        related.append(
                            {
                                "payload": {
                                    "source_file": "__graph__",
                                    "chunk_text": f"{current} 通过 {relation_type} 关联到 {neighbor}",
                                    "chunk_index": len(related),
                                    "graph_entity": entity,
                                },
                                "score": round(max(0.25, 0.5 - (0.05 * depth)), 4),
                                "vector_score": 0.0,
                                "keyword_score": 0.0,
                            }
                        )
                    if neighbor not in visited:
                        visited.add(neighbor)
                        frontier.append((neighbor, depth + 1))
        return related

    def query_context(self, query: str) -> List[Dict[str, object]]:
        if not config.graph_rag_enabled:
            return []

        entities = self.extract_entities(query)
        related = self._query_neo4j(entities)
        if related:
            return related
        return self._query_in_memory(entities)

"""
NEXUS Semantic Memory & Knowledge Graph
=========================================
Structured knowledge storage using a lightweight knowledge graph.
Inspired by Zep/Graphiti temporal KG and MAGMA multi-graph architecture.

Features:
- Entity-Relationship-Entity triplet storage
- Temporal awareness (when facts were learned/updated)
- Community detection for related concepts
- Incremental updates without batch recomputation
- JSON-based persistence (no external DB required)
- Hybrid retrieval: graph traversal + keyword search
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class Entity:
    """A node in the knowledge graph."""
    id: str
    name: str
    entity_type: str = "concept"  # concept, person, tool, model, skill, location, event
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    confidence: float = 1.0

    def touch(self):
        self.access_count += 1
        self.updated_at = time.time()


@dataclass
class Relationship:
    """An edge in the knowledge graph."""
    source_id: str
    target_id: str
    relation_type: str  # is_a, has, uses, depends_on, related_to, caused_by, etc.
    weight: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    valid_from: float = field(default_factory=time.time)
    valid_until: float | None = None  # None = still valid
    confidence: float = 1.0


@dataclass
class Community:
    """A cluster of related entities."""
    id: str
    name: str
    entity_ids: list[str] = field(default_factory=list)
    summary: str = ""
    created_at: float = field(default_factory=time.time)


class KnowledgeGraph:
    """
    A lightweight, persistent knowledge graph for semantic memory.

    Stores entities and relationships as triplets:
    (Entity) --[Relationship]--> (Entity)

    Supports:
    - Entity CRUD with type-based indexing
    - Relationship management with temporal validity
    - Multi-hop graph traversal
    - Community detection (simple connected components)
    - JSON persistence
    """

    def __init__(self, storage_path: str = "./data/nexus_memory/knowledge_graph.json"):
        self.storage_path = Path(storage_path)
        self.entities: dict[str, Entity] = {}
        self.relationships: list[Relationship] = []
        self.communities: dict[str, Community] = {}
        self._entity_index: dict[str, list[str]] = {}  # type -> [entity_ids]
        self._load()

    def _load(self):
        """Load knowledge graph from persistent storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                for e_data in data.get("entities", []):
                    entity = Entity(**e_data)
                    self.entities[entity.id] = entity
                    self._index_entity(entity)
                for r_data in data.get("relationships", []):
                    self.relationships.append(Relationship(**r_data))
                for c_data in data.get("communities", []):
                    community = Community(**c_data)
                    self.communities[community.id] = community
            except (json.JSONDecodeError, TypeError):
                pass

    def save(self):
        """Persist knowledge graph to storage."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entities": [asdict(e) for e in self.entities.values()],
            "relationships": [asdict(r) for r in self.relationships],
            "communities": [asdict(c) for c in self.communities.values()],
            "metadata": {
                "entity_count": len(self.entities),
                "relationship_count": len(self.relationships),
                "saved_at": time.time(),
            },
        }
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _index_entity(self, entity: Entity):
        """Add entity to type index."""
        if entity.entity_type not in self._entity_index:
            self._entity_index[entity.entity_type] = []
        if entity.id not in self._entity_index[entity.entity_type]:
            self._entity_index[entity.entity_type].append(entity.id)

    def add_entity(self, name: str, entity_type: str = "concept", properties: dict | None = None, confidence: float = 1.0) -> Entity:
        """Add or update an entity."""
        entity_id = f"{entity_type}:{name.lower().replace(' ', '_')}"

        if entity_id in self.entities:
            # Update existing
            existing = self.entities[entity_id]
            if properties:
                existing.properties.update(properties)
            existing.updated_at = time.time()
            existing.confidence = max(existing.confidence, confidence)
            return existing

        entity = Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            properties=properties or {},
            confidence=confidence,
        )
        self.entities[entity_id] = entity
        self._index_entity(entity)
        return entity

    def add_relationship(
        self,
        source: str | Entity,
        target: str | Entity,
        relation_type: str,
        weight: float = 1.0,
        properties: dict | None = None,
        confidence: float = 1.0,
    ) -> Relationship:
        """Add a relationship between entities."""
        source_id = source.id if isinstance(source, Entity) else source
        target_id = target.id if isinstance(target, Entity) else target

        rel = Relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            properties=properties or {},
            confidence=confidence,
        )
        self.relationships.append(rel)
        return rel

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        entity = self.entities.get(entity_id)
        if entity:
            entity.touch()
        return entity

    def find_entities(self, name_query: str, entity_type: str | None = None) -> list[Entity]:
        """Find entities by name (partial match)."""
        query_lower = name_query.lower()
        results = []
        for entity in self.entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            if query_lower in entity.name.lower():
                results.append(entity)
        return results

    def get_by_type(self, entity_type: str) -> list[Entity]:
        """Get all entities of a given type."""
        ids = self._entity_index.get(entity_type, [])
        return [self.entities[eid] for eid in ids if eid in self.entities]

    def get_relationships(self, entity_id: str, direction: str = "both") -> list[Relationship]:
        """Get all relationships for an entity."""
        results = []
        for rel in self.relationships:
            if rel.valid_until and rel.valid_until < time.time():
                continue  # Skip expired relationships
            if direction in ("both", "outgoing") and rel.source_id == entity_id:
                results.append(rel)
            if direction in ("both", "incoming") and rel.target_id == entity_id:
                results.append(rel)
        return results

    def traverse(self, start_id: str, max_hops: int = 2) -> dict[str, Any]:
        """
        Multi-hop graph traversal from a starting entity.
        Returns connected subgraph within max_hops.
        """
        visited: set[str] = set()
        frontier = {start_id}
        subgraph: dict[str, list[dict]] = {}

        for hop in range(max_hops):
            next_frontier: set[str] = set()
            for entity_id in frontier:
                if entity_id in visited:
                    continue
                visited.add(entity_id)

                rels = self.get_relationships(entity_id)
                subgraph[entity_id] = []
                for rel in rels:
                    neighbor_id = rel.target_id if rel.source_id == entity_id else rel.source_id
                    subgraph[entity_id].append({
                        "neighbor": neighbor_id,
                        "relation": rel.relation_type,
                        "weight": rel.weight,
                    })
                    if neighbor_id not in visited:
                        next_frontier.add(neighbor_id)

            frontier = next_frontier

        return {
            "root": start_id,
            "hops": max_hops,
            "nodes_visited": len(visited),
            "graph": subgraph,
        }

    def find_path(self, source_id: str, target_id: str, max_depth: int = 5) -> list[str] | None:
        """Find shortest path between two entities using BFS."""
        if source_id == target_id:
            return [source_id]

        visited = {source_id}
        queue: list[list[str]] = [[source_id]]

        while queue:
            path = queue.pop(0)
            if len(path) > max_depth:
                return None

            current = path[-1]
            for rel in self.get_relationships(current):
                neighbor = rel.target_id if rel.source_id == current else rel.source_id
                if neighbor == target_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        return None

    def detect_communities(self) -> list[Community]:
        """Simple connected component detection for community finding."""
        visited: set[str] = set()
        communities: list[Community] = []

        for entity_id in self.entities:
            if entity_id in visited:
                continue

            # BFS to find connected component
            component: list[str] = []
            queue = [entity_id]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)

                for rel in self.get_relationships(current):
                    neighbor = rel.target_id if rel.source_id == current else rel.source_id
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(component) > 1:
                community = Community(
                    id=f"community_{len(communities)}",
                    name=f"Community {len(communities)}",
                    entity_ids=component,
                )
                communities.append(community)

        self.communities = {c.id: c for c in communities}
        return communities

    def query(self, question: str) -> list[dict]:
        """
        Natural language query against the knowledge graph.
        Returns relevant entities and their relationships.
        """
        # Find matching entities
        entities = self.find_entities(question)
        results = []

        for entity in entities[:5]:
            rels = self.get_relationships(entity.id)
            connected = []
            for rel in rels[:10]:
                neighbor_id = rel.target_id if rel.source_id == entity.id else rel.source_id
                neighbor = self.entities.get(neighbor_id)
                if neighbor:
                    connected.append({
                        "entity": neighbor.name,
                        "relation": rel.relation_type,
                        "confidence": rel.confidence,
                    })

            results.append({
                "entity": entity.name,
                "type": entity.entity_type,
                "properties": entity.properties,
                "connections": connected,
                "confidence": entity.confidence,
            })

        return results

    @property
    def stats(self) -> dict:
        return {
            "entities": len(self.entities),
            "relationships": len(self.relationships),
            "communities": len(self.communities),
            "entity_types": list(self._entity_index.keys()),
        }


class SemanticMemory:
    """
    High-level semantic memory interface wrapping the KnowledgeGraph.
    Provides concept-level operations for agent use.
    """

    def __init__(self, storage_path: str = "./data/nexus_memory/knowledge_graph.json"):
        self.kg = KnowledgeGraph(storage_path=storage_path)

    def learn_fact(self, subject: str, predicate: str, obj: str, confidence: float = 1.0):
        """Store a fact as a triplet: subject --predicate--> object."""
        source = self.kg.add_entity(subject)
        target = self.kg.add_entity(obj)
        self.kg.add_relationship(source, target, predicate, confidence=confidence)

    def learn_concept(self, name: str, concept_type: str = "concept", properties: dict | None = None):
        """Store a concept with its properties."""
        self.kg.add_entity(name, entity_type=concept_type, properties=properties)

    def recall(self, query: str) -> list[dict]:
        """Recall knowledge relevant to a query."""
        return self.kg.query(query)

    def explore(self, concept: str, depth: int = 2) -> dict:
        """Explore knowledge around a concept."""
        entities = self.kg.find_entities(concept)
        if entities:
            return self.kg.traverse(entities[0].id, max_hops=depth)
        return {"root": concept, "graph": {}}

    def save(self):
        """Persist semantic memory."""
        self.kg.save()

    @property
    def stats(self) -> dict:
        return self.kg.stats

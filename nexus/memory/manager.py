"""
NEXUS Memory Manager
=====================
Coordinates the triple-layer memory system:
- Working Memory (short-term, per-session)
- Episodic Memory (long-term experiences)
- Semantic Memory (knowledge graph)

Handles memory consolidation, retrieval routing, and persistence.
"""

from __future__ import annotations

from typing import Any

from nexus.core.config import NexusConfig
from nexus.memory.working import WorkingMemory
from nexus.memory.episodic import EpisodicMemory
from nexus.memory.semantic import SemanticMemory


class MemoryManager:
    """
    Unified memory manager coordinating all three memory layers.

    Memory Flow:
    1. New information enters Working Memory
    2. Important experiences are consolidated to Episodic Memory
    3. Patterns and facts are extracted to Semantic Memory (Knowledge Graph)
    4. Retrieval searches across all layers and merges results
    """

    def __init__(self, config: NexusConfig):
        self.config = config
        self._working_memories: dict[str, WorkingMemory] = {}
        self.episodic = EpisodicMemory(
            storage_path=f"{config.memory_dir}/episodes.jsonl",
            max_episodes=config.max_episodic_memory,
        )
        self.semantic = SemanticMemory(
            storage_path=f"{config.memory_dir}/knowledge_graph.json",
        )

    def get_working_memory(self, session_id: str = "default") -> WorkingMemory:
        """Get or create working memory for a session."""
        if session_id not in self._working_memories:
            self._working_memories[session_id] = WorkingMemory(
                session_id=session_id,
                max_items=self.config.max_working_memory,
            )
        return self._working_memories[session_id]

    def attach(self, agent):
        """Attach memory system to an agent."""
        agent._memory_manager = self

    async def store_episode(self, episode_data: dict[str, Any]):
        """Store an experience episode and extract knowledge."""
        episode = await self.episodic.store(episode_data)

        # Extract entities and relationships for semantic memory
        if episode.task:
            self.semantic.learn_concept(
                episode.task[:100],
                concept_type="task",
                properties={
                    "success": episode.success,
                    "task_type": episode.task_type,
                },
            )

        if episode.model_used:
            self.semantic.learn_fact(
                episode.task[:50], "used_model", episode.model_used
            )

        for tool in episode.tools_used:
            self.semantic.learn_fact(
                episode.task[:50], "used_tool", tool
            )

    async def search_relevant(self, query: str, limit: int = 5) -> list[dict]:
        """Search across all memory layers for relevant context."""
        results = []

        # Search episodic memory
        episodes = self.episodic.search(query, limit=limit)
        for ep in episodes:
            results.append({
                "role": "system",
                "content": f"[Past experience] Task: {ep.task[:100]} → Result: {ep.result[:200]}",
                "source": "episodic",
            })

        # Search semantic memory
        knowledge = self.semantic.recall(query)
        for k in knowledge[:limit]:
            connections = ", ".join(
                f"{c['relation']} {c['entity']}" for c in k.get("connections", [])[:3]
            )
            results.append({
                "role": "system",
                "content": f"[Knowledge] {k['entity']} ({k['type']}): {connections}",
                "source": "semantic",
            })

        return results[:limit]

    def get_experience_patterns(self, task_type: str = "") -> dict:
        """Get successful patterns from episodic memory."""
        return self.episodic.get_successful_patterns(task_type)

    def save_all(self):
        """Persist all memory layers."""
        self.semantic.save()

    @property
    def total_entries(self) -> int:
        working_count = sum(wm.count for wm in self._working_memories.values())
        return working_count + self.episodic.count + self.semantic.stats["entities"]

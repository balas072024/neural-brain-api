"""
NEXUS Episodic Memory
======================
Long-term storage of experiences and episodes.
Inspired by MAGMA and Memoria architectures.

Each episode captures:
- What happened (task, result, context)
- When it happened (timestamps)
- How it went (success, quality, duration)
- What was learned (insights, patterns)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class Episode:
    """A single experience episode."""
    id: str = ""
    task: str = ""
    task_type: str = ""
    result: str = ""
    success: bool = True
    quality_score: float = 0.0
    duration_ms: float = 0
    agents_used: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    model_used: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    insights: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)


class EpisodicMemory:
    """
    Long-term episodic memory for storing and retrieving experiences.

    Features:
    - Persistent storage (JSON-based)
    - Similarity-based retrieval (keyword matching)
    - Temporal queries (recent, time-range)
    - Pattern extraction across episodes
    - Experience-based learning signals
    """

    def __init__(self, storage_path: str = "./data/nexus_memory/episodes.jsonl", max_episodes: int = 10000):
        self.storage_path = Path(storage_path)
        self.max_episodes = max_episodes
        self._episodes: list[Episode] = []
        self._load()

    def _load(self):
        """Load episodes from persistent storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            self._episodes.append(Episode(**data))
            except (json.JSONDecodeError, TypeError):
                self._episodes = []

    def _save_episode(self, episode: Episode):
        """Append an episode to persistent storage."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "a") as f:
            f.write(json.dumps(asdict(episode)) + "\n")

    async def store(self, episode_data: dict[str, Any]) -> Episode:
        """Store a new episode."""
        episode = Episode(
            id=f"ep_{len(self._episodes)}_{int(time.time())}",
            **{k: v for k, v in episode_data.items() if hasattr(Episode, k)},
        )
        self._episodes.append(episode)
        self._save_episode(episode)

        # Evict old episodes if over limit
        if len(self._episodes) > self.max_episodes:
            self._episodes = self._episodes[-self.max_episodes:]

        return episode

    def search(self, query: str, limit: int = 10) -> list[Episode]:
        """Search episodes by keyword similarity."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []
        for ep in self._episodes:
            ep_text = f"{ep.task} {ep.result} {' '.join(ep.tags)}".lower()
            ep_words = set(ep_text.split())
            overlap = len(query_words.intersection(ep_words))
            if overlap > 0:
                scored.append((overlap, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:limit]]

    def get_recent(self, n: int = 10) -> list[Episode]:
        """Get the N most recent episodes."""
        return self._episodes[-n:]

    def get_by_task_type(self, task_type: str) -> list[Episode]:
        """Get episodes filtered by task type."""
        return [ep for ep in self._episodes if ep.task_type == task_type]

    def get_successful_patterns(self, task_type: str = "") -> dict[str, Any]:
        """Extract patterns from successful episodes."""
        relevant = self._episodes
        if task_type:
            relevant = [ep for ep in relevant if ep.task_type == task_type]

        successful = [ep for ep in relevant if ep.success]
        if not successful:
            return {"patterns": [], "success_rate": 0.0}

        # Extract common patterns
        model_freq: dict[str, int] = {}
        tool_freq: dict[str, int] = {}
        avg_duration = 0.0

        for ep in successful:
            if ep.model_used:
                model_freq[ep.model_used] = model_freq.get(ep.model_used, 0) + 1
            for tool in ep.tools_used:
                tool_freq[tool] = tool_freq.get(tool, 0) + 1
            avg_duration += ep.duration_ms

        avg_duration /= len(successful) if successful else 1

        return {
            "success_rate": len(successful) / len(relevant) if relevant else 0,
            "best_models": sorted(model_freq.items(), key=lambda x: x[1], reverse=True)[:3],
            "best_tools": sorted(tool_freq.items(), key=lambda x: x[1], reverse=True)[:5],
            "avg_duration_ms": avg_duration,
            "total_episodes": len(relevant),
        }

    def get_insights(self) -> list[str]:
        """Extract accumulated insights from all episodes."""
        all_insights = []
        for ep in self._episodes:
            all_insights.extend(ep.insights)
        return list(set(all_insights))

    @property
    def count(self) -> int:
        return len(self._episodes)

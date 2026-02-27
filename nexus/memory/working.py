"""
NEXUS Working Memory
=====================
Short-term memory for active conversations and tasks.
Equivalent to human working memory — limited capacity, fast access.
Implements a sliding window with priority-based eviction.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryItem:
    """A single item in working memory."""
    content: dict[str, Any]
    priority: float = 1.0
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    tags: list[str] = field(default_factory=list)

    def touch(self):
        """Update access timestamp and count."""
        self.accessed_at = time.time()
        self.access_count += 1


class WorkingMemory:
    """
    Short-term working memory for an active session.

    Features:
    - Fixed capacity with priority-based eviction
    - Recency-weighted retrieval
    - Tag-based filtering
    - Automatic summarization trigger when capacity is reached
    """

    def __init__(self, session_id: str = "default", max_items: int = 50):
        self.session_id = session_id
        self.max_items = max_items
        self._items: deque[MemoryItem] = deque(maxlen=max_items * 2)
        self._pinned: list[MemoryItem] = []  # Important items that won't be evicted

    def add(self, content: dict[str, Any], priority: float = 1.0, tags: list[str] | None = None):
        """Add an item to working memory."""
        item = MemoryItem(content=content, priority=priority, tags=tags or [])
        self._items.append(item)
        self._maybe_evict()

    def pin(self, content: dict[str, Any], tags: list[str] | None = None):
        """Pin an important item (won't be evicted)."""
        item = MemoryItem(content=content, priority=10.0, tags=tags or ["pinned"])
        self._pinned.append(item)

    def get_recent(self, n: int = 10) -> list[dict]:
        """Get the N most recent items."""
        items = list(self._items)[-n:]
        for item in items:
            item.touch()
        return [item.content for item in items]

    def get_by_tags(self, tags: list[str]) -> list[dict]:
        """Get items matching any of the given tags."""
        tag_set = set(tags)
        matching = [
            item for item in self._items
            if tag_set.intersection(item.tags)
        ]
        return [item.content for item in matching]

    def get_all(self) -> list[dict]:
        """Get all items including pinned."""
        pinned = [item.content for item in self._pinned]
        regular = [item.content for item in self._items]
        return pinned + regular

    def search(self, query: str) -> list[dict]:
        """Simple keyword search across working memory."""
        query_lower = query.lower()
        results = []
        for item in list(self._pinned) + list(self._items):
            content_str = str(item.content).lower()
            if query_lower in content_str:
                item.touch()
                results.append(item.content)
        return results

    def summarize_context(self) -> str:
        """Generate a text summary of current working memory for LLM context."""
        lines = []
        if self._pinned:
            lines.append("=== Pinned Context ===")
            for item in self._pinned:
                lines.append(str(item.content))

        recent = list(self._items)[-10:]
        if recent:
            lines.append("=== Recent Context ===")
            for item in recent:
                lines.append(str(item.content))

        return "\n".join(lines)

    def clear(self):
        """Clear all working memory (preserves pinned)."""
        self._items.clear()

    def _maybe_evict(self):
        """Evict lowest-priority items when over capacity."""
        while len(self._items) > self.max_items:
            # Find and remove lowest priority item
            min_item = min(self._items, key=lambda i: i.priority * (i.access_count + 1))
            self._items.remove(min_item)

    @property
    def count(self) -> int:
        return len(self._items) + len(self._pinned)

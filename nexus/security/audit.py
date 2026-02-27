"""
NEXUS Audit Trail
==================
Comprehensive audit logging for all framework operations.
Provides traceability and compliance for enterprise deployments.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from nexus.core.config import NexusConfig


@dataclass
class AuditEntry:
    """A single audit log entry."""
    event: str
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, error, critical
    source: str = "system"
    correlation_id: str = ""


class AuditTrail:
    """
    Audit trail for all NEXUS operations.

    Logs:
    - Agent creation and destruction
    - Task execution starts and completions
    - Tool invocations and results
    - Permission checks and approvals
    - Security events (blocked actions, violations)
    - Model API calls
    """

    def __init__(self, config: NexusConfig):
        self.config = config
        self._entries: list[AuditEntry] = []
        self._log_path = Path(config.audit_dir) / "audit.jsonl"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, severity: str = "info", **details):
        """Log an audit event."""
        if not self.config.audit_enabled:
            return

        entry = AuditEntry(
            event=event,
            severity=severity,
            details=details,
        )
        self._entries.append(entry)

        # Persist to file
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        except OSError:
            pass

    def get_recent(self, n: int = 50) -> list[dict]:
        """Get recent audit entries."""
        return [asdict(e) for e in self._entries[-n:]]

    def get_by_event(self, event: str) -> list[dict]:
        """Get audit entries by event type."""
        return [asdict(e) for e in self._entries if e.event == event]

    def get_errors(self) -> list[dict]:
        """Get all error and critical entries."""
        return [
            asdict(e) for e in self._entries
            if e.severity in ("error", "critical")
        ]

    def search(self, query: str) -> list[dict]:
        """Search audit entries."""
        query_lower = query.lower()
        return [
            asdict(e) for e in self._entries
            if query_lower in e.event.lower() or query_lower in str(e.details).lower()
        ]

    @property
    def count(self) -> int:
        return len(self._entries)

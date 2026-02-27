"""
NEXUS Security Sandbox
========================
Process-level isolation and resource control.
Addresses OpenClaw's security concerns with a zero-trust approach.
"""

from __future__ import annotations

from nexus.core.config import NexusConfig, SecurityLevel
from nexus.tools.executor import SandboxedExecutor


class SecuritySandbox:
    """
    Security sandbox providing:
    - Command allowlist/blocklist
    - File system access control
    - Network access control
    - Resource limits (CPU, memory, time)
    - Process isolation
    """

    def __init__(self, config: NexusConfig):
        self.config = config
        self.executor = SandboxedExecutor(config)
        self._allowed_paths: list[str] = ["."]
        self._blocked_paths: list[str] = ["/etc", "/root", "/var"]
        self._allowed_hosts: list[str] = ["*"]
        self._blocked_hosts: list[str] = []

    def is_path_allowed(self, path: str) -> bool:
        """Check if a file path is accessible."""
        if self.config.security_level == SecurityLevel.OPEN:
            return True

        for blocked in self._blocked_paths:
            if path.startswith(blocked):
                return False

        if self.config.security_level == SecurityLevel.PARANOID:
            return any(path.startswith(allowed) for allowed in self._allowed_paths)

        return True

    def is_host_allowed(self, host: str) -> bool:
        """Check if a network host is accessible."""
        if "*" in self._allowed_hosts:
            return host not in self._blocked_hosts
        return host in self._allowed_hosts

    def allow_path(self, path: str):
        """Add a path to the allowlist."""
        self._allowed_paths.append(path)

    def block_path(self, path: str):
        """Add a path to the blocklist."""
        self._blocked_paths.append(path)

    def allow_host(self, host: str):
        """Add a host to the allowlist."""
        self._allowed_hosts.append(host)

    def block_host(self, host: str):
        """Add a host to the blocklist."""
        self._blocked_hosts.append(host)

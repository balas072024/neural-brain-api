"""
NEXUS Permission System
========================
Fine-grained permission control for agents and tools.
Follows zero-trust: deny-all by default, allowlist explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Permission(str, Enum):
    FILE_READ = "file:read"
    FILE_WRITE = "file:write"
    FILE_DELETE = "file:delete"
    SHELL_EXEC = "shell:exec"
    NETWORK_ACCESS = "network:access"
    MODEL_CALL = "model:call"
    TOOL_USE = "tool:use"
    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"
    AGENT_CREATE = "agent:create"
    AGENT_DELEGATE = "agent:delegate"


@dataclass
class PermissionPolicy:
    """A permission policy for an agent or tool."""
    name: str
    grants: list[Permission] = field(default_factory=list)
    denials: list[Permission] = field(default_factory=list)
    resource_limits: dict[str, Any] = field(default_factory=dict)
    requires_approval: list[Permission] = field(default_factory=list)

    def is_allowed(self, permission: Permission) -> bool | None:
        """Check if a permission is allowed. Returns None if not specified."""
        if permission in self.denials:
            return False
        if permission in self.grants:
            return True
        return None

    def needs_approval(self, permission: Permission) -> bool:
        """Check if a permission requires human approval."""
        return permission in self.requires_approval


# Pre-defined security profiles
SECURITY_PROFILES = {
    "minimal": PermissionPolicy(
        name="minimal",
        grants=[Permission.FILE_READ, Permission.MODEL_CALL, Permission.MEMORY_READ],
        denials=[Permission.FILE_DELETE, Permission.SHELL_EXEC],
    ),
    "standard": PermissionPolicy(
        name="standard",
        grants=[
            Permission.FILE_READ, Permission.FILE_WRITE, Permission.MODEL_CALL,
            Permission.TOOL_USE, Permission.MEMORY_READ, Permission.MEMORY_WRITE,
            Permission.NETWORK_ACCESS,
        ],
        denials=[Permission.FILE_DELETE],
        requires_approval=[Permission.SHELL_EXEC, Permission.AGENT_CREATE],
    ),
    "full": PermissionPolicy(
        name="full",
        grants=list(Permission),
        requires_approval=[Permission.FILE_DELETE, Permission.SHELL_EXEC],
    ),
    "sandbox": PermissionPolicy(
        name="sandbox",
        grants=[Permission.FILE_READ, Permission.MODEL_CALL, Permission.MEMORY_READ],
        denials=[
            Permission.FILE_WRITE, Permission.FILE_DELETE,
            Permission.SHELL_EXEC, Permission.NETWORK_ACCESS,
        ],
    ),
}


class PermissionSystem:
    """
    Permission manager for all agents and tools.
    Zero-trust: deny by default, require explicit grants.
    """

    def __init__(self, config=None):
        self.config = config
        self._agent_policies: dict[str, PermissionPolicy] = {}
        self._default_policy = SECURITY_PROFILES["standard"]
        self._approval_handler = None

    def set_policy(self, agent_name: str, policy: PermissionPolicy | str):
        """Assign a permission policy to an agent."""
        if isinstance(policy, str):
            policy = SECURITY_PROFILES.get(policy, self._default_policy)
        self._agent_policies[agent_name] = policy

    def check(self, agent_name: str, permission: Permission) -> bool:
        """Check if an agent has a specific permission."""
        policy = self._agent_policies.get(agent_name, self._default_policy)
        result = policy.is_allowed(permission)
        if result is None:
            return False  # Deny by default
        return result

    def requires_approval(self, agent_name: str, permission: Permission) -> bool:
        """Check if a permission requires human approval."""
        policy = self._agent_policies.get(agent_name, self._default_policy)
        return policy.needs_approval(permission)

    def on_approval_needed(self, handler):
        """Register a handler for when human approval is needed."""
        self._approval_handler = handler

    def get_agent_permissions(self, agent_name: str) -> dict:
        """Get all permissions for an agent."""
        policy = self._agent_policies.get(agent_name, self._default_policy)
        return {
            "policy": policy.name,
            "grants": [p.value for p in policy.grants],
            "denials": [p.value for p in policy.denials],
            "requires_approval": [p.value for p in policy.requires_approval],
        }

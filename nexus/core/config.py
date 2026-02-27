"""
NEXUS Configuration System
===========================
Hierarchical configuration with environment variable overrides,
provider auto-discovery, and runtime reconfiguration.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RoutingStrategy(str, Enum):
    FASTEST = "fastest"
    CHEAPEST = "cheapest"
    BEST_QUALITY = "best_quality"
    CAPABILITY = "capability"
    ROUND_ROBIN = "round_robin"
    FALLBACK = "fallback"
    LOCAL_FIRST = "local_first"
    ADAPTIVE = "adaptive"  # Self-learning adaptive routing


class MemoryBackend(str, Enum):
    LOCAL = "local"       # In-memory + JSON persistence
    SQLITE = "sqlite"     # SQLite for persistence
    REDIS = "redis"       # Redis for distributed
    NEO4J = "neo4j"       # Neo4j for knowledge graph


class SecurityLevel(str, Enum):
    OPEN = "open"           # No restrictions (development)
    STANDARD = "standard"   # Confirm destructive actions
    STRICT = "strict"       # All actions require approval
    PARANOID = "paranoid"   # Sandboxed + audit + approval


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    name: str
    api_key: str = ""
    base_url: str = ""
    models: list[str] = field(default_factory=list)
    enabled: bool = True
    max_concurrent: int = 10
    timeout: float = 120.0
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    capabilities: list[str] = field(default_factory=list)
    priority: int = 0

    @classmethod
    def from_env(cls, name: str) -> ProviderConfig | None:
        """Auto-discover provider from environment variables."""
        key_var = f"{name.upper()}_API_KEY"
        url_var = f"{name.upper()}_BASE_URL"
        api_key = os.getenv(key_var, "")
        if not api_key and name not in ("ollama", "lm_studio", "local"):
            return None
        return cls(
            name=name,
            api_key=api_key,
            base_url=os.getenv(url_var, ""),
        )


@dataclass
class NexusConfig:
    """Master configuration for the NEXUS framework."""

    # Identity
    name: str = "NEXUS"
    version: str = "1.0.0"

    # Default model
    default_model: str = "claude-sonnet-4-6"
    fallback_model: str = "llama3.2:3b"

    # Routing
    routing_strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE
    max_retries: int = 3
    retry_delay: float = 1.0

    # Memory
    memory_backend: MemoryBackend = MemoryBackend.LOCAL
    memory_dir: str = "./data/nexus_memory"
    max_working_memory: int = 50       # Max items in working memory
    max_episodic_memory: int = 10000   # Max episodes stored
    knowledge_graph_max_nodes: int = 100000
    memory_persistence: bool = True

    # Agents
    max_agents: int = 50
    max_agent_steps: int = 100   # Max steps per agent execution
    agent_timeout: float = 300.0  # 5 minutes default

    # Tools
    tool_timeout: float = 60.0
    sandbox_enabled: bool = True
    max_tool_calls_per_step: int = 10
    mcp_enabled: bool = True

    # Reflection
    reflection_enabled: bool = True
    max_reflection_cycles: int = 3
    reflection_quality_threshold: float = 0.8

    # Multi-modal
    vision_enabled: bool = True
    audio_enabled: bool = True
    video_enabled: bool = False  # Experimental

    # Communication
    webhook_port: int = 8765
    websocket_enabled: bool = True

    # Security
    security_level: SecurityLevel = SecurityLevel.STANDARD
    audit_enabled: bool = True
    audit_dir: str = "./data/nexus_audit"
    allowed_commands: list[str] = field(default_factory=lambda: [
        "python", "node", "git", "curl", "ls", "cat", "echo",
    ])
    blocked_commands: list[str] = field(default_factory=lambda: [
        "rm -rf /", "mkfs", "dd if=", ":(){ :|:& };:",
    ])

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8200
    api_cors_origins: list[str] = field(default_factory=lambda: ["*"])

    # Providers (auto-discovered)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)

    # Custom settings
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-discover providers from environment."""
        if not self.providers:
            self._auto_discover_providers()
        Path(self.memory_dir).mkdir(parents=True, exist_ok=True)
        Path(self.audit_dir).mkdir(parents=True, exist_ok=True)

    def _auto_discover_providers(self):
        """Discover available LLM providers from environment variables."""
        provider_names = [
            "anthropic", "openai", "google", "groq", "mistral",
            "together", "fireworks", "deepseek", "cohere", "perplexity",
            "openrouter", "ollama", "lm_studio", "cerebras", "sambanova",
        ]
        for name in provider_names:
            cfg = ProviderConfig.from_env(name)
            if cfg:
                self.providers[name] = cfg

        # Always include ollama as local provider
        if "ollama" not in self.providers:
            self.providers["ollama"] = ProviderConfig(
                name="ollama",
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                capabilities=["text", "code", "vision"],
            )

    @classmethod
    def from_file(cls, path: str) -> NexusConfig:
        """Load configuration from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_env(cls) -> NexusConfig:
        """Load configuration primarily from environment variables."""
        return cls(
            default_model=os.getenv("NEXUS_DEFAULT_MODEL", "claude-sonnet-4-6"),
            routing_strategy=RoutingStrategy(
                os.getenv("NEXUS_ROUTING_STRATEGY", "adaptive")
            ),
            security_level=SecurityLevel(
                os.getenv("NEXUS_SECURITY_LEVEL", "standard")
            ),
            memory_backend=MemoryBackend(
                os.getenv("NEXUS_MEMORY_BACKEND", "local")
            ),
        )

    def to_dict(self) -> dict:
        """Serialize configuration to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "default_model": self.default_model,
            "routing_strategy": self.routing_strategy.value,
            "memory_backend": self.memory_backend.value,
            "security_level": self.security_level.value,
            "providers": {
                k: {"name": v.name, "enabled": v.enabled}
                for k, v in self.providers.items()
            },
            "reflection_enabled": self.reflection_enabled,
            "mcp_enabled": self.mcp_enabled,
        }

"""
NEXUS Agent Manager
====================
Manages agent lifecycle: creation, retrieval, disposal, and monitoring.
"""

from __future__ import annotations

from nexus.agents.base import Agent, AgentState
from nexus.core.config import NexusConfig


class AgentManager:
    """Manages the lifecycle of all agents in the system."""

    def __init__(self, config: NexusConfig):
        self.config = config
        self._agents: dict[str, Agent] = {}

    def create_agent(
        self,
        name: str,
        role: str = "general",
        model: str | None = None,
        tools: list[str] | None = None,
        memory_enabled: bool = True,
        reflection_enabled: bool = False,
        **kwargs,
    ) -> Agent:
        """Create and register a new agent."""
        if name in self._agents:
            raise ValueError(f"Agent '{name}' already exists")
        if len(self._agents) >= self.config.max_agents:
            raise ValueError(f"Maximum agent limit ({self.config.max_agents}) reached")

        agent = Agent(
            name=name,
            role=role,
            model=model or self.config.default_model,
            tools=tools,
            memory_enabled=memory_enabled,
            reflection_enabled=reflection_enabled,
            max_steps=self.config.max_agent_steps,
            config=self.config,
            **kwargs,
        )
        self._agents[name] = agent
        return agent

    def get(self, name: str) -> Agent:
        """Retrieve an agent by name."""
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' not found")
        return self._agents[name]

    def remove(self, name: str):
        """Remove an agent."""
        if name in self._agents:
            del self._agents[name]

    def list_agents(self) -> list[dict]:
        """List all agents with their status."""
        return [agent.info for agent in self._agents.values()]

    def get_by_role(self, role: str) -> list[Agent]:
        """Get all agents with a specific role."""
        return [a for a in self._agents.values() if a.role == role]

    def get_idle(self) -> list[Agent]:
        """Get all idle agents."""
        return [a for a in self._agents.values() if a.state == AgentState.IDLE]

    @property
    def count(self) -> int:
        return len(self._agents)

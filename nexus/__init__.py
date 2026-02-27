"""
NEXUS AI Framework v1.0
========================
A next-generation autonomous AI agent framework that unifies and surpasses
OpenClaw + JARVIS + LangGraph + CrewAI + AutoGen.

Core Innovations:
  - Graph-based workflow orchestration with cycles, branches, checkpoints
  - Triple-layer memory: Working + Episodic + Semantic Knowledge Graph
  - Self-improving reflection engine with critic agents
  - MCP-compatible tool/skill ecosystem with sandboxed execution
  - Multi-agent collaboration with roles, consensus, debate, chain patterns
  - Provider-agnostic model router across 15+ LLM providers
  - Multi-modal pipeline: text, image, audio, video
  - Multi-platform communication: WhatsApp, Telegram, Slack, Discord, etc.
  - Security-first: sandboxed execution, permissions, audit trails
  - A2A protocol support for inter-agent communication

Usage:
    from nexus import Nexus

    nx = Nexus()
    nx.agent("researcher", role="research", model="claude-sonnet-4-6")
    nx.agent("coder", role="code", model="claude-sonnet-4-6")
    nx.crew("dev_team", agents=["researcher", "coder"], strategy="chain")
    result = await nx.run("Build a web scraper for news articles")
"""

__version__ = "1.0.0"
__author__ = "Neural Brain API"
__codename__ = "NEXUS"

from nexus.core.config import NexusConfig
from nexus.core.orchestrator import NexusOrchestrator
from nexus.core.workflow import WorkflowGraph, WorkflowNode
from nexus.agents.base import Agent, AgentState
from nexus.agents.manager import AgentManager
from nexus.agents.roles import RoleRegistry
from nexus.agents.collaboration import Crew, CollaborationStrategy
from nexus.memory.manager import MemoryManager
from nexus.memory.working import WorkingMemory
from nexus.memory.episodic import EpisodicMemory
from nexus.memory.semantic import SemanticMemory, KnowledgeGraph
from nexus.tools.registry import ToolRegistry
from nexus.tools.mcp import MCPServer, MCPClient
from nexus.tools.executor import SandboxedExecutor
from nexus.models.router import ModelRouter
from nexus.models.providers import ProviderManager
from nexus.reflection.engine import ReflectionEngine
from nexus.multimodal.pipeline import MultiModalPipeline
from nexus.communication.channels import ChannelManager
from nexus.security.sandbox import SecuritySandbox
from nexus.security.permissions import PermissionSystem
from nexus.security.audit import AuditTrail


class Nexus:
    """
    Main entry point for the NEXUS framework.
    Provides a fluent API for building autonomous AI agent systems.
    """

    def __init__(self, config: NexusConfig | None = None):
        self.config = config or NexusConfig()
        self.orchestrator = NexusOrchestrator(self.config)
        self.agent_manager = AgentManager(self.config)
        self.memory_manager = MemoryManager(self.config)
        self.tool_registry = ToolRegistry(self.config)
        self.model_router = ModelRouter(self.config)
        self.provider_manager = ProviderManager(self.config)
        self.reflection_engine = ReflectionEngine(self.config)
        self.multimodal = MultiModalPipeline(self.config)
        self.channels = ChannelManager(self.config)
        self.security = SecuritySandbox(self.config)
        self.permissions = PermissionSystem(self.config)
        self.audit = AuditTrail(self.config)
        self._crews: dict[str, Crew] = {}

    def agent(
        self,
        name: str,
        role: str = "general",
        model: str | None = None,
        tools: list[str] | None = None,
        memory: bool = True,
        reflection: bool = False,
        **kwargs,
    ) -> "Agent":
        """Register and create a new agent."""
        ag = self.agent_manager.create_agent(
            name=name,
            role=role,
            model=model or self.config.default_model,
            tools=tools or [],
            memory_enabled=memory,
            reflection_enabled=reflection,
            **kwargs,
        )
        if memory:
            self.memory_manager.attach(ag)
        if reflection:
            self.reflection_engine.attach(ag)
        self.audit.log("agent_created", agent=name, role=role)
        return ag

    def crew(
        self,
        name: str,
        agents: list[str],
        strategy: str = "chain",
        manager_model: str | None = None,
    ) -> "Crew":
        """Create a multi-agent crew with collaboration strategy."""
        agent_objs = [self.agent_manager.get(a) for a in agents]
        c = Crew(
            name=name,
            agents=agent_objs,
            strategy=CollaborationStrategy(strategy),
            manager_model=manager_model or self.config.default_model,
            config=self.config,
        )
        self._crews[name] = c
        self.audit.log("crew_created", crew=name, agents=agents, strategy=strategy)
        return c

    def tool(self, name: str, func=None, description: str = "", **kwargs):
        """Register a tool/skill in the ecosystem."""
        if func is None:
            def decorator(f):
                self.tool_registry.register(name, f, description, **kwargs)
                return f
            return decorator
        self.tool_registry.register(name, func, description, **kwargs)

    def workflow(self, name: str = "default") -> "WorkflowGraph":
        """Create a new graph-based workflow."""
        wf = WorkflowGraph(name=name, config=self.config)
        return wf

    async def run(
        self,
        task: str,
        crew: str | None = None,
        agent: str | None = None,
        workflow: WorkflowGraph | None = None,
        stream: bool = False,
        **kwargs,
    ):
        """Execute a task through the NEXUS orchestrator."""
        self.audit.log("task_started", task=task[:200])

        result = await self.orchestrator.execute(
            task=task,
            crew=self._crews.get(crew) if crew else None,
            agent=self.agent_manager.get(agent) if agent else None,
            workflow=workflow,
            tool_registry=self.tool_registry,
            model_router=self.model_router,
            memory_manager=self.memory_manager,
            reflection_engine=self.reflection_engine,
            stream=stream,
            **kwargs,
        )

        self.audit.log("task_completed", task=task[:200], success=result.success)
        return result

    async def chat(self, message: str, session_id: str = "default", **kwargs):
        """Interactive chat with memory and context."""
        return await self.orchestrator.chat(
            message=message,
            session_id=session_id,
            memory_manager=self.memory_manager,
            model_router=self.model_router,
            **kwargs,
        )

    def connect_channel(self, channel_type: str, **credentials):
        """Connect a messaging channel (WhatsApp, Telegram, Slack, etc.)."""
        self.channels.connect(channel_type, **credentials)
        self.audit.log("channel_connected", channel=channel_type)

    def mcp_server(self, name: str, **kwargs) -> "MCPServer":
        """Create an MCP-compatible tool server."""
        return MCPServer(name=name, registry=self.tool_registry, config=self.config, **kwargs)

    def mcp_client(self, server_url: str, **kwargs) -> "MCPClient":
        """Connect to an external MCP server."""
        client = MCPClient(server_url=server_url, config=self.config, **kwargs)
        self.tool_registry.register_mcp_client(client)
        return client

    @property
    def status(self) -> dict:
        """Get framework status."""
        return {
            "version": __version__,
            "agents": self.agent_manager.count,
            "crews": len(self._crews),
            "tools": self.tool_registry.count,
            "memory_entries": self.memory_manager.total_entries,
            "models_available": self.model_router.available_models,
            "channels_connected": self.channels.connected_count,
        }

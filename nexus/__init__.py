"""
TIRAM AI Framework v2.0
========================
The most advanced autonomous AI agent framework — your personal AI companion
that speaks, teaches, builds, and evolves.

Combines and surpasses: OpenClaw + JARVIS + LangGraph + CrewAI + AutoGen +
Khanmigo + Bolt.new + NVIDIA Audio2Face

Core Systems:
  - Orchestrator:     JARVIS-style 6-stage pipeline with graph workflows
  - Agent System:     ReAct agents with 8+ roles, 6 collaboration strategies
  - Memory:           Triple-layer — Working + Episodic + Knowledge Graph
  - Tools:            MCP-compatible ecosystem with sandboxed execution
  - Model Router:     15+ providers with adaptive routing
  - Reflection:       Self-improving with critique-refine cycles
  - Self-Upgrade:     Runtime skill discovery, install, and strategy evolution
  - World Skills:     50+ professional domains, 200+ templates
  - 3D Avatar:        Female AI persona with lip-sync, emotions, and voice
  - Automation:       End-to-end builders for websites, apps, APIs, ML, SaaS
  - Teaching:         Adaptive tutor with Socratic method, courses, assessments
  - Multilingual:     60+ languages with auto-detection and translation
  - Communication:    Telegram, Slack, Discord, WhatsApp, A2A protocol
  - Security:         Zero-trust permissions, sandbox, audit trails

Usage:
    from nexus import Tiram

    tiram = Tiram()

    # Create agents
    tiram.agent("researcher", role="research")
    tiram.agent("coder", role="code")
    tiram.crew("team", agents=["researcher", "coder"], strategy="chain")

    # Execute tasks
    result = await tiram.run("Build a complete SaaS dashboard with auth and payments")

    # Teach
    await tiram.teach("Explain quantum computing", language="es")

    # Build end-to-end
    await tiram.build("E-commerce website with product catalog and checkout")

    # Chat with avatar
    await tiram.avatar.process_interaction("Hello Tiram, help me learn Python")

    # Translate
    await tiram.translate("Hello world", target="ja")
"""

__version__ = "2.0.0"
__author__ = "Neural Brain API"
__codename__ = "TIRAM"

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

# TIRAM v2.0 new systems
from nexus.upgrade import SelfUpgradeEngine
from nexus.skills import WorldSkillsEngine
from nexus.avatar import AvatarPipeline, AvatarPersonality, AvatarVoice, AvatarAppearance
from nexus.automation import AutomationEngine
from nexus.teaching import TeachingEngine
from nexus.languages import MultilingualEngine


class Tiram:
    """
    TIRAM — Your personal AI companion.

    A 3D avatar-powered, multilingual, self-evolving AI that can:
    - Build complete websites, apps, APIs, and ML systems end-to-end
    - Teach any subject in 60+ languages adaptively
    - Speak to you with a 3D animated female avatar
    - Self-upgrade: discover, install, and evolve skills at runtime
    - Master 50+ professional domains
    - Collaborate with multi-agent teams
    - Remember everything with triple-layer memory
    """

    def __init__(self, config: NexusConfig | None = None):
        self.config = config or NexusConfig(name="TIRAM")

        # Core systems (from v1)
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

        # TIRAM v2.0 new systems
        self.upgrade_engine = SelfUpgradeEngine(self.config)
        self.world_skills = WorldSkillsEngine(self.config)
        self.avatar = AvatarPipeline(
            personality=AvatarPersonality(name="Tiram"),
            config=self.config,
        )
        self.automation = AutomationEngine(self.config)
        self.teaching = TeachingEngine(self.config)
        self.multilingual = MultilingualEngine(self.config)

    # ===== Agent & Crew Management =====

    def agent(self, name: str, role: str = "general", model: str | None = None,
              tools: list[str] | None = None, memory: bool = True,
              reflection: bool = False, **kwargs) -> "Agent":
        ag = self.agent_manager.create_agent(
            name=name, role=role, model=model or self.config.default_model,
            tools=tools or [], memory_enabled=memory, reflection_enabled=reflection, **kwargs,
        )
        if memory:
            self.memory_manager.attach(ag)
        if reflection:
            self.reflection_engine.attach(ag)
        self.audit.log("agent_created", agent=name, role=role)
        return ag

    def crew(self, name: str, agents: list[str], strategy: str = "chain",
             manager_model: str | None = None) -> "Crew":
        agent_objs = [self.agent_manager.get(a) for a in agents]
        c = Crew(name=name, agents=agent_objs, strategy=CollaborationStrategy(strategy),
                 manager_model=manager_model or self.config.default_model, config=self.config)
        self._crews[name] = c
        self.audit.log("crew_created", crew=name, agents=agents, strategy=strategy)
        return c

    def tool(self, name: str, func=None, description: str = "", **kwargs):
        if func is None:
            def decorator(f):
                self.tool_registry.register(name, f, description, **kwargs)
                return f
            return decorator
        self.tool_registry.register(name, func, description, **kwargs)

    def workflow(self, name: str = "default") -> "WorkflowGraph":
        return WorkflowGraph(name=name, config=self.config)

    # ===== Task Execution =====

    async def run(self, task: str, crew: str | None = None, agent: str | None = None,
                  workflow: WorkflowGraph | None = None, stream: bool = False, **kwargs):
        self.audit.log("task_started", task=task[:200])
        skill = self.world_skills.find_skill_for_task(task)
        if skill:
            self.audit.log("skill_matched", skill=skill.domain.value)
        result = await self.orchestrator.execute(
            task=task, crew=self._crews.get(crew) if crew else None,
            agent=self.agent_manager.get(agent) if agent else None, workflow=workflow,
            tool_registry=self.tool_registry, model_router=self.model_router,
            memory_manager=self.memory_manager, reflection_engine=self.reflection_engine,
            stream=stream, **kwargs,
        )
        self.upgrade_engine.record_task_outcome(
            task_type=self.orchestrator._classify_task_type(task), success=result.success,
            quality=0.8 if result.success else 0.3, duration_ms=result.total_duration_ms,
            model_used=self.config.default_model, skills_used=[skill.domain.value] if skill else [],
        )
        self.audit.log("task_completed", task=task[:200], success=result.success)
        return result

    async def chat(self, message: str, session_id: str = "default", **kwargs):
        return await self.orchestrator.chat(
            message=message, session_id=session_id,
            memory_manager=self.memory_manager, model_router=self.model_router, **kwargs,
        )

    # ===== TIRAM v2.0: Build, Teach, Translate, Speak =====

    async def build(self, description: str, pipeline_type: str = "auto", **kwargs):
        """Build a complete project end-to-end."""
        self.audit.log("build_started", description=description[:200])
        return await self.automation.build(
            description=description, pipeline_type=pipeline_type,
            model_router=self.model_router, tool_registry=self.tool_registry, **kwargs,
        )

    async def teach(self, topic: str, learner_id: str = "default",
                    language: str = "en", method: str | None = None, **kwargs):
        """Teach a topic adaptively in any language."""
        from nexus.teaching import TeachingMethod
        tm = TeachingMethod(method) if method else None
        return await self.teaching.teach(
            topic=topic, learner_id=learner_id, method=tm,
            language=language, model_router=self.model_router, **kwargs,
        )

    async def translate(self, text: str, target: str, source: str = "auto"):
        """Translate text to any of 60+ languages."""
        return await self.multilingual.translate(
            text=text, target_language=target, source_language=source,
            model_router=self.model_router,
        )

    async def speak(self, message: str, language: str = "en"):
        """Interact with the 3D avatar."""
        return await self.avatar.process_interaction(
            user_input=message, model_router=self.model_router, input_language=language,
        )

    async def generate_course(self, topic: str, difficulty: str = "beginner",
                               language: str = "en", modules: int = 5):
        """Generate a complete course on any topic."""
        from nexus.teaching import DifficultyLevel
        return await self.teaching.generate_course(
            topic=topic, difficulty=DifficultyLevel(difficulty),
            num_modules=modules, language=language, model_router=self.model_router,
        )

    async def upgrade(self):
        """Run self-optimization cycle."""
        return await self.upgrade_engine.optimize(self.model_router)

    async def install_skill(self, skill_name: str):
        """Install a new skill at runtime."""
        return await self.upgrade_engine.install_skill(skill_name)

    def connect_channel(self, channel_type: str, **credentials):
        self.channels.connect(channel_type, **credentials)
        self.audit.log("channel_connected", channel=channel_type)

    def mcp_server(self, name: str, **kwargs) -> "MCPServer":
        return MCPServer(name=name, registry=self.tool_registry, config=self.config, **kwargs)

    def mcp_client(self, server_url: str, **kwargs) -> "MCPClient":
        client = MCPClient(server_url=server_url, config=self.config, **kwargs)
        self.tool_registry.register_mcp_client(client)
        return client

    @property
    def status(self) -> dict:
        return {
            "name": "TIRAM",
            "version": __version__,
            "agents": self.agent_manager.count,
            "crews": len(self._crews),
            "tools": self.tool_registry.count,
            "memory_entries": self.memory_manager.total_entries,
            "models_available": self.model_router.available_models,
            "channels_connected": self.channels.connected_count,
            "world_skills": self.world_skills.stats,
            "languages_supported": self.multilingual.language_count,
            "upgrade_metrics": self.upgrade_engine.metrics,
            "teaching_stats": self.teaching.stats,
            "automation_pipelines": len(self.automation.list_pipelines()),
            "avatar_state": self.avatar.state,
        }


# Backward compatibility
Nexus = Tiram

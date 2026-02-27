"""
NEXUS Framework Test Suite
============================
Comprehensive tests for all framework components.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 1. Configuration Tests
# ============================================================

class TestNexusConfig(unittest.TestCase):
    """Test the configuration system."""

    def test_default_config(self):
        from nexus.core.config import NexusConfig
        config = NexusConfig()
        self.assertEqual(config.name, "NEXUS")
        self.assertEqual(config.version, "1.0.0")
        self.assertEqual(config.default_model, "claude-sonnet-4-6")
        self.assertTrue(config.reflection_enabled)
        self.assertTrue(config.mcp_enabled)

    def test_config_routing_strategies(self):
        from nexus.core.config import NexusConfig, RoutingStrategy
        config = NexusConfig(routing_strategy=RoutingStrategy.FASTEST)
        self.assertEqual(config.routing_strategy, RoutingStrategy.FASTEST)

    def test_config_security_levels(self):
        from nexus.core.config import NexusConfig, SecurityLevel
        config = NexusConfig(security_level=SecurityLevel.PARANOID)
        self.assertEqual(config.security_level, SecurityLevel.PARANOID)

    def test_config_auto_discover_providers(self):
        from nexus.core.config import NexusConfig
        config = NexusConfig()
        # Should always have ollama as default
        self.assertIn("ollama", config.providers)

    def test_config_to_dict(self):
        from nexus.core.config import NexusConfig
        config = NexusConfig()
        d = config.to_dict()
        self.assertEqual(d["name"], "NEXUS")
        self.assertEqual(d["version"], "1.0.0")
        self.assertIn("providers", d)

    def test_provider_config(self):
        from nexus.core.config import ProviderConfig
        pc = ProviderConfig(name="test", api_key="key123", base_url="http://localhost")
        self.assertEqual(pc.name, "test")
        self.assertTrue(pc.enabled)
        self.assertEqual(pc.max_concurrent, 10)


# ============================================================
# 2. Workflow Engine Tests
# ============================================================

class TestWorkflowEngine(unittest.TestCase):
    """Test the graph-based workflow engine."""

    def test_create_workflow(self):
        from nexus.core.workflow import WorkflowGraph, NodeType
        wf = WorkflowGraph(name="test")
        wf.add_node("start", NodeType.AGENT)
        wf.add_node("end", NodeType.AGENT)
        wf.add_edge("start", "end")
        self.assertEqual(len(wf.nodes), 2)
        self.assertEqual(len(wf.edges), 1)

    def test_workflow_compilation(self):
        from nexus.core.workflow import WorkflowGraph, NodeType
        wf = WorkflowGraph(name="test")
        wf.add_node("start", NodeType.AGENT)
        wf.add_node("end", NodeType.AGENT)
        wf.add_edge("start", "end")
        wf.compile()
        self.assertTrue(wf._compiled)

    def test_workflow_invalid_compilation(self):
        from nexus.core.workflow import WorkflowGraph, NodeType
        wf = WorkflowGraph(name="test")
        wf.entry_node = "nonexistent"
        with self.assertRaises(ValueError):
            wf.compile()

    def test_workflow_chaining(self):
        from nexus.core.workflow import WorkflowGraph, NodeType
        wf = (
            WorkflowGraph(name="chain")
            .add_node("a", NodeType.AGENT)
            .add_node("b", NodeType.AGENT)
            .add_node("c", NodeType.AGENT)
            .add_edge("a", "b")
            .add_edge("b", "c")
        )
        self.assertEqual(len(wf.nodes), 3)
        self.assertEqual(len(wf.edges), 2)

    def test_workflow_state(self):
        from nexus.core.workflow import WorkflowState
        state = WorkflowState()
        state.set("key", "value")
        self.assertEqual(state.get("key"), "value")
        state.add_message("user", "hello")
        self.assertEqual(len(state.messages), 1)

    def test_workflow_checkpoint(self):
        from nexus.core.workflow import WorkflowState
        state = WorkflowState()
        state.set("x", 42)
        state.checkpoint("test_checkpoint")
        self.assertEqual(len(state.checkpoints), 1)
        self.assertEqual(state.checkpoints[0]["label"], "test_checkpoint")

    def test_workflow_conditional_edges(self):
        from nexus.core.workflow import WorkflowGraph, WorkflowState, NodeType
        wf = WorkflowGraph(name="conditional")
        wf.add_node("router", NodeType.CONDITION)
        wf.add_node("path_a", NodeType.AGENT)
        wf.add_node("path_b", NodeType.AGENT)

        wf.add_conditional_edges(
            "router",
            lambda state: "a" if state.get("choice") == "a" else "b",
            {"a": "path_a", "b": "path_b"},
        )

        state = WorkflowState(data={"choice": "a"})
        next_nodes = wf.get_next_nodes("router", state)
        self.assertIn("path_a", next_nodes)

    def test_workflow_execute(self):
        from nexus.core.workflow import WorkflowGraph, WorkflowState, NodeType

        async def handler(state):
            state.set("processed", True)
            return state

        wf = WorkflowGraph(name="exec_test")
        wf.add_node("process", NodeType.TRANSFORM, handler=handler)
        wf.compile()

        state = asyncio.get_event_loop().run_until_complete(wf.execute())
        self.assertTrue(state.get("processed"))

    def test_workflow_visualize(self):
        from nexus.core.workflow import WorkflowGraph, NodeType
        wf = WorkflowGraph(name="viz")
        wf.add_node("a", NodeType.AGENT)
        wf.add_node("b", NodeType.TOOL)
        wf.add_edge("a", "b")
        viz = wf.visualize()
        self.assertIn("viz", viz)
        self.assertIn("a", viz)


# ============================================================
# 3. Agent Tests
# ============================================================

class TestAgent(unittest.TestCase):
    """Test the agent system."""

    def test_create_agent(self):
        from nexus.agents.base import Agent, AgentState
        agent = Agent(name="test_agent", role="coder", model="test-model")
        self.assertEqual(agent.name, "test_agent")
        self.assertEqual(agent.role, "coder")
        self.assertEqual(agent.state, AgentState.IDLE)

    def test_agent_default_prompt(self):
        from nexus.agents.base import Agent
        agent = Agent(name="coder1", role="code")
        self.assertIn("software engineer", agent.system_prompt)

    def test_agent_custom_prompt(self):
        from nexus.agents.base import Agent
        agent = Agent(name="custom", system_prompt="You are custom.")
        self.assertEqual(agent.system_prompt, "You are custom.")

    def test_agent_info(self):
        from nexus.agents.base import Agent
        agent = Agent(name="info_test", role="research")
        info = agent.info
        self.assertEqual(info["name"], "info_test")
        self.assertEqual(info["role"], "research")
        self.assertIn("state", info)

    def test_agent_hook(self):
        from nexus.agents.base import Agent
        agent = Agent(name="hook_test")
        called = []
        agent.hook("before_execute", lambda **kw: called.append("before"))
        self.assertEqual(len(agent._hooks["before_execute"]), 1)

    def test_agent_repr(self):
        from nexus.agents.base import Agent
        agent = Agent(name="repr_test", role="code", model="test")
        self.assertIn("repr_test", repr(agent))


# ============================================================
# 4. Agent Manager Tests
# ============================================================

class TestAgentManager(unittest.TestCase):
    """Test agent lifecycle management."""

    def test_create_and_get(self):
        from nexus.core.config import NexusConfig
        from nexus.agents.manager import AgentManager
        mgr = AgentManager(NexusConfig())
        agent = mgr.create_agent("test", role="code")
        retrieved = mgr.get("test")
        self.assertEqual(agent.name, retrieved.name)

    def test_duplicate_name_raises(self):
        from nexus.core.config import NexusConfig
        from nexus.agents.manager import AgentManager
        mgr = AgentManager(NexusConfig())
        mgr.create_agent("dup")
        with self.assertRaises(ValueError):
            mgr.create_agent("dup")

    def test_get_nonexistent_raises(self):
        from nexus.core.config import NexusConfig
        from nexus.agents.manager import AgentManager
        mgr = AgentManager(NexusConfig())
        with self.assertRaises(KeyError):
            mgr.get("nonexistent")

    def test_remove_agent(self):
        from nexus.core.config import NexusConfig
        from nexus.agents.manager import AgentManager
        mgr = AgentManager(NexusConfig())
        mgr.create_agent("removable")
        mgr.remove("removable")
        self.assertEqual(mgr.count, 0)

    def test_list_agents(self):
        from nexus.core.config import NexusConfig
        from nexus.agents.manager import AgentManager
        mgr = AgentManager(NexusConfig())
        mgr.create_agent("a1", role="code")
        mgr.create_agent("a2", role="research")
        listing = mgr.list_agents()
        self.assertEqual(len(listing), 2)

    def test_get_by_role(self):
        from nexus.core.config import NexusConfig
        from nexus.agents.manager import AgentManager
        mgr = AgentManager(NexusConfig())
        mgr.create_agent("c1", role="code")
        mgr.create_agent("c2", role="code")
        mgr.create_agent("r1", role="research")
        coders = mgr.get_by_role("code")
        self.assertEqual(len(coders), 2)


# ============================================================
# 5. Role Registry Tests
# ============================================================

class TestRoleRegistry(unittest.TestCase):
    """Test the role registry."""

    def test_builtin_roles(self):
        from nexus.agents.roles import RoleRegistry
        registry = RoleRegistry()
        roles = registry.list_roles()
        self.assertIn("researcher", roles)
        self.assertIn("coder", roles)
        self.assertIn("critic", roles)
        self.assertIn("architect", roles)

    def test_get_role(self):
        from nexus.agents.roles import RoleRegistry
        registry = RoleRegistry()
        role = registry.get("coder")
        self.assertEqual(role.name, "coder")
        self.assertIn("code_generation", role.capabilities)

    def test_custom_role(self):
        from nexus.agents.roles import RoleRegistry, RoleDefinition
        registry = RoleRegistry()
        custom = RoleDefinition(name="custom_role", description="Test", system_prompt="Test prompt")
        registry.register(custom)
        retrieved = registry.get("custom_role")
        self.assertEqual(retrieved.description, "Test")

    def test_role_for_task_type(self):
        from nexus.agents.roles import RoleRegistry
        registry = RoleRegistry()
        self.assertEqual(registry.get_for_task_type("code"), "coder")
        self.assertEqual(registry.get_for_task_type("research"), "researcher")
        self.assertEqual(registry.get_for_task_type("review"), "critic")


# ============================================================
# 6. Collaboration Tests
# ============================================================

class TestCollaboration(unittest.TestCase):
    """Test multi-agent collaboration patterns."""

    def test_create_crew(self):
        from nexus.agents.base import Agent
        from nexus.agents.collaboration import Crew, CollaborationStrategy
        agents = [Agent(name="a"), Agent(name="b")]
        crew = Crew(name="test_crew", agents=agents, strategy=CollaborationStrategy.CHAIN)
        self.assertEqual(crew.name, "test_crew")
        self.assertEqual(len(crew.agents), 2)

    def test_crew_info(self):
        from nexus.agents.base import Agent
        from nexus.agents.collaboration import Crew, CollaborationStrategy
        agents = [Agent(name="x")]
        crew = Crew(name="info_crew", agents=agents)
        info = crew.info
        self.assertEqual(info["name"], "info_crew")
        self.assertIn("agents", info)

    def test_all_strategies_exist(self):
        from nexus.agents.collaboration import CollaborationStrategy
        strategies = list(CollaborationStrategy)
        self.assertIn(CollaborationStrategy.CHAIN, strategies)
        self.assertIn(CollaborationStrategy.CONSENSUS, strategies)
        self.assertIn(CollaborationStrategy.DEBATE, strategies)
        self.assertIn(CollaborationStrategy.PARALLEL, strategies)
        self.assertIn(CollaborationStrategy.HIERARCHY, strategies)
        self.assertIn(CollaborationStrategy.SWARM, strategies)


# ============================================================
# 7. Working Memory Tests
# ============================================================

class TestWorkingMemory(unittest.TestCase):
    """Test short-term working memory."""

    def test_add_and_retrieve(self):
        from nexus.memory.working import WorkingMemory
        wm = WorkingMemory(max_items=10)
        wm.add({"role": "user", "content": "hello"})
        recent = wm.get_recent(1)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["content"], "hello")

    def test_capacity_limit(self):
        from nexus.memory.working import WorkingMemory
        wm = WorkingMemory(max_items=5)
        for i in range(20):
            wm.add({"i": i})
        # Should not exceed max_items
        all_items = wm.get_all()
        self.assertLessEqual(len(all_items), 10)  # max_items * 2 deque

    def test_pinned_items(self):
        from nexus.memory.working import WorkingMemory
        wm = WorkingMemory(max_items=5)
        wm.pin({"important": True})
        all_items = wm.get_all()
        self.assertTrue(any(item.get("important") for item in all_items))

    def test_search(self):
        from nexus.memory.working import WorkingMemory
        wm = WorkingMemory()
        wm.add({"content": "The quick brown fox"})
        wm.add({"content": "Lazy dog sleeping"})
        results = wm.search("fox")
        self.assertEqual(len(results), 1)

    def test_clear(self):
        from nexus.memory.working import WorkingMemory
        wm = WorkingMemory()
        wm.add({"test": 1})
        wm.clear()
        self.assertEqual(len(wm.get_recent()), 0)

    def test_tags(self):
        from nexus.memory.working import WorkingMemory
        wm = WorkingMemory()
        wm.add({"content": "tagged"}, tags=["important"])
        results = wm.get_by_tags(["important"])
        self.assertEqual(len(results), 1)

    def test_summarize_context(self):
        from nexus.memory.working import WorkingMemory
        wm = WorkingMemory()
        wm.add({"role": "user", "content": "test message"})
        summary = wm.summarize_context()
        self.assertIn("test message", summary)


# ============================================================
# 8. Episodic Memory Tests
# ============================================================

class TestEpisodicMemory(unittest.TestCase):
    """Test long-term episodic memory."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.tmpdir, "episodes.jsonl")

    def test_store_episode(self):
        from nexus.memory.episodic import EpisodicMemory
        em = EpisodicMemory(storage_path=self.storage_path)
        ep = asyncio.get_event_loop().run_until_complete(
            em.store({"task": "test task", "result": "success", "success": True})
        )
        self.assertIsNotNone(ep.id)
        self.assertEqual(em.count, 1)

    def test_search_episodes(self):
        from nexus.memory.episodic import EpisodicMemory
        em = EpisodicMemory(storage_path=self.storage_path)
        asyncio.get_event_loop().run_until_complete(
            em.store({"task": "build a website", "result": "done", "success": True})
        )
        asyncio.get_event_loop().run_until_complete(
            em.store({"task": "fix a database bug", "result": "fixed", "success": True})
        )
        results = em.search("website")
        self.assertGreater(len(results), 0)

    def test_get_recent(self):
        from nexus.memory.episodic import EpisodicMemory
        em = EpisodicMemory(storage_path=self.storage_path)
        for i in range(5):
            asyncio.get_event_loop().run_until_complete(
                em.store({"task": f"task {i}", "success": True})
            )
        recent = em.get_recent(3)
        self.assertEqual(len(recent), 3)

    def test_successful_patterns(self):
        from nexus.memory.episodic import EpisodicMemory
        em = EpisodicMemory(storage_path=self.storage_path)
        asyncio.get_event_loop().run_until_complete(
            em.store({"task": "code task", "success": True, "model_used": "claude", "task_type": "code"})
        )
        patterns = em.get_successful_patterns("code")
        self.assertGreater(patterns["success_rate"], 0)

    def test_persistence(self):
        from nexus.memory.episodic import EpisodicMemory
        em1 = EpisodicMemory(storage_path=self.storage_path)
        asyncio.get_event_loop().run_until_complete(
            em1.store({"task": "persistent task", "success": True})
        )

        # Reload from disk
        em2 = EpisodicMemory(storage_path=self.storage_path)
        self.assertEqual(em2.count, 1)


# ============================================================
# 9. Knowledge Graph Tests
# ============================================================

class TestKnowledgeGraph(unittest.TestCase):
    """Test the knowledge graph / semantic memory."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.tmpdir, "kg.json")

    def test_add_entity(self):
        from nexus.memory.semantic import KnowledgeGraph
        kg = KnowledgeGraph(storage_path=self.storage_path)
        entity = kg.add_entity("Python", entity_type="language")
        self.assertEqual(entity.name, "Python")
        self.assertEqual(entity.entity_type, "language")

    def test_add_relationship(self):
        from nexus.memory.semantic import KnowledgeGraph
        kg = KnowledgeGraph(storage_path=self.storage_path)
        e1 = kg.add_entity("Python")
        e2 = kg.add_entity("FastAPI")
        rel = kg.add_relationship(e1, e2, "is_used_by")
        self.assertEqual(rel.relation_type, "is_used_by")

    def test_find_entities(self):
        from nexus.memory.semantic import KnowledgeGraph
        kg = KnowledgeGraph(storage_path=self.storage_path)
        kg.add_entity("Python Programming")
        kg.add_entity("JavaScript")
        results = kg.find_entities("python")
        self.assertEqual(len(results), 1)

    def test_get_relationships(self):
        from nexus.memory.semantic import KnowledgeGraph
        kg = KnowledgeGraph(storage_path=self.storage_path)
        e1 = kg.add_entity("A")
        e2 = kg.add_entity("B")
        kg.add_relationship(e1, e2, "connects_to")
        rels = kg.get_relationships(e1.id)
        self.assertEqual(len(rels), 1)

    def test_traverse(self):
        from nexus.memory.semantic import KnowledgeGraph
        kg = KnowledgeGraph(storage_path=self.storage_path)
        e1 = kg.add_entity("Root")
        e2 = kg.add_entity("Child1")
        e3 = kg.add_entity("Child2")
        kg.add_relationship(e1, e2, "has")
        kg.add_relationship(e1, e3, "has")
        result = kg.traverse(e1.id, max_hops=2)
        self.assertEqual(result["nodes_visited"], 3)

    def test_find_path(self):
        from nexus.memory.semantic import KnowledgeGraph
        kg = KnowledgeGraph(storage_path=self.storage_path)
        e1 = kg.add_entity("Start")
        e2 = kg.add_entity("Middle")
        e3 = kg.add_entity("End")
        kg.add_relationship(e1, e2, "leads_to")
        kg.add_relationship(e2, e3, "leads_to")
        path = kg.find_path(e1.id, e3.id)
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 3)

    def test_detect_communities(self):
        from nexus.memory.semantic import KnowledgeGraph
        kg = KnowledgeGraph(storage_path=self.storage_path)
        e1 = kg.add_entity("A")
        e2 = kg.add_entity("B")
        e3 = kg.add_entity("C")
        kg.add_relationship(e1, e2, "related")
        kg.add_relationship(e2, e3, "related")
        communities = kg.detect_communities()
        self.assertGreater(len(communities), 0)

    def test_save_and_load(self):
        from nexus.memory.semantic import KnowledgeGraph
        kg1 = KnowledgeGraph(storage_path=self.storage_path)
        kg1.add_entity("Persistent", entity_type="test")
        kg1.save()

        kg2 = KnowledgeGraph(storage_path=self.storage_path)
        self.assertEqual(len(kg2.entities), 1)

    def test_semantic_memory(self):
        from nexus.memory.semantic import SemanticMemory
        sm = SemanticMemory(storage_path=self.storage_path)
        sm.learn_fact("Python", "is_a", "Programming Language")
        results = sm.recall("Python")
        self.assertGreater(len(results), 0)

    def test_knowledge_query(self):
        from nexus.memory.semantic import KnowledgeGraph
        kg = KnowledgeGraph(storage_path=self.storage_path)
        kg.add_entity("Machine Learning")
        kg.add_entity("Neural Network")
        e1 = kg.add_entity("Deep Learning")
        e2 = kg.add_entity("TensorFlow")
        kg.add_relationship(e1, e2, "uses")
        results = kg.query("Deep Learning")
        self.assertGreater(len(results), 0)

    def test_stats(self):
        from nexus.memory.semantic import KnowledgeGraph
        kg = KnowledgeGraph(storage_path=self.storage_path)
        kg.add_entity("X")
        kg.add_entity("Y")
        stats = kg.stats
        self.assertEqual(stats["entities"], 2)


# ============================================================
# 10. Tool Registry Tests
# ============================================================

class TestToolRegistry(unittest.TestCase):
    """Test the tool registry."""

    def test_register_tool(self):
        from nexus.core.config import NexusConfig
        from nexus.tools.registry import ToolRegistry

        registry = ToolRegistry(NexusConfig())

        async def my_tool(text: str) -> str:
            return f"Processed: {text}"

        registry.register("my_tool", my_tool, description="Test tool")
        self.assertEqual(registry.count, 1)

    def test_get_definitions(self):
        from nexus.core.config import NexusConfig
        from nexus.tools.registry import ToolRegistry

        registry = ToolRegistry(NexusConfig())

        async def greet(name: str) -> str:
            return f"Hello {name}"

        registry.register("greet", greet, description="Greet someone")
        defs = registry.get_definitions()
        self.assertEqual(len(defs), 1)
        self.assertEqual(defs[0]["function"]["name"], "greet")

    def test_execute_tool(self):
        from nexus.core.config import NexusConfig
        from nexus.tools.registry import ToolRegistry

        registry = ToolRegistry(NexusConfig())

        async def add(a: int, b: int) -> int:
            return a + b

        registry.register("add", add, description="Add numbers")
        result = asyncio.get_event_loop().run_until_complete(
            registry.execute("add", {"a": 3, "b": 4})
        )
        self.assertEqual(result, 7)

    def test_execute_nonexistent_tool(self):
        from nexus.core.config import NexusConfig
        from nexus.tools.registry import ToolRegistry

        registry = ToolRegistry(NexusConfig())
        result = asyncio.get_event_loop().run_until_complete(
            registry.execute("nonexistent", {})
        )
        self.assertIn("not found", result)

    def test_match_tools(self):
        from nexus.core.config import NexusConfig
        from nexus.tools.registry import ToolRegistry

        registry = ToolRegistry(NexusConfig())

        async def search(query: str) -> str:
            return query

        registry.register("web_search", search, description="Search the web for information", tags=["web", "search"])
        matches = registry.match_tools("search for information on the web")
        self.assertIn("web_search", matches)

    def test_builtin_tools_registration(self):
        from nexus.core.config import NexusConfig
        from nexus.tools.registry import ToolRegistry
        from nexus.tools.builtins import register_builtin_tools

        registry = ToolRegistry(NexusConfig())
        register_builtin_tools(registry)
        self.assertGreaterEqual(registry.count, 8)  # At least 8 built-in tools
        tools = registry.list_tools()
        names = [t["name"] for t in tools]
        self.assertIn("file_read", names)
        self.assertIn("shell", names)
        self.assertIn("web_fetch", names)


# ============================================================
# 11. MCP Protocol Tests
# ============================================================

class TestMCPProtocol(unittest.TestCase):
    """Test MCP server and client."""

    def test_mcp_server_initialize(self):
        from nexus.tools.mcp import MCPServer
        server = MCPServer(name="test_server")
        result = asyncio.get_event_loop().run_until_complete(
            server.handle_request({"method": "initialize", "params": {}, "id": 1})
        )
        self.assertEqual(result["id"], 1)
        self.assertIn("result", result)
        self.assertIn("protocolVersion", result["result"])

    def test_mcp_server_tools_list(self):
        from nexus.core.config import NexusConfig
        from nexus.tools.mcp import MCPServer
        from nexus.tools.registry import ToolRegistry

        registry = ToolRegistry(NexusConfig())

        async def test_tool(x: str) -> str:
            return x

        registry.register("test_tool", test_tool, description="Test")
        server = MCPServer(name="test", registry=registry)

        result = asyncio.get_event_loop().run_until_complete(
            server.handle_request({"method": "tools/list", "params": {}, "id": 2})
        )
        tools = result["result"]["tools"]
        self.assertEqual(len(tools), 1)

    def test_mcp_server_tools_call(self):
        from nexus.core.config import NexusConfig
        from nexus.tools.mcp import MCPServer
        from nexus.tools.registry import ToolRegistry

        registry = ToolRegistry(NexusConfig())

        async def echo(text: str) -> str:
            return f"Echo: {text}"

        registry.register("echo", echo, description="Echo")
        server = MCPServer(name="test", registry=registry)

        result = asyncio.get_event_loop().run_until_complete(
            server.handle_request({
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {"text": "hello"}},
                "id": 3,
            })
        )
        self.assertIn("Echo: hello", result["result"]["content"][0]["text"])

    def test_mcp_server_unknown_method(self):
        from nexus.tools.mcp import MCPServer
        server = MCPServer(name="test")
        result = asyncio.get_event_loop().run_until_complete(
            server.handle_request({"method": "unknown/method", "params": {}, "id": 4})
        )
        self.assertIn("error", result)

    def test_mcp_server_resources(self):
        from nexus.tools.mcp import MCPServer
        server = MCPServer(name="test")
        server.add_resource("file:///test.txt", "Test File", description="A test file")
        result = asyncio.get_event_loop().run_until_complete(
            server.handle_request({"method": "resources/list", "params": {}, "id": 5})
        )
        self.assertEqual(len(result["result"]["resources"]), 1)

    def test_mcp_server_prompts(self):
        from nexus.tools.mcp import MCPServer
        server = MCPServer(name="test")
        server.add_prompt("summarize", template="Summarize: {text}", description="Summarize text")
        result = asyncio.get_event_loop().run_until_complete(
            server.handle_request({"method": "prompts/list", "params": {}, "id": 6})
        )
        self.assertEqual(len(result["result"]["prompts"]), 1)

    def test_mcp_client_creation(self):
        from nexus.tools.mcp import MCPClient
        client = MCPClient(server_url="http://localhost:8080")
        self.assertFalse(client._connected)


# ============================================================
# 12. Model Router Tests
# ============================================================

class TestModelRouter(unittest.TestCase):
    """Test the model router."""

    def test_default_models(self):
        from nexus.models.router import DEFAULT_MODELS
        self.assertIn("claude-opus-4-6", DEFAULT_MODELS)
        self.assertIn("claude-sonnet-4-6", DEFAULT_MODELS)
        self.assertIn("gpt-4o", DEFAULT_MODELS)
        self.assertIn("gemini-2.0-flash", DEFAULT_MODELS)
        self.assertIn("deepseek-r1", DEFAULT_MODELS)

    def test_model_profiles(self):
        from nexus.models.router import DEFAULT_MODELS
        opus = DEFAULT_MODELS["claude-opus-4-6"]
        self.assertEqual(opus.provider, "anthropic")
        self.assertTrue(opus.supports_vision)
        self.assertGreater(opus.quality_score, 0.9)

    def test_select_model_fastest(self):
        from nexus.core.config import NexusConfig, RoutingStrategy
        from nexus.models.router import ModelRouter
        config = NexusConfig()
        router = ModelRouter(config)
        model = asyncio.get_event_loop().run_until_complete(
            router.select_model(strategy=RoutingStrategy.FASTEST)
        )
        self.assertIsNotNone(model)

    def test_select_model_cheapest(self):
        from nexus.core.config import NexusConfig, RoutingStrategy
        from nexus.models.router import ModelRouter
        config = NexusConfig()
        router = ModelRouter(config)
        model = asyncio.get_event_loop().run_until_complete(
            router.select_model(strategy=RoutingStrategy.CHEAPEST)
        )
        self.assertIsNotNone(model)

    def test_select_model_capability(self):
        from nexus.core.config import NexusConfig, RoutingStrategy
        from nexus.models.router import ModelRouter
        config = NexusConfig()
        router = ModelRouter(config)
        model = asyncio.get_event_loop().run_until_complete(
            router.select_model(task_type="code", strategy=RoutingStrategy.CAPABILITY)
        )
        self.assertIsNotNone(model)

    def test_select_vision_model(self):
        from nexus.core.config import NexusConfig, RoutingStrategy
        from nexus.models.router import ModelRouter
        config = NexusConfig()
        router = ModelRouter(config)
        model = asyncio.get_event_loop().run_until_complete(
            router.select_model(requires_vision=True, strategy=RoutingStrategy.BEST_QUALITY)
        )
        self.assertIsNotNone(model)

    def test_available_models(self):
        from nexus.core.config import NexusConfig
        from nexus.models.router import ModelRouter
        config = NexusConfig()
        router = ModelRouter(config)
        models = router.available_models
        self.assertGreater(len(models), 5)


# ============================================================
# 13. Reflection Engine Tests
# ============================================================

class TestReflectionEngine(unittest.TestCase):
    """Test the self-improvement reflection engine."""

    def test_score_quality(self):
        from nexus.core.config import NexusConfig
        from nexus.reflection.engine import ReflectionEngine
        engine = ReflectionEngine(NexusConfig())
        score = engine._score_quality(
            "Write a Python function",
            "def hello():\n    print('hello world')\n\nThis function prints hello world.",
            {"scores": {}},
        )
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_score_with_errors_penalized(self):
        from nexus.core.config import NexusConfig
        from nexus.reflection.engine import ReflectionEngine
        engine = ReflectionEngine(NexusConfig())
        score = engine._score_quality("task", "Error: unable to process", {"scores": {}})
        low_score = score
        score_good = engine._score_quality("task", "Here is a complete solution with details.", {"scores": {}})
        self.assertGreater(score_good, low_score)

    def test_score_from_critique(self):
        from nexus.core.config import NexusConfig
        from nexus.reflection.engine import ReflectionEngine
        engine = ReflectionEngine(NexusConfig())
        score = engine._score_quality("task", "result", {
            "scores": {"correctness": 0.9, "completeness": 0.8, "clarity": 0.7}
        })
        self.assertAlmostEqual(score, 0.8, places=1)

    def test_meta_reflect_empty(self):
        from nexus.core.config import NexusConfig
        from nexus.reflection.engine import ReflectionEngine
        engine = ReflectionEngine(NexusConfig())
        result = asyncio.get_event_loop().run_until_complete(engine.meta_reflect())
        self.assertEqual(result["avg_quality"], 0.0)

    def test_quality_rubric(self):
        from nexus.reflection.engine import QualityRubric
        rubric = QualityRubric(correctness=0.9, completeness=0.8, clarity=0.7, relevance=0.6, actionability=0.5)
        self.assertAlmostEqual(rubric.overall, 0.7, places=1)


# ============================================================
# 14. Multimodal Pipeline Tests
# ============================================================

class TestMultiModalPipeline(unittest.TestCase):
    """Test multi-modal processing."""

    def test_detect_text(self):
        from nexus.core.config import NexusConfig
        from nexus.multimodal.pipeline import MultiModalPipeline, Modality
        pipeline = MultiModalPipeline(NexusConfig())
        self.assertEqual(pipeline.detect_modality("Hello world"), Modality.TEXT)

    def test_detect_image(self):
        from nexus.core.config import NexusConfig
        from nexus.multimodal.pipeline import MultiModalPipeline, Modality
        pipeline = MultiModalPipeline(NexusConfig())
        self.assertEqual(pipeline.detect_modality("photo.jpg"), Modality.IMAGE)
        self.assertEqual(pipeline.detect_modality("image.png"), Modality.IMAGE)

    def test_detect_audio(self):
        from nexus.core.config import NexusConfig
        from nexus.multimodal.pipeline import MultiModalPipeline, Modality
        pipeline = MultiModalPipeline(NexusConfig())
        self.assertEqual(pipeline.detect_modality("recording.mp3"), Modality.AUDIO)

    def test_detect_code(self):
        from nexus.core.config import NexusConfig
        from nexus.multimodal.pipeline import MultiModalPipeline, Modality
        pipeline = MultiModalPipeline(NexusConfig())
        self.assertEqual(pipeline.detect_modality("script.py"), Modality.CODE)

    def test_detect_document(self):
        from nexus.core.config import NexusConfig
        from nexus.multimodal.pipeline import MultiModalPipeline, Modality
        pipeline = MultiModalPipeline(NexusConfig())
        self.assertEqual(pipeline.detect_modality("report.pdf"), Modality.DOCUMENT)

    def test_voice_pipeline(self):
        from nexus.core.config import NexusConfig
        from nexus.multimodal.pipeline import MultiModalPipeline
        pipeline = MultiModalPipeline(NexusConfig())
        voice = pipeline.create_voice_pipeline()
        self.assertFalse(voice.is_active)


# ============================================================
# 15. Communication Channel Tests
# ============================================================

class TestCommunication(unittest.TestCase):
    """Test communication channels."""

    def test_channel_types(self):
        from nexus.communication.channels import ChannelType
        types = list(ChannelType)
        self.assertIn(ChannelType.TELEGRAM, types)
        self.assertIn(ChannelType.SLACK, types)
        self.assertIn(ChannelType.DISCORD, types)
        self.assertIn(ChannelType.A2A, types)

    def test_channel_manager(self):
        from nexus.communication.channels import ChannelManager
        mgr = ChannelManager()
        self.assertEqual(mgr.connected_count, 0)

    def test_connect_webchat(self):
        from nexus.communication.channels import ChannelManager
        mgr = ChannelManager()
        adapter = mgr.connect("webchat")
        self.assertIsNotNone(adapter)

    def test_normalized_message(self):
        from nexus.communication.channels import NormalizedMessage, ChannelType
        msg = NormalizedMessage(
            channel=ChannelType.TELEGRAM,
            sender_id="user123",
            content="Hello bot",
        )
        self.assertEqual(msg.content, "Hello bot")
        self.assertEqual(msg.channel, ChannelType.TELEGRAM)

    def test_list_channels(self):
        from nexus.communication.channels import ChannelManager
        mgr = ChannelManager()
        mgr.connect("webchat")
        mgr.connect("slack", bot_token="fake-token")
        channels = mgr.list_channels()
        self.assertEqual(len(channels), 2)


# ============================================================
# 16. Security Tests
# ============================================================

class TestSecurity(unittest.TestCase):
    """Test security components."""

    def test_permission_check(self):
        from nexus.security.permissions import PermissionSystem, Permission
        ps = PermissionSystem()
        # Default policy should grant standard permissions
        self.assertTrue(ps.check("agent1", Permission.FILE_READ))
        self.assertTrue(ps.check("agent1", Permission.MODEL_CALL))

    def test_permission_denial(self):
        from nexus.security.permissions import PermissionSystem, Permission, SECURITY_PROFILES
        ps = PermissionSystem()
        ps.set_policy("restricted", "sandbox")
        self.assertFalse(ps.check("restricted", Permission.SHELL_EXEC))
        self.assertFalse(ps.check("restricted", Permission.FILE_WRITE))

    def test_permission_approval_needed(self):
        from nexus.security.permissions import PermissionSystem, Permission
        ps = PermissionSystem()
        # Standard policy requires approval for shell
        self.assertTrue(ps.requires_approval("agent1", Permission.SHELL_EXEC))

    def test_security_profiles(self):
        from nexus.security.permissions import SECURITY_PROFILES
        self.assertIn("minimal", SECURITY_PROFILES)
        self.assertIn("standard", SECURITY_PROFILES)
        self.assertIn("full", SECURITY_PROFILES)
        self.assertIn("sandbox", SECURITY_PROFILES)

    def test_sandbox_command_check(self):
        from nexus.core.config import NexusConfig
        from nexus.security.sandbox import SecuritySandbox
        sandbox = SecuritySandbox(NexusConfig())
        allowed, _ = sandbox.executor.is_command_allowed("ls -la")
        self.assertTrue(allowed)

    def test_sandbox_blocked_command(self):
        from nexus.core.config import NexusConfig
        from nexus.security.sandbox import SecuritySandbox
        sandbox = SecuritySandbox(NexusConfig())
        allowed, _ = sandbox.executor.is_command_allowed("rm -rf /")
        self.assertFalse(allowed)

    def test_audit_trail(self):
        from nexus.core.config import NexusConfig
        from nexus.security.audit import AuditTrail
        config = NexusConfig()
        audit = AuditTrail(config)
        audit.log("test_event", severity="info", key="value")
        entries = audit.get_recent(10)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["event"], "test_event")

    def test_audit_search(self):
        from nexus.core.config import NexusConfig
        from nexus.security.audit import AuditTrail
        config = NexusConfig()
        audit = AuditTrail(config)
        audit.log("agent_created", agent="test")
        audit.log("task_started", task="hello")
        results = audit.search("agent")
        self.assertEqual(len(results), 1)

    def test_audit_errors(self):
        from nexus.core.config import NexusConfig
        from nexus.security.audit import AuditTrail
        config = NexusConfig()
        audit = AuditTrail(config)
        audit.log("ok_event", severity="info")
        audit.log("bad_event", severity="error")
        errors = audit.get_errors()
        self.assertEqual(len(errors), 1)


# ============================================================
# 17. Orchestrator Tests
# ============================================================

class TestOrchestrator(unittest.TestCase):
    """Test the central orchestrator."""

    def test_assess_complexity(self):
        from nexus.core.config import NexusConfig
        from nexus.core.orchestrator import NexusOrchestrator
        orch = NexusOrchestrator(NexusConfig())
        self.assertEqual(orch._assess_complexity("hello"), "simple")
        self.assertEqual(orch._assess_complexity("build and deploy multiple services"), "medium")

    def test_classify_task_type(self):
        from nexus.core.config import NexusConfig
        from nexus.core.orchestrator import NexusOrchestrator
        orch = NexusOrchestrator(NexusConfig())
        self.assertEqual(orch._classify_task_type("write a Python function"), "code")
        self.assertEqual(orch._classify_task_type("research AI frameworks"), "research")
        self.assertEqual(orch._classify_task_type("analyze the data"), "analysis")

    def test_decompose_task(self):
        from nexus.core.config import NexusConfig
        from nexus.core.orchestrator import NexusOrchestrator
        orch = NexusOrchestrator(NexusConfig())
        subtasks = orch._decompose_task("first research the topic then write a summary")
        self.assertGreater(len(subtasks), 1)

    def test_plan_task(self):
        from nexus.core.config import NexusConfig
        from nexus.core.orchestrator import NexusOrchestrator
        orch = NexusOrchestrator(NexusConfig())
        plan = asyncio.get_event_loop().run_until_complete(
            orch._plan_task("build a web app")
        )
        self.assertIsNotNone(plan.id)
        self.assertGreater(len(plan.subtasks), 0)


# ============================================================
# 18. Integration: NEXUS Main Class Tests
# ============================================================

class TestNexusMain(unittest.TestCase):
    """Test the main Nexus class."""

    def test_nexus_creation(self):
        from nexus import Nexus
        nx = Nexus()
        self.assertIsNotNone(nx.orchestrator)
        self.assertIsNotNone(nx.agent_manager)
        self.assertIsNotNone(nx.memory_manager)
        self.assertIsNotNone(nx.tool_registry)
        self.assertIsNotNone(nx.model_router)

    def test_nexus_agent_creation(self):
        from nexus import Nexus
        nx = Nexus()
        agent = nx.agent("test_agent", role="code", model="test-model")
        self.assertEqual(agent.name, "test_agent")

    def test_nexus_crew_creation(self):
        from nexus import Nexus
        nx = Nexus()
        nx.agent("researcher", role="research")
        nx.agent("coder", role="code")
        crew = nx.crew("dev_team", agents=["researcher", "coder"], strategy="chain")
        self.assertEqual(crew.name, "dev_team")

    def test_nexus_tool_decorator(self):
        from nexus import Nexus
        nx = Nexus()

        @nx.tool("greet", description="Greet someone")
        async def greet(name: str) -> str:
            return f"Hello {name}"

        self.assertEqual(nx.tool_registry.count, 1)

    def test_nexus_workflow(self):
        from nexus import Nexus
        nx = Nexus()
        wf = nx.workflow("test")
        self.assertIsNotNone(wf)

    def test_nexus_status(self):
        from nexus import Nexus
        nx = Nexus()
        status = nx.status
        self.assertEqual(status["version"], "2.0.0")
        self.assertIn("agents", status)
        self.assertIn("tools", status)


# ============================================================
# 19. Memory Manager Integration Tests
# ============================================================

class TestMemoryManager(unittest.TestCase):
    """Test the unified memory manager."""

    def test_working_memory_per_session(self):
        from nexus.core.config import NexusConfig
        from nexus.memory.manager import MemoryManager
        mgr = MemoryManager(NexusConfig())
        wm1 = mgr.get_working_memory("session1")
        wm2 = mgr.get_working_memory("session2")
        wm1.add({"msg": "hello from session1"})
        self.assertEqual(wm1.count, 1)
        self.assertEqual(wm2.count, 0)

    def test_search_relevant(self):
        from nexus.core.config import NexusConfig
        from nexus.memory.manager import MemoryManager
        mgr = MemoryManager(NexusConfig())
        mgr.semantic.learn_fact("Python", "is_a", "Language")
        results = asyncio.get_event_loop().run_until_complete(
            mgr.search_relevant("Python")
        )
        self.assertGreater(len(results), 0)


# ============================================================
# 20. TIRAM v2.0: Self-Upgrade Engine Tests
# ============================================================

class TestSelfUpgradeEngine(unittest.TestCase):
    """Test the self-upgrade engine."""

    def test_create_engine(self):
        from nexus.core.config import NexusConfig
        from nexus.upgrade import SelfUpgradeEngine
        engine = SelfUpgradeEngine(NexusConfig())
        self.assertIsNotNone(engine)

    def test_record_task_outcome(self):
        from nexus.core.config import NexusConfig
        from nexus.upgrade import SelfUpgradeEngine
        engine = SelfUpgradeEngine(NexusConfig())
        engine.record_task_outcome(
            task_type="code", success=True, quality=0.9,
            duration_ms=1500, model_used="claude", skills_used=["code"],
        )
        self.assertEqual(engine._metrics.tasks_completed, 1)
        self.assertGreater(engine._metrics.task_success_rate, 0)

    def test_discover_skills(self):
        from nexus.core.config import NexusConfig
        from nexus.upgrade import SelfUpgradeEngine
        engine = SelfUpgradeEngine(NexusConfig())
        skills = asyncio.get_event_loop().run_until_complete(
            engine.discover_skills()
        )
        self.assertGreater(len(skills), 10)  # At least 10 discoverable skills

    def test_install_skill(self):
        from nexus.core.config import NexusConfig
        from nexus.upgrade import SelfUpgradeEngine, SkillManifest
        engine = SelfUpgradeEngine(NexusConfig())
        unique_name = f"test_skill_{id(self)}"
        engine.register_skill(SkillManifest(name=unique_name, version="1.0.0"))
        self.assertFalse(engine.get_skill(unique_name).installed)
        result = asyncio.get_event_loop().run_until_complete(
            engine.install_skill(unique_name)
        )
        self.assertTrue(result)
        self.assertTrue(engine.get_skill(unique_name).installed)
        self.assertIn(unique_name, [s.name for s in engine.get_installed_skills()])

    def test_upgrade_skill(self):
        from nexus.core.config import NexusConfig
        from nexus.upgrade import SelfUpgradeEngine, SkillManifest
        engine = SelfUpgradeEngine(NexusConfig())
        engine.register_skill(SkillManifest(name="upgradable", version="1.0.0", installed=True))
        result = asyncio.get_event_loop().run_until_complete(
            engine.upgrade_skill("upgradable")
        )
        self.assertTrue(result)
        self.assertEqual(engine.get_skill("upgradable").version, "1.0.1")

    def test_performance_metrics(self):
        from nexus.core.config import NexusConfig
        from nexus.upgrade import SelfUpgradeEngine
        engine = SelfUpgradeEngine(NexusConfig())
        for i in range(10):
            engine.record_task_outcome(
                task_type="code", success=i < 8, quality=0.7 + i * 0.02,
                duration_ms=1000, model_used="claude", skills_used=[],
            )
        metrics = engine.metrics
        self.assertEqual(metrics["tasks_completed"], 10)
        self.assertGreater(metrics["success_rate"], 0.5)

    def test_optimize_recommendations(self):
        from nexus.core.config import NexusConfig
        from nexus.upgrade import SelfUpgradeEngine
        engine = SelfUpgradeEngine(NexusConfig())
        recs = asyncio.get_event_loop().run_until_complete(engine.optimize())
        self.assertIsInstance(recs, list)


# ============================================================
# 21. TIRAM v2.0: World Skills Tests
# ============================================================

class TestWorldSkills(unittest.TestCase):
    """Test the world skills engine."""

    def test_create_engine(self):
        from nexus.skills import WorldSkillsEngine
        engine = WorldSkillsEngine()
        self.assertGreater(len(engine.list_domains()), 15)

    def test_find_skill_for_task(self):
        from nexus.skills import WorldSkillsEngine
        engine = WorldSkillsEngine()
        skill = engine.find_skill_for_task("Build a React dashboard with TailwindCSS")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.domain.value, "web_frontend")

    def test_find_backend_skill(self):
        from nexus.skills import WorldSkillsEngine
        engine = WorldSkillsEngine()
        skill = engine.find_skill_for_task("Create a FastAPI REST endpoint for users")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.domain.value, "web_backend")

    def test_find_ml_skill(self):
        from nexus.skills import WorldSkillsEngine
        engine = WorldSkillsEngine()
        skill = engine.find_skill_for_task("Train a machine learning model to classify images")
        self.assertIsNotNone(skill)

    def test_find_teaching_skill(self):
        from nexus.skills import WorldSkillsEngine
        engine = WorldSkillsEngine()
        skill = engine.find_skill_for_task("Teach me about databases and explain SQL joins")
        self.assertIsNotNone(skill)

    def test_list_templates(self):
        from nexus.skills import WorldSkillsEngine
        engine = WorldSkillsEngine()
        templates = engine.list_templates()
        self.assertGreater(len(templates), 5)  # Has multiple templates

    def test_get_template(self):
        from nexus.skills import WorldSkillsEngine
        engine = WorldSkillsEngine()
        template = engine.get_template("react_component")
        self.assertIsNotNone(template)
        self.assertIn("{name}", template.template)

    def test_stats(self):
        from nexus.skills import WorldSkillsEngine
        engine = WorldSkillsEngine()
        stats = engine.stats
        self.assertGreater(stats["total_skills"], 15)
        self.assertGreater(stats["total_templates"], 3)


# ============================================================
# 22. TIRAM v2.0: Avatar System Tests
# ============================================================

class TestAvatarSystem(unittest.TestCase):
    """Test the 3D avatar system."""

    def test_avatar_pipeline_creation(self):
        from nexus.avatar import AvatarPipeline
        pipeline = AvatarPipeline()
        self.assertEqual(pipeline.personality.name, "Tiram")

    def test_avatar_personality(self):
        from nexus.avatar import AvatarPersonality
        personality = AvatarPersonality(name="TestAvatar")
        prompt = personality.to_system_prompt()
        self.assertIn("TestAvatar", prompt)
        self.assertIn("intelligent", prompt)

    def test_avatar_voice_config(self):
        from nexus.avatar import AvatarVoice, VoiceStyle
        voice = AvatarVoice(provider="elevenlabs", style=VoiceStyle.WARM)
        self.assertEqual(voice.style, VoiceStyle.WARM)

    def test_avatar_appearance(self):
        from nexus.avatar import AvatarAppearance
        app = AvatarAppearance(name="Tiram", gender="female")
        self.assertEqual(app.gender, "female")

    def test_facial_expressions(self):
        from nexus.avatar import FacialExpression, EmotionalState
        for emotion in EmotionalState:
            expr = FacialExpression.for_emotion(emotion)
            self.assertEqual(expr.emotion, emotion)

    def test_emotion_detection(self):
        from nexus.avatar import AvatarPipeline
        pipeline = AvatarPipeline()
        emotion = pipeline._detect_emotion("I'm so happy with this!")
        self.assertEqual(emotion.value, "happy")

    def test_emotion_concerned(self):
        from nexus.avatar import AvatarPipeline
        pipeline = AvatarPipeline()
        emotion = pipeline._detect_emotion("There's a bug in the code, it's broken")
        self.assertEqual(emotion.value, "concerned")

    def test_viseme_generation(self):
        from nexus.avatar import AvatarPipeline
        pipeline = AvatarPipeline()
        visemes = pipeline._generate_viseme_sequence("Hello world")
        self.assertGreater(len(visemes), 0)

    def test_webgl_config(self):
        from nexus.avatar import AvatarPipeline
        pipeline = AvatarPipeline()
        config = pipeline.get_webgl_config()
        self.assertIn("avatar", config)
        self.assertIn("voice", config)
        self.assertIn("personality", config)
        self.assertIn("rendering", config)
        self.assertEqual(config["rendering"]["fps"], 60)

    def test_avatar_state(self):
        from nexus.avatar import AvatarPipeline
        pipeline = AvatarPipeline()
        state = pipeline.state
        self.assertEqual(state["name"], "Tiram")
        self.assertIn("emotion", state)


# ============================================================
# 23. TIRAM v2.0: Automation Pipeline Tests
# ============================================================

class TestAutomation(unittest.TestCase):
    """Test the end-to-end automation engine."""

    def test_create_engine(self):
        from nexus.automation import AutomationEngine
        engine = AutomationEngine()
        self.assertGreater(len(engine.list_pipelines()), 3)

    def test_list_pipelines(self):
        from nexus.automation import AutomationEngine
        engine = AutomationEngine()
        pipelines = engine.list_pipelines()
        types = [p["type"] for p in pipelines]
        self.assertIn("website", types)
        self.assertIn("saas", types)
        self.assertIn("api", types)
        self.assertIn("ml", types)

    def test_get_pipeline(self):
        from nexus.automation import AutomationEngine
        engine = AutomationEngine()
        pipeline = engine.get_pipeline("website")
        self.assertIsNotNone(pipeline)
        self.assertGreater(len(pipeline.steps), 5)

    def test_saas_pipeline(self):
        from nexus.automation import AutomationEngine
        engine = AutomationEngine()
        pipeline = engine.get_pipeline("saas")
        self.assertIsNotNone(pipeline)
        self.assertIn("Stripe", pipeline.tech_stack.get("payments", ""))

    def test_detect_pipeline_type(self):
        from nexus.automation import AutomationEngine
        engine = AutomationEngine()
        self.assertEqual(engine._detect_pipeline_type("Build a SaaS app with subscriptions"), "saas")
        self.assertEqual(engine._detect_pipeline_type("Create a REST API for users"), "api")
        self.assertEqual(engine._detect_pipeline_type("Train a machine learning model"), "ml")
        self.assertEqual(engine._detect_pipeline_type("Build a dashboard with analytics"), "dashboard")

    def test_pipeline_steps_have_prompts(self):
        from nexus.automation import AutomationEngine
        engine = AutomationEngine()
        for pipeline in engine._pipelines.values():
            for step in pipeline.steps:
                self.assertTrue(len(step.prompt_template) > 0, f"Step {step.name} has empty prompt")


# ============================================================
# 24. TIRAM v2.0: Teaching Engine Tests
# ============================================================

class TestTeachingEngine(unittest.TestCase):
    """Test the adaptive teaching engine."""

    def test_create_engine(self):
        from nexus.teaching import TeachingEngine
        engine = TeachingEngine()
        self.assertIsNotNone(engine)

    def test_learner_profile(self):
        from nexus.teaching import TeachingEngine, DifficultyLevel
        engine = TeachingEngine()
        learner = engine.get_learner("test_student")
        self.assertEqual(learner.level, DifficultyLevel.BEGINNER)
        self.assertEqual(learner.mastery_rate, 0.0)

    def test_learner_context(self):
        from nexus.teaching import TeachingEngine
        engine = TeachingEngine()
        learner = engine.get_learner("test")
        learner.name = "Alice"
        learner.topics_mastered = ["Python basics", "Variables"]
        context = learner.to_context()
        self.assertIn("Alice", context)
        self.assertIn("Python basics", context)

    def test_update_learner(self):
        from nexus.teaching import TeachingEngine, DifficultyLevel
        engine = TeachingEngine()
        engine.update_learner("test", name="Bob", level=DifficultyLevel.ADVANCED)
        learner = engine.get_learner("test")
        self.assertEqual(learner.name, "Bob")
        self.assertEqual(learner.level, DifficultyLevel.ADVANCED)

    def test_teaching_methods(self):
        from nexus.teaching import TeachingMethod
        methods = list(TeachingMethod)
        self.assertIn(TeachingMethod.SOCRATIC, methods)
        self.assertIn(TeachingMethod.EXPLAIN, methods)
        self.assertIn(TeachingMethod.EXAMPLE, methods)
        self.assertIn(TeachingMethod.PROJECT, methods)

    def test_difficulty_levels(self):
        from nexus.teaching import DifficultyLevel
        levels = list(DifficultyLevel)
        self.assertEqual(len(levels), 5)

    def test_stats(self):
        from nexus.teaching import TeachingEngine
        engine = TeachingEngine()
        engine.get_learner("a")
        engine.get_learner("b")
        stats = engine.stats
        self.assertEqual(stats["total_learners"], 2)


# ============================================================
# 25. TIRAM v2.0: Multilingual Engine Tests
# ============================================================

class TestMultilingual(unittest.TestCase):
    """Test the multilingual intelligence system."""

    def test_language_count(self):
        from nexus.languages import MultilingualEngine
        engine = MultilingualEngine()
        self.assertGreaterEqual(engine.language_count, 60)

    def test_detect_english(self):
        from nexus.languages import MultilingualEngine
        engine = MultilingualEngine()
        self.assertEqual(engine.detect_language("Hello, how are you?"), "en")

    def test_detect_spanish(self):
        from nexus.languages import MultilingualEngine
        engine = MultilingualEngine()
        self.assertEqual(engine.detect_language("Hola, como estas amigo"), "es")

    def test_detect_french(self):
        from nexus.languages import MultilingualEngine
        engine = MultilingualEngine()
        self.assertEqual(engine.detect_language("Bonjour, comment allez-vous dans cette ville"), "fr")

    def test_detect_chinese(self):
        from nexus.languages import MultilingualEngine
        engine = MultilingualEngine()
        self.assertEqual(engine.detect_language("你好世界"), "zh")

    def test_detect_japanese(self):
        from nexus.languages import MultilingualEngine
        engine = MultilingualEngine()
        self.assertEqual(engine.detect_language("こんにちは世界"), "ja")

    def test_detect_korean(self):
        from nexus.languages import MultilingualEngine
        engine = MultilingualEngine()
        self.assertEqual(engine.detect_language("안녕하세요"), "ko")

    def test_detect_arabic(self):
        from nexus.languages import MultilingualEngine
        engine = MultilingualEngine()
        self.assertEqual(engine.detect_language("مرحبا بالعالم"), "ar")

    def test_detect_hindi(self):
        from nexus.languages import MultilingualEngine
        engine = MultilingualEngine()
        self.assertEqual(engine.detect_language("नमस्ते दुनिया"), "hi")

    def test_detect_german(self):
        from nexus.languages import MultilingualEngine
        engine = MultilingualEngine()
        self.assertEqual(engine.detect_language("Guten Tag, wie geht es Ihnen in der Stadt"), "de")

    def test_detect_russian(self):
        from nexus.languages import MultilingualEngine
        engine = MultilingualEngine()
        self.assertEqual(engine.detect_language("Привет мир"), "ru")

    def test_list_languages(self):
        from nexus.languages import MultilingualEngine
        engine = MultilingualEngine()
        langs = engine.list_languages()
        self.assertGreaterEqual(len(langs), 60)
        codes = [l["code"] for l in langs]
        self.assertIn("en", codes)
        self.assertIn("zh", codes)
        self.assertIn("ar", codes)
        self.assertIn("hi", codes)

    def test_get_language_info(self):
        from nexus.languages import MultilingualEngine
        engine = MultilingualEngine()
        info = engine.get_language_info("ar")
        self.assertIsNotNone(info)
        self.assertEqual(info["direction"], "rtl")
        self.assertEqual(info["name"], "Arabic")

    def test_rtl_languages(self):
        from nexus.languages import MultilingualEngine
        engine = MultilingualEngine()
        rtl_codes = ["ar", "he", "fa", "ur"]
        for code in rtl_codes:
            info = engine.get_language_info(code)
            self.assertEqual(info["direction"], "rtl", f"{code} should be RTL")


# ============================================================
# 26. TIRAM v2.0: Integration Tests
# ============================================================

class TestTiramIntegration(unittest.TestCase):
    """Test the full TIRAM integration."""

    def test_tiram_creation(self):
        from nexus import Tiram
        tiram = Tiram()
        self.assertIsNotNone(tiram.upgrade_engine)
        self.assertIsNotNone(tiram.world_skills)
        self.assertIsNotNone(tiram.avatar)
        self.assertIsNotNone(tiram.automation)
        self.assertIsNotNone(tiram.teaching)
        self.assertIsNotNone(tiram.multilingual)

    def test_tiram_backward_compat(self):
        from nexus import Nexus, Tiram
        self.assertIs(Nexus, Tiram)

    def test_tiram_status(self):
        from nexus import Tiram
        tiram = Tiram()
        status = tiram.status
        self.assertEqual(status["name"], "TIRAM")
        self.assertEqual(status["version"], "2.0.0")
        self.assertIn("world_skills", status)
        self.assertIn("languages_supported", status)
        self.assertIn("upgrade_metrics", status)
        self.assertIn("teaching_stats", status)
        self.assertIn("avatar_state", status)
        self.assertGreaterEqual(status["languages_supported"], 60)

    def test_tiram_all_systems_initialized(self):
        from nexus import Tiram
        tiram = Tiram()
        # v1 systems
        self.assertIsNotNone(tiram.orchestrator)
        self.assertIsNotNone(tiram.agent_manager)
        self.assertIsNotNone(tiram.memory_manager)
        self.assertIsNotNone(tiram.tool_registry)
        self.assertIsNotNone(tiram.model_router)
        self.assertIsNotNone(tiram.reflection_engine)
        self.assertIsNotNone(tiram.security)
        self.assertIsNotNone(tiram.permissions)
        self.assertIsNotNone(tiram.audit)
        # v2 systems
        self.assertIsNotNone(tiram.upgrade_engine)
        self.assertIsNotNone(tiram.world_skills)
        self.assertIsNotNone(tiram.avatar)
        self.assertIsNotNone(tiram.automation)
        self.assertIsNotNone(tiram.teaching)
        self.assertIsNotNone(tiram.multilingual)

    def test_tiram_agent_workflow(self):
        from nexus import Tiram
        tiram = Tiram()
        agent = tiram.agent("test_agent", role="code")
        self.assertEqual(agent.name, "test_agent")

    def test_tiram_tool_decorator(self):
        from nexus import Tiram
        tiram = Tiram()

        @tiram.tool("greet", description="Greet someone")
        async def greet(name: str) -> str:
            return f"Hello {name}"

        self.assertEqual(tiram.tool_registry.count, 1)


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":
    # Count and run
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    test_count = 0
    for test_group in suite:
        for test in test_group:
            test_count += 1

    print(f"\n{'='*60}")
    print(f"  NEXUS Framework Test Suite — {test_count} tests")
    print(f"{'='*60}\n")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n{'='*60}")
    print(f"  Results: {result.testsRun} run, "
          f"{len(result.failures)} failed, "
          f"{len(result.errors)} errors")
    print(f"{'='*60}")

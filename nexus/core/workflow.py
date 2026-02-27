"""
NEXUS Workflow Engine
======================
Graph-based workflow orchestration with support for:
- Directed acyclic and cyclic graphs
- Conditional branching and merging
- Parallel execution lanes
- Checkpointing and resumption
- Dynamic node insertion at runtime
- Sub-workflow composition

Inspired by LangGraph but with cycles, checkpointing, and multi-agent support.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable


class NodeType(str, Enum):
    AGENT = "agent"           # Execute an agent
    TOOL = "tool"             # Execute a tool
    CONDITION = "condition"   # Conditional branch
    PARALLEL = "parallel"     # Parallel fork
    JOIN = "join"             # Parallel merge/join
    SUBWORKFLOW = "subworkflow"  # Nested workflow
    HUMAN = "human"           # Human-in-the-loop
    CHECKPOINT = "checkpoint" # Save state
    TRANSFORM = "transform"   # Data transformation
    REFLECTION = "reflection" # Self-reflection step


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"  # Waiting for human input or external event


@dataclass
class WorkflowState:
    """Mutable state that flows through the workflow graph."""
    data: dict[str, Any] = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    checkpoints: list[dict] = field(default_factory=list)

    def checkpoint(self, label: str = ""):
        """Save a checkpoint of current state."""
        import copy
        self.checkpoints.append({
            "label": label,
            "timestamp": time.time(),
            "data": copy.deepcopy(self.data),
            "message_count": len(self.messages),
        })

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        self.data[key] = value

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})

    def add_error(self, node: str, error: str, **kwargs):
        self.errors.append({"node": node, "error": error, "timestamp": time.time(), **kwargs})


@dataclass
class Edge:
    """Directed edge between workflow nodes."""
    source: str
    target: str
    condition: Callable[[WorkflowState], bool] | None = None
    label: str = ""
    priority: int = 0


@dataclass
class WorkflowNode:
    """A node in the workflow graph."""
    id: str
    name: str
    node_type: NodeType
    handler: Callable[[WorkflowState], Awaitable[WorkflowState]] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    duration_ms: float = 0
    retries: int = 0
    max_retries: int = 2
    timeout: float = 120.0

    # For agent nodes
    agent_name: str = ""
    agent_prompt: str = ""

    # For tool nodes
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)

    # For condition nodes
    condition_func: Callable[[WorkflowState], str] | None = None

    # For parallel nodes
    parallel_branches: list[str] = field(default_factory=list)


class WorkflowGraph:
    """
    A directed graph representing an agent workflow.

    Supports cycles (for reflection loops), conditional branching,
    parallel execution, checkpointing, and sub-workflow composition.
    """

    def __init__(self, name: str = "default", config=None):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.config = config
        self.nodes: dict[str, WorkflowNode] = {}
        self.edges: list[Edge] = []
        self.entry_node: str | None = None
        self._compiled = False

    def add_node(
        self,
        name: str,
        node_type: NodeType = NodeType.AGENT,
        handler: Callable | None = None,
        **kwargs,
    ) -> WorkflowGraph:
        """Add a node to the workflow graph. Returns self for chaining."""
        node = WorkflowNode(
            id=f"{self.id}_{name}",
            name=name,
            node_type=node_type,
            handler=handler,
            **kwargs,
        )
        self.nodes[name] = node
        if not self.entry_node:
            self.entry_node = name
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        condition: Callable[[WorkflowState], bool] | None = None,
        label: str = "",
        priority: int = 0,
    ) -> WorkflowGraph:
        """Add a directed edge between nodes. Returns self for chaining."""
        self.edges.append(Edge(
            source=source,
            target=target,
            condition=condition,
            label=label,
            priority=priority,
        ))
        return self

    def add_conditional_edges(
        self,
        source: str,
        condition_func: Callable[[WorkflowState], str],
        targets: dict[str, str],
    ) -> WorkflowGraph:
        """Add conditional branching from a node."""
        if source in self.nodes:
            self.nodes[source].condition_func = condition_func
        for label, target in targets.items():
            self.add_edge(source, target, label=label)
        return self

    def set_entry(self, node_name: str) -> WorkflowGraph:
        """Set the entry point of the workflow."""
        self.entry_node = node_name
        return self

    def compile(self) -> WorkflowGraph:
        """Validate and compile the workflow graph."""
        if not self.entry_node or self.entry_node not in self.nodes:
            raise ValueError(f"Invalid entry node: {self.entry_node}")

        # Validate all edges reference existing nodes
        for edge in self.edges:
            if edge.source not in self.nodes:
                raise ValueError(f"Edge source '{edge.source}' not found in nodes")
            if edge.target not in self.nodes:
                raise ValueError(f"Edge target '{edge.target}' not found in nodes")

        self._compiled = True
        return self

    def get_next_nodes(self, current: str, state: WorkflowState) -> list[str]:
        """Get the next nodes to execute based on current node and state."""
        node = self.nodes[current]
        outgoing = [e for e in self.edges if e.source == current]

        # Sort by priority (higher first)
        outgoing.sort(key=lambda e: e.priority, reverse=True)

        # Conditional branching
        if node.condition_func:
            branch = node.condition_func(state)
            matching = [e for e in outgoing if e.label == branch]
            return [e.target for e in matching] if matching else []

        # Standard edges with optional conditions
        next_nodes = []
        for edge in outgoing:
            if edge.condition is None or edge.condition(state):
                next_nodes.append(edge.target)

        return next_nodes

    def get_parallel_groups(self, nodes: list[str]) -> list[list[str]]:
        """Group nodes that can be executed in parallel."""
        # Nodes with no dependencies on each other can run in parallel
        groups: list[list[str]] = []
        seen = set()
        for name in nodes:
            if name not in seen:
                group = [name]
                seen.add(name)
                groups.append(group)
        return groups

    async def execute(
        self,
        initial_state: WorkflowState | None = None,
        max_steps: int = 100,
    ) -> WorkflowState:
        """
        Execute the workflow graph.

        Traverses the graph from the entry node, executing each node's handler
        and following edges based on conditions and state.
        """
        if not self._compiled:
            self.compile()

        state = initial_state or WorkflowState()
        current_nodes = [self.entry_node]
        step = 0

        while current_nodes and step < max_steps:
            step += 1

            # Execute current batch (parallel if multiple)
            if len(current_nodes) == 1:
                node_name = current_nodes[0]
                state = await self._execute_node(node_name, state)
            else:
                # Parallel execution
                tasks = [
                    self._execute_node(name, state)
                    for name in current_nodes
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, WorkflowState):
                        state.data.update(r.data)
                        state.messages.extend(r.messages)
                    elif isinstance(r, Exception):
                        state.add_error("parallel", str(r))

            # Determine next nodes
            next_nodes = []
            for name in current_nodes:
                nexts = self.get_next_nodes(name, state)
                next_nodes.extend(nexts)

            current_nodes = list(dict.fromkeys(next_nodes))  # Deduplicate preserving order

        state.metadata["total_steps"] = step
        state.metadata["completed"] = not current_nodes
        return state

    async def _execute_node(
        self, node_name: str, state: WorkflowState
    ) -> WorkflowState:
        """Execute a single workflow node."""
        node = self.nodes[node_name]
        node.status = NodeStatus.RUNNING
        start = time.time()

        try:
            if node.handler:
                state = await node.handler(state)
            elif node.node_type == NodeType.CHECKPOINT:
                state.checkpoint(label=node.name)
            elif node.node_type == NodeType.TRANSFORM:
                transform_fn = node.config.get("transform")
                if transform_fn:
                    state = transform_fn(state)

            node.status = NodeStatus.COMPLETED
        except Exception as e:
            node.status = NodeStatus.FAILED
            state.add_error(node_name, str(e))
            if node.retries < node.max_retries:
                node.retries += 1
                return await self._execute_node(node_name, state)
        finally:
            node.duration_ms = (time.time() - start) * 1000

        return state

    def visualize(self) -> str:
        """Generate ASCII visualization of the workflow graph."""
        lines = [f"Workflow: {self.name}", "=" * 40]
        for name, node in self.nodes.items():
            marker = "→ " if name == self.entry_node else "  "
            status = f"[{node.status.value}]" if node.status != NodeStatus.PENDING else ""
            lines.append(f"{marker}{name} ({node.node_type.value}) {status}")

        lines.append("")
        lines.append("Edges:")
        for edge in self.edges:
            cond = f" [{edge.label}]" if edge.label else ""
            lines.append(f"  {edge.source} → {edge.target}{cond}")

        return "\n".join(lines)

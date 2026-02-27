"""
NEXUS Tool Registry
====================
Central registry for all tools/skills available to agents.
Supports both native functions and MCP-connected tools.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from nexus.core.config import NexusConfig


@dataclass
class ToolDefinition:
    """Definition of a registered tool."""
    name: str
    description: str
    handler: Callable
    parameters: dict[str, Any] = field(default_factory=dict)
    required_params: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    requires_approval: bool = False
    timeout: float = 60.0
    source: str = "native"  # native, mcp, plugin


class ToolRegistry:
    """
    Central tool registry supporting native functions and MCP servers.

    Features:
    - Register Python functions as tools
    - Auto-extract parameter schemas from type hints
    - MCP client integration for external tools
    - Tag-based tool discovery
    - Capability matching for task routing
    - Tool usage analytics
    """

    def __init__(self, config: NexusConfig):
        self.config = config
        self._tools: dict[str, ToolDefinition] = {}
        self._mcp_clients: list[Any] = []
        self._usage_stats: dict[str, int] = {}

    def register(
        self,
        name: str,
        func: Callable,
        description: str = "",
        tags: list[str] | None = None,
        requires_approval: bool = False,
        **kwargs,
    ):
        """Register a native Python function as a tool."""
        # Auto-extract parameters from function signature
        sig = inspect.signature(func)
        parameters = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                type_map = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}
                param_type = type_map.get(param.annotation, "string")

            parameters[param_name] = {"type": param_type, "description": f"Parameter: {param_name}"}
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        tool_def = ToolDefinition(
            name=name,
            description=description or func.__doc__ or f"Tool: {name}",
            handler=func,
            parameters=parameters,
            required_params=required,
            tags=tags or [],
            requires_approval=requires_approval,
            timeout=kwargs.get("timeout", self.config.tool_timeout),
            source="native",
        )
        self._tools[name] = tool_def

    def register_mcp_client(self, client):
        """Register an MCP client for external tool access."""
        self._mcp_clients.append(client)

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name with given arguments."""
        if tool_name not in self._tools:
            # Try MCP clients
            for client in self._mcp_clients:
                if hasattr(client, "has_tool") and client.has_tool(tool_name):
                    return await client.call_tool(tool_name, arguments)
            return f"Tool '{tool_name}' not found"

        tool = self._tools[tool_name]
        self._usage_stats[tool_name] = self._usage_stats.get(tool_name, 0) + 1

        try:
            if asyncio.iscoroutinefunction(tool.handler):
                result = await asyncio.wait_for(
                    tool.handler(**arguments),
                    timeout=tool.timeout,
                )
            else:
                result = tool.handler(**arguments)
            return result
        except asyncio.TimeoutError:
            return f"Tool '{tool_name}' timed out after {tool.timeout}s"
        except Exception as e:
            return f"Tool '{tool_name}' error: {e}"

    def get_definitions(self, tool_names: list[str] | None = None) -> list[dict]:
        """Get OpenAI-compatible tool definitions for the model."""
        tools = self._tools.values()
        if tool_names:
            tools = [t for t in tools if t.name in tool_names]

        definitions = []
        for tool in tools:
            definitions.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": tool.parameters,
                        "required": tool.required_params,
                    },
                },
            })
        return definitions

    def match_tools(self, task_description: str) -> list[str]:
        """Find tools relevant to a task based on keyword matching."""
        task_lower = task_description.lower()
        matches = []
        for name, tool in self._tools.items():
            search_text = f"{tool.description} {' '.join(tool.tags)} {' '.join(tool.capabilities)}".lower()
            if any(word in search_text for word in task_lower.split() if len(word) > 3):
                matches.append(name)
        return matches

    def list_tools(self) -> list[dict]:
        """List all available tools."""
        return [
            {"name": t.name, "description": t.description, "tags": t.tags, "source": t.source}
            for t in self._tools.values()
        ]

    @property
    def count(self) -> int:
        return len(self._tools)

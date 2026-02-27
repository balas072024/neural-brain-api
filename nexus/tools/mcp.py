"""
NEXUS MCP (Model Context Protocol) Support
============================================
Implements both MCP Server and MCP Client for standards-compliant
tool interoperability following Anthropic's MCP specification.

MCP enables:
- Standardized tool discovery and invocation
- Cross-framework tool sharing
- Secure, scoped access to resources
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MCPToolDef:
    """MCP tool definition following the protocol spec."""
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPResource:
    """MCP resource (read-only data source)."""
    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"


@dataclass
class MCPPrompt:
    """MCP reusable prompt template."""
    name: str
    description: str = ""
    arguments: list[dict] = field(default_factory=list)
    template: str = ""


class MCPServer:
    """
    NEXUS MCP Server - Exposes tools as MCP-compatible endpoints.

    Follows the Model Context Protocol specification:
    - JSON-RPC 2.0 transport
    - Tool discovery via tools/list
    - Tool execution via tools/call
    - Resource access via resources/list, resources/read
    - Prompt templates via prompts/list, prompts/get
    """

    def __init__(self, name: str, registry=None, config=None, **kwargs):
        self.name = name
        self.registry = registry
        self.config = config
        self.version = "2025-11-25"  # MCP spec version
        self._resources: dict[str, MCPResource] = {}
        self._prompts: dict[str, MCPPrompt] = {}

    def add_resource(self, uri: str, name: str, description: str = "", mime_type: str = "text/plain"):
        """Register a resource endpoint."""
        self._resources[uri] = MCPResource(
            uri=uri, name=name, description=description, mime_type=mime_type
        )

    def add_prompt(self, name: str, template: str, description: str = "", arguments: list[dict] | None = None):
        """Register a prompt template."""
        self._prompts[name] = MCPPrompt(
            name=name, description=description, arguments=arguments or [], template=template
        )

    async def handle_request(self, request: dict) -> dict:
        """Handle an incoming MCP JSON-RPC request."""
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        handlers = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "prompts/list": self._handle_prompts_list,
            "prompts/get": self._handle_prompts_get,
        }

        handler = handlers.get(method)
        if not handler:
            return self._error_response(req_id, -32601, f"Method not found: {method}")

        try:
            result = await handler(params)
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as e:
            return self._error_response(req_id, -32000, str(e))

    async def _handle_initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": self.version,
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"subscribe": False, "listChanged": True},
                "prompts": {"listChanged": True},
            },
            "serverInfo": {"name": self.name, "version": "1.0.0"},
        }

    async def _handle_tools_list(self, params: dict) -> dict:
        tools = []
        if self.registry:
            for tool_def in self.registry._tools.values():
                tools.append({
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "inputSchema": {
                        "type": "object",
                        "properties": tool_def.parameters,
                        "required": tool_def.required_params,
                    },
                })
        return {"tools": tools}

    async def _handle_tools_call(self, params: dict) -> dict:
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if self.registry:
            result = await self.registry.execute(tool_name, arguments)
            return {
                "content": [{"type": "text", "text": str(result)}],
                "isError": False,
            }
        return {"content": [{"type": "text", "text": "No registry"}], "isError": True}

    async def _handle_resources_list(self, params: dict) -> dict:
        return {
            "resources": [
                {"uri": r.uri, "name": r.name, "description": r.description, "mimeType": r.mime_type}
                for r in self._resources.values()
            ]
        }

    async def _handle_resources_read(self, params: dict) -> dict:
        uri = params.get("uri", "")
        resource = self._resources.get(uri)
        if not resource:
            raise ValueError(f"Resource not found: {uri}")
        return {"contents": [{"uri": uri, "mimeType": resource.mime_type, "text": ""}]}

    async def _handle_prompts_list(self, params: dict) -> dict:
        return {
            "prompts": [
                {"name": p.name, "description": p.description, "arguments": p.arguments}
                for p in self._prompts.values()
            ]
        }

    async def _handle_prompts_get(self, params: dict) -> dict:
        name = params.get("name", "")
        prompt = self._prompts.get(name)
        if not prompt:
            raise ValueError(f"Prompt not found: {name}")
        return {
            "description": prompt.description,
            "messages": [{"role": "user", "content": {"type": "text", "text": prompt.template}}],
        }

    def _error_response(self, req_id, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


class MCPClient:
    """
    NEXUS MCP Client - Connect to external MCP servers.

    Enables agents to use tools from any MCP-compatible server,
    including tools from other frameworks (LangChain, Semantic Kernel, etc.).
    """

    def __init__(self, server_url: str, config=None, **kwargs):
        self.server_url = server_url
        self.config = config
        self._tools: dict[str, MCPToolDef] = {}
        self._connected = False
        self._request_id = 0

    async def connect(self):
        """Initialize connection to MCP server."""
        response = await self._send_request("initialize", {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "nexus-client", "version": "1.0.0"},
        })
        self._connected = True

        # Discover available tools
        tools_response = await self._send_request("tools/list", {})
        for tool_data in tools_response.get("tools", []):
            self._tools[tool_data["name"]] = MCPToolDef(
                name=tool_data["name"],
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema", {}),
            )

        return self

    def has_tool(self, name: str) -> bool:
        """Check if a tool is available."""
        return name in self._tools

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Call a tool on the MCP server."""
        if not self._connected:
            await self.connect()

        response = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        content = response.get("content", [])
        if content:
            return content[0].get("text", "")
        return ""

    def list_tools(self) -> list[dict]:
        """List available tools from this MCP server."""
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]

    async def _send_request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request to the MCP server."""
        self._request_id += 1
        # In production, this would use aiohttp/httpx for HTTP transport
        # or websockets for stdio/SSE transport
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        # Placeholder: actual transport implementation
        return {}

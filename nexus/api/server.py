"""
NEXUS API Server
==================
FastAPI-based REST API providing:
- OpenAI-compatible chat/completions endpoint
- Agent management endpoints
- Crew/workflow management
- Memory and knowledge graph queries
- Tool registry
- MCP server endpoints
- Health and status monitoring
"""

from __future__ import annotations

import time
import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from nexus.core.config import NexusConfig


# ===== Request/Response Models =====

class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""

class ChatRequest(BaseModel):
    model: str = "auto"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    tools: list[dict] | None = None
    # NEXUS extensions
    agent: str | None = None
    crew: str | None = None
    session_id: str = "default"
    reflection: bool = False

class ChatResponse(BaseModel):
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[dict] = []
    usage: dict = {}

class AgentCreateRequest(BaseModel):
    name: str
    role: str = "general"
    model: str | None = None
    tools: list[str] = []
    memory: bool = True
    reflection: bool = False

class CrewCreateRequest(BaseModel):
    name: str
    agents: list[str]
    strategy: str = "chain"

class TaskRequest(BaseModel):
    task: str
    agent: str | None = None
    crew: str | None = None
    stream: bool = False

class MemoryQueryRequest(BaseModel):
    query: str
    limit: int = 10
    layer: str = "all"  # all, working, episodic, semantic

class KnowledgeFactRequest(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0


# ===== API Server =====

def create_api(nexus_instance) -> FastAPI:
    """Create the NEXUS FastAPI application."""
    app = FastAPI(
        title="NEXUS AI Framework",
        description="Next-generation autonomous AI agent framework API",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    nx = nexus_instance

    # ===== Health & Status =====

    @app.get("/health")
    async def health():
        return {"status": "healthy", "framework": "NEXUS", "version": "1.0.0"}

    @app.get("/status")
    async def status():
        return nx.status

    # ===== OpenAI-Compatible Chat =====

    @app.post("/v1/chat/completions")
    @app.post("/api/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        """OpenAI-compatible chat completions endpoint with NEXUS extensions."""
        start_time = time.time()

        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        task = messages[-1]["content"] if messages else ""

        if request.crew:
            result = await nx.run(task, crew=request.crew)
        elif request.agent:
            result = await nx.run(task, agent=request.agent)
        else:
            result = await nx.chat(
                message=task,
                session_id=request.session_id,
            )

        return ChatResponse(
            id=f"nexus-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": result.output},
                "finish_reason": "stop",
            }],
            usage={
                "total_tokens": result.tokens_used,
                "completion_tokens": result.tokens_used,
                "prompt_tokens": 0,
            },
        )

    # ===== Agent Management =====

    @app.post("/api/agents")
    async def create_agent(request: AgentCreateRequest):
        agent = nx.agent(
            name=request.name,
            role=request.role,
            model=request.model,
            tools=request.tools,
            memory=request.memory,
            reflection=request.reflection,
        )
        return agent.info

    @app.get("/api/agents")
    async def list_agents():
        return nx.agent_manager.list_agents()

    @app.get("/api/agents/{name}")
    async def get_agent(name: str):
        try:
            agent = nx.agent_manager.get(name)
            return agent.info
        except KeyError:
            raise HTTPException(404, f"Agent '{name}' not found")

    @app.delete("/api/agents/{name}")
    async def delete_agent(name: str):
        nx.agent_manager.remove(name)
        return {"deleted": name}

    # ===== Crew Management =====

    @app.post("/api/crews")
    async def create_crew(request: CrewCreateRequest):
        crew = nx.crew(
            name=request.name,
            agents=request.agents,
            strategy=request.strategy,
        )
        return crew.info

    @app.get("/api/crews")
    async def list_crews():
        return {name: crew.info for name, crew in nx._crews.items()}

    # ===== Task Execution =====

    @app.post("/api/tasks")
    async def execute_task(request: TaskRequest):
        result = await nx.run(
            task=request.task,
            agent=request.agent,
            crew=request.crew,
            stream=request.stream,
        )
        return {
            "success": result.success,
            "output": result.output,
            "duration_ms": result.total_duration_ms,
            "tokens_used": result.tokens_used,
            "cost_estimate": result.cost_estimate,
        }

    # ===== Memory =====

    @app.post("/api/memory/query")
    async def query_memory(request: MemoryQueryRequest):
        results = await nx.memory_manager.search_relevant(
            request.query, limit=request.limit
        )
        return {"results": results}

    @app.post("/api/memory/knowledge")
    async def add_knowledge(request: KnowledgeFactRequest):
        nx.memory_manager.semantic.learn_fact(
            request.subject, request.predicate, request.object,
            confidence=request.confidence,
        )
        return {"stored": True}

    @app.get("/api/memory/stats")
    async def memory_stats():
        return {
            "total_entries": nx.memory_manager.total_entries,
            "episodic_count": nx.memory_manager.episodic.count,
            "semantic_stats": nx.memory_manager.semantic.stats,
        }

    # ===== Knowledge Graph =====

    @app.get("/api/knowledge/query")
    async def knowledge_query(q: str):
        results = nx.memory_manager.semantic.recall(q)
        return {"results": results}

    @app.get("/api/knowledge/explore/{concept}")
    async def knowledge_explore(concept: str, depth: int = 2):
        return nx.memory_manager.semantic.explore(concept, depth=depth)

    @app.get("/api/knowledge/stats")
    async def knowledge_stats():
        return nx.memory_manager.semantic.stats

    # ===== Tools =====

    @app.get("/api/tools")
    async def list_tools():
        return nx.tool_registry.list_tools()

    @app.post("/api/tools/{name}/execute")
    async def execute_tool(name: str, request: Request):
        body = await request.json()
        result = await nx.tool_registry.execute(name, body)
        return {"result": result}

    # ===== Models =====

    @app.get("/api/models")
    @app.get("/v1/models")
    async def list_models():
        models = nx.model_router.available_models
        return {
            "object": "list",
            "data": [
                {"id": m, "object": "model", "owned_by": "nexus"}
                for m in models
            ],
        }

    # ===== MCP Server Endpoints =====

    @app.post("/mcp")
    async def mcp_endpoint(request: Request):
        """Handle MCP JSON-RPC requests."""
        body = await request.json()
        server = nx.mcp_server("default")
        response = await server.handle_request(body)
        return response

    # ===== Reflection & Analytics =====

    @app.get("/api/reflection/meta")
    async def meta_reflection():
        return await nx.reflection_engine.meta_reflect()

    @app.get("/api/audit")
    async def audit_log(n: int = 50):
        return nx.audit.get_recent(n)

    return app

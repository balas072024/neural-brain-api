"""
KaasAI Neural Brain v2.1 — REST API Gateway + Ensemble Engine
Central LLM endpoint for all Kaashmikhaa products + OpenClaw fallback.
84+ models, 15 providers, 6 categories.
NEW: Ensemble mode — multi-model orchestration with specialist routing.
"""
import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

# ── Load .env FIRST (before any imports that use env vars) ──
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_path):
    with open(env_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key and value:
                    os.environ.setdefault(key, value)

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.brain import (
    NeuralBrain, CompletionRequest, CompletionResponse, EmbeddingRequest,
    ProviderType, ModelCapability, RoutingStrategy, brain
)
from core.ensemble import EnsembleEngine

logger = logging.getLogger("neural-brain.api")

# ── Initialize Ensemble Engine ──
ensemble = EnsembleEngine(brain)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        models = await brain.discover_ollama_models()
        if models:
            logger.info(f"Discovered {len(models)} Ollama models: {models}")
    except Exception:
        pass
    status = brain.get_status()
    logger.info(f"Neural Brain v3.0 started — {status['total_models']} models, {status['total_providers']} providers")
    logger.info(f"Categories: {status['categories']}")
    logger.info(f"Ensemble Engine active — LOCAL-FIRST specialist routing enabled")
    yield


app = FastAPI(
    title="KaasAI Neural Brain",
    description="Local-First LLM Gateway v3.0 — 100+ models, 15 providers, local-first routing. "
                "Zero API keys needed. Runs on any hardware with Ollama. "
                "Ensemble multi-model orchestration with auto-detection of best local models.",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ═══ Request/Response Models ═══

class ChatRequest(BaseModel):
    messages: List[Dict]
    model: str = ""
    provider: str = ""
    system: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    tools: List[Dict] = []
    product: str = ""
    routing_strategy: str = "local_first"
    required_capabilities: List[str] = []
    max_cost_per_request: float = 0.0
    tags: Dict[str, str] = {}

class EnsembleRequest(BaseModel):
    messages: List[Dict]
    system: str = ""
    temperature: float = 0.7
    max_tokens: int = 8192
    mode: str = "smart"
    verify: bool = False

class EmbedRequest(BaseModel):
    texts: List[str]
    model: str = ""
    provider: str = ""
    dimensions: int = 0
    product: str = ""

class ProviderSetup(BaseModel):
    provider: str
    api_key: str = ""
    base_url: str = ""
    enabled: bool = True

class ModelAdd(BaseModel):
    model_id: str
    provider: str
    name: str
    endpoint: str = ""
    context_window: int = 128000
    max_output: int = 4096
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    capabilities: List[str] = []
    is_local: bool = False
    category: str = "general"

class ProductRegister(BaseModel):
    product_name: str
    default_model: str = ""
    routing_strategy: str = "local_first"
    required_capabilities: List[str] = []
    max_tokens: int = 4096
    temperature: float = 0.7
    description: str = ""

class ModelToggle(BaseModel):
    enabled: bool


# ═══ Health & Status ═══

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "kaasai-neural-brain",
        "version": "3.0.0",
        "providers": sum(1 for p in brain.providers.values() if p.enabled),
        "models": len(brain.models),
        "ensemble": True,
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.get("/api/status")
async def status():
    base = brain.get_status()
    base["ensemble"] = ensemble.get_stats()
    return base

@app.get("/api/usage")
async def usage():
    return brain.usage.get_summary()

@app.get("/api/usage/{product}")
async def product_usage(product: str):
    return brain.usage.get_product_usage(product)


# ═══════════════════════════════════════════════════════════════
#  Core: Chat Completion (OpenAI-compatible) + Ensemble Support
# ═══════════════════════════════════════════════════════════════

ENSEMBLE_MODEL_IDS = {
    "ensemble", "ensemble/smart", "ensemble/consensus", "ensemble/chain",
    "ensemble/fastest", "ensemble/strongest",
    "neural-brain-auto", "neural-brain-ensemble",
}

@app.post("/api/v1/chat/completions")
async def chat_completions(request: Request, body: ChatRequest):
    # Log request for debugging OpenClaw integration
    msg_preview = str(body.messages)[:300] if body.messages else 'EMPTY'
    logger.warning(f"[INCOMING] model={body.model!r} msgs={len(body.messages)} stream={body.stream} temp={body.temperature} max_tokens={body.max_tokens} routing={body.routing_strategy}")
    logger.warning(f"[INCOMING MSGS] {msg_preview}")
    model_lower = (body.model or "").lower().strip()

    if model_lower in ENSEMBLE_MODEL_IDS or model_lower.startswith("ensemble"):
        if "/" in model_lower:
            mode = model_lower.split("/", 1)[1]
        elif model_lower in ("neural-brain-auto", "neural-brain-ensemble"):
            mode = "smart"
        else:
            mode = "smart"

        # Handle streaming for ensemble (OpenClaw sends stream=true by default)
        if body.stream:
            async def _ensemble_stream():
                try:
                    result = await ensemble.complete(
                        messages=body.messages, system=body.system,
                        temperature=body.temperature, max_tokens=body.max_tokens,
                        mode=mode,
                    )
                    logger.warning(f"[ENSEMBLE STREAM] model={result.model_used} type={result.query_type} len={len(result.content)} latency={result.latency_ms:.0f}ms")
                    # Send content as a single SSE chunk
                    data = json.dumps({"choices": [{"delta": {"role": "assistant", "content": result.content}}]})
                    yield f"data: {data}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error(f"[ENSEMBLE STREAM ERROR] {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return StreamingResponse(
                _ensemble_stream(), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            result = await ensemble.complete(
                messages=body.messages, system=body.system,
                temperature=body.temperature, max_tokens=body.max_tokens,
                mode=mode,
            )
            logger.warning(f"[ENSEMBLE RESULT] model={result.model_used} type={result.query_type} len={len(result.content)} latency={result.latency_ms:.0f}ms")
            return {
                "id": f"ensemble-{id(result)}", "object": "chat.completion",
                "model": result.model_used,
                "provider": "ensemble",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result.content},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "cost_usd": 0.0,
                "latency_ms": round(result.latency_ms, 1),
                "cached": False,
                "ensemble_info": {
                    "query_type": result.query_type,
                    "confidence": round(result.confidence, 3),
                    "models_consulted": result.models_consulted,
                    "verified": result.verified,
                },
            }
        except Exception as e:
            raise HTTPException(500, f"Ensemble failed: {str(e)}")

    req = CompletionRequest(
        messages=body.messages, model=body.model, provider=body.provider,
        system=body.system, temperature=body.temperature, max_tokens=body.max_tokens,
        stream=body.stream, tools=body.tools, product=body.product, tags=body.tags,
    )
    try:
        req.routing_strategy = RoutingStrategy(body.routing_strategy)
    except ValueError:
        req.routing_strategy = RoutingStrategy.LOCAL_FIRST
    for cap_str in body.required_capabilities:
        try:
            req.required_capabilities.append(ModelCapability(cap_str))
        except ValueError:
            pass
    req.max_cost_per_request = body.max_cost_per_request

    if body.stream:
        return StreamingResponse(
            _stream_response(req), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        resp = await brain.complete(req)
        return {
            "id": resp.id, "object": "chat.completion", "model": resp.model,
            "provider": resp.provider,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": resp.content},
                        "finish_reason": resp.finish_reason}],
            "usage": resp.usage, "cost_usd": resp.cost,
            "latency_ms": round(resp.latency_ms, 1), "cached": resp.cached,
        }
    except Exception as e:
        raise HTTPException(500, f"Completion failed: {str(e)}")


async def _stream_response(req: CompletionRequest):
    try:
        async for chunk in brain.complete_stream(req):
            data = json.dumps({"choices": [{"delta": {"content": chunk}}]})
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# ═══ Ensemble-Specific Endpoints ═══

@app.post("/api/v1/ensemble")
async def ensemble_complete(body: EnsembleRequest):
    try:
        result = await ensemble.complete(
            messages=body.messages, system=body.system,
            temperature=body.temperature, max_tokens=body.max_tokens,
            mode=body.mode, verify=body.verify,
        )
        return {
            "content": result.content,
            "model_used": result.model_used,
            "query_type": result.query_type,
            "confidence": round(result.confidence, 3),
            "latency_ms": round(result.latency_ms, 1),
            "models_consulted": result.models_consulted,
            "verified": result.verified,
            "verification_model": result.verification_model,
        }
    except Exception as e:
        raise HTTPException(500, f"Ensemble failed: {str(e)}")

@app.get("/api/v1/ensemble/classify")
async def classify_query(text: str):
    result = ensemble.classify_query([{"role": "user", "content": text}])
    return {
        "query_type": result.query_type,
        "confidence": round(result.confidence, 3),
        "scores": {k: round(v, 2) for k, v in result.scores.items()},
        "specialist": result.specialist,
        "fallbacks": result.fallbacks,
    }

@app.get("/api/v1/ensemble/stats")
async def ensemble_stats():
    return ensemble.get_stats()


# ═══ Embeddings ═══

@app.post("/api/v1/embeddings")
async def embeddings(body: EmbedRequest):
    req = EmbeddingRequest(texts=body.texts, model=body.model,
                          provider=body.provider, dimensions=body.dimensions, product=body.product)
    try:
        resp = await brain.embed(req)
        return {"object": "list",
                "data": [{"object": "embedding", "embedding": emb, "index": i}
                         for i, emb in enumerate(resp.embeddings)],
                "model": resp.model, "usage": resp.usage}
    except Exception as e:
        raise HTTPException(500, str(e))


# ═══ Model Management ═══

@app.get("/api/models")
async def list_models(provider: str = None, capability: str = None,
                      local_only: bool = False, category: str = None):
    return brain.list_models(provider=provider, capability=capability,
                            local_only=local_only, category=category)

@app.get("/api/models/categories")
async def model_categories():
    categories = {}
    for m in brain.models.values():
        cat = m.category or "general"
        if cat not in categories:
            categories[cat] = []
        categories[cat].append({
            "id": m.id, "name": m.name, "provider": m.provider.value,
            "is_local": m.is_local, "cost": m.cost_per_1k_input + m.cost_per_1k_output,
            "capabilities": [c.value for c in m.capabilities],
        })
    return categories

@app.get("/api/models/{model_id:path}")
async def get_model(model_id: str):
    model = brain.models.get(model_id)
    if not model:
        raise HTTPException(404, "Model not found")
    return {
        "id": model.id, "name": model.name, "provider": model.provider.value,
        "enabled": model.enabled, "is_local": model.is_local, "category": model.category,
        "context_window": model.context_window, "max_output": model.max_output,
        "cost_input": model.cost_per_1k_input, "cost_output": model.cost_per_1k_output,
        "capabilities": [c.value for c in model.capabilities],
        "stats": {"total_requests": model.total_requests,
                  "avg_latency_ms": round(model.avg_latency_ms, 1),
                  "success_rate": model.success_rate, "total_cost": round(model.total_cost, 4)},
    }

@app.patch("/api/models/{model_id:path}")
async def toggle_model(model_id: str, body: ModelToggle):
    model = brain.models.get(model_id)
    if not model:
        raise HTTPException(404, "Model not found")
    model.enabled = body.enabled
    return {"model": model_id, "enabled": model.enabled}

@app.post("/api/models")
async def add_model(body: ModelAdd):
    caps = []
    for c in body.capabilities:
        try: caps.append(ModelCapability(c))
        except ValueError: pass
    model = brain.add_custom_model(
        model_id=body.model_id, provider=body.provider, name=body.name,
        endpoint=body.endpoint, context_window=body.context_window,
        max_output=body.max_output, cost_per_1k_input=body.cost_per_1k_input,
        cost_per_1k_output=body.cost_per_1k_output, capabilities=caps,
        is_local=body.is_local, category=body.category,
    )
    return {"id": model.id, "name": model.name, "added": True}


# ═══ Provider Management ═══

@app.get("/api/providers")
async def list_providers():
    return {p.value: {"name": c.name, "enabled": c.enabled, "is_local": c.is_local,
                      "models": len(c.models), "base_url": c.base_url}
            for p, c in brain.providers.items()}

@app.post("/api/providers")
async def setup_provider(body: ProviderSetup):
    config = brain.configure_provider(body.provider, api_key=body.api_key, base_url=body.base_url)
    return {"provider": body.provider, "configured": True, "models": len(config.models)}

@app.post("/api/providers/ollama/discover")
async def discover_ollama():
    models = await brain.discover_ollama_models()
    return {"discovered": len(models), "models": models}


# ═══ Product Registration ═══

@app.get("/api/products")
async def list_products():
    return brain.product_presets

@app.post("/api/products")
async def register_product(body: ProductRegister):
    preset = {"default_model": body.default_model, "description": body.description,
              "max_tokens": body.max_tokens, "temperature": body.temperature}
    try: preset["routing_strategy"] = RoutingStrategy(body.routing_strategy)
    except ValueError: preset["routing_strategy"] = RoutingStrategy.LOCAL_FIRST
    caps = []
    for c in body.required_capabilities:
        try: caps.append(ModelCapability(c))
        except ValueError: pass
    preset["required_capabilities"] = caps
    brain.register_product(body.product_name, preset)
    return {"product": body.product_name, "registered": True}


# ═══ Cache ═══

@app.get("/api/cache/stats")
async def cache_stats():
    return brain.cache.stats()

@app.post("/api/cache/clear")
async def clear_cache():
    brain.cache.cache.clear()
    brain.cache.hits = 0
    brain.cache.misses = 0
    return {"cleared": True}


# ═══ WebSocket ═══

@app.websocket("/ws")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = json.loads(await ws.receive_text())
            if data.get("type") == "chat":
                req = CompletionRequest(
                    messages=data.get("messages", []), model=data.get("model", ""),
                    product=data.get("product", ""), system=data.get("system", ""),
                    temperature=data.get("temperature", 0.7),
                    max_tokens=data.get("max_tokens", 4096), stream=True,
                )
                async for chunk in brain.complete_stream(req):
                    await ws.send_text(json.dumps({"type": "chunk", "content": chunk}))
                await ws.send_text(json.dumps({"type": "done"}))
            elif data.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass


# ═══ Serve Frontend ═══
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200, log_level="info")

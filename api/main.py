"""
KaasAI Neural Brain v4.0 — Self-Learning Local-First LLM Gateway
Central LLM endpoint for all Kaashmikhaa products + OpenClaw fallback.
100+ models, 15 providers, 6 categories.

v4.0 FEATURES:
- Self-learning: adapts routing based on real performance data
- Quantization: auto-detects and prefers compressed models for speed
- Distillation: larger models teach smaller ones
- Sub-2-second responses: aggressive caching, warmup, connection pooling
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
    # Discover what's already installed
    try:
        models = await brain.discover_ollama_models()
        if models:
            logger.info(f"Discovered {len(models)} Ollama models")
    except Exception:
        pass

    # Scan model sizes/quantization for optimization recommendations
    if brain.quantization:
        try:
            await brain.quantization.scan_models()
        except Exception:
            pass

    # Preload the fast model immediately for instant first response
    async def _warmup():
        try:
            warmed = await brain.warmup_models()
            if warmed:
                logger.info(f"Fast model preloaded: {warmed}")
        except Exception:
            pass
    asyncio.create_task(_warmup())

    # Auto-pull essential models if not installed (runs in background)
    if os.getenv("AUTO_PULL_MODELS", "true").lower() == "true":
        async def _background_pull():
            pulled = await brain.auto_pull_models()
            if pulled:
                logger.info(f"Auto-pulled {len(pulled)} models: {pulled}")
                await brain.discover_ollama_models()  # Refresh after pull
                # Re-scan quantization after pulling
                if brain.quantization:
                    await brain.quantization.scan_models()
        asyncio.create_task(_background_pull())

    # Background task: periodic learning data save
    async def _periodic_save():
        while True:
            await asyncio.sleep(300)  # Save every 5 minutes
            if brain.learning:
                try:
                    brain.learning.save()
                except Exception:
                    pass
    asyncio.create_task(_periodic_save())

    status = brain.get_status()
    logger.info(f"Neural Brain v4.0 started — {status['total_models']} models, {status['total_providers']} providers")
    logger.info(f"Categories: {status['categories']}")
    v4 = status.get("v4_engines", {})
    logger.info(f"v4.0 engines: learning={v4.get('self_learning')}, quant={v4.get('quantization')}, distill={v4.get('distillation')}")
    logger.info(f"Performance target: <2s quality responses")
    yield
    # Cleanup: save learning data and close connections
    await brain.close()


app = FastAPI(
    title="KaasAI Neural Brain",
    description="Self-Learning Local-First LLM Gateway v4.0 — 100+ models, 15 providers. "
                "Zero API keys needed. Self-learning adaptive routing, model quantization/compression, "
                "knowledge distillation. Sub-2-second quality responses on any hardware.",
    version="4.0.0",
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
    speed: str = ""  # "fast", "medium", "thinking" — auto-selects model if no model specified
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
        "version": "4.0.0",
        "providers": sum(1 for p in brain.providers.values() if p.enabled),
        "models": len(brain.models),
        "ensemble": True,
        "self_learning": brain.learning is not None,
        "quantization": brain.quantization is not None,
        "distillation": brain.distillation is not None,
        "performance_target": "<2s",
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


# ═══ Simple Chat Endpoint (shortcut) ═══

# Speed tier → model preferences (tried in order, first installed wins)
SPEED_TIERS = {
    "fast": [
        "ollama/qwen3:4b", "ollama/llama3.2:3b", "ollama/phi3:mini",
        "ollama/llama3.2", "ollama/phi3", "ollama/phi3:latest",
        "ollama/llama3.2:latest", "ollama/qwen3:1.7b",
    ],
    "medium": [
        "ollama/qwen3:8b", "ollama/deepseek-r1:8b", "ollama/qwen2.5-coder:7b",
        "ollama/gemma3:12b", "ollama/phi4:latest", "ollama/qwen2.5:7b",
        "ollama/llama3.1:latest", "ollama/llama3:latest",
        "ollama/deepseek-r1:latest",
    ],
    "thinking": [
        "ollama/deepseek-r1:32b", "ollama/qwen3:32b", "ollama/qwen3-coder:30b",
        "ollama/qwen2.5-coder:32b", "ollama/llama3.3:latest", "ollama/llama3.1:70b",
        "ollama/codellama:34b", "ollama/mistral-small:latest",
        "ollama/mistral-small3.1:latest", "ollama/glm-4.7-flash:latest",
    ],
}

class SimpleChatRequest(BaseModel):
    message: str
    model: str = ""
    speed: str = "fast"  # "fast" (<2s), "medium" (2-10s), "thinking" (deep reasoning)
    system: str = ""
    temperature: float = 0.7
    max_tokens: int = 512  # Default 512 for fast responses (use higher for thinking)

def _is_installed(model_id: str) -> bool:
    """Check if a model is actually installed in Ollama (not just registered)."""
    return model_id in brain._installed_ollama_models

@app.post("/api/v1/chat")
async def simple_chat(body: SimpleChatRequest):
    """Simple chat endpoint with speed tier selection.
    Speed options: 'fast' (<2s, small models), 'medium' (balanced), 'thinking' (deep reasoning, large models)."""
    # Resolve model name: "qwen3:4b" -> "ollama/qwen3:4b"
    model = body.model.strip()
    if model and model not in brain.models:
        if f"ollama/{model}" in brain.models:
            model = f"ollama/{model}"

    # If no model specified, pick from speed tier (only installed models)
    if not model:
        tier = body.speed.lower() if body.speed else "fast"
        tier_models = SPEED_TIERS.get(tier, SPEED_TIERS["fast"])
        for candidate in tier_models:
            if _is_installed(candidate):
                model = candidate
                break
        # Fallback: pick any actually installed local model
        if not model:
            for m_id in brain._installed_ollama_models:
                if m_id in brain.models and brain.models[m_id].enabled:
                    model = m_id
                    break

    # Auto-set max_tokens based on speed tier if user didn't override
    max_tokens = body.max_tokens
    if max_tokens == 512:  # Default, auto-adjust per tier
        tier = body.speed.lower() if body.speed else "fast"
        max_tokens = {"fast": 256, "medium": 512, "thinking": 2048}.get(tier, 512)

    messages = [{"role": "user", "content": body.message}]
    req = CompletionRequest(
        messages=messages, model=model, system=body.system,
        temperature=body.temperature, max_tokens=max_tokens,
    )
    req.routing_strategy = RoutingStrategy.LOCAL_FIRST
    try:
        resp = await brain.complete(req)
        return {
            "response": resp.content,
            "model": resp.model,
            "provider": resp.provider,
            "speed_tier": body.speed or "fast",
            "latency_ms": round(resp.latency_ms, 1),
            "cached": resp.cached,
        }
    except Exception as e:
        raise HTTPException(500, f"Chat failed: {str(e)}")


@app.get("/api/v1/chat/speeds")
async def chat_speed_tiers():
    """List available speed tiers and which models are actually installed."""
    result = {}
    for tier, models in SPEED_TIERS.items():
        installed = [m for m in models if _is_installed(m)]
        result[tier] = {
            "description": {"fast": "Quick responses <2s (small models)",
                            "medium": "Balanced quality + speed (7-12B models)",
                            "thinking": "Deep reasoning (large 30B+ models)"}[tier],
            "active_model": installed[0] if installed else None,
            "installed_models": installed,
        }
    return result


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

    # Resolve model name: "qwen3:4b" -> "ollama/qwen3:4b"
    resolved_model = body.model.strip()
    if resolved_model and resolved_model not in brain.models:
        if f"ollama/{resolved_model}" in brain.models:
            resolved_model = f"ollama/{resolved_model}"

    # Speed tier: auto-select model if none specified (only installed models)
    if not resolved_model and body.speed:
        tier_models = SPEED_TIERS.get(body.speed.lower(), [])
        for candidate in tier_models:
            if _is_installed(candidate):
                resolved_model = candidate
                break

    req = CompletionRequest(
        messages=body.messages, model=resolved_model, provider=body.provider,
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


# ═══════════════════════════════════════════════════════════════
#  v4.0: Self-Learning Engine
# ═══════════════════════════════════════════════════════════════

@app.get("/api/v1/learning/insights")
async def learning_insights():
    """Get self-learning performance insights and adaptive rankings."""
    if not brain.learning:
        return {"error": "Self-learning engine not available"}
    return brain.learning.get_insights()

@app.get("/api/v1/learning/model/{model_id:path}")
async def learning_model_report(model_id: str):
    """Get detailed learning report for a specific model."""
    if not brain.learning:
        return {"error": "Self-learning engine not available"}
    return brain.learning.get_model_report(model_id)

@app.get("/api/v1/learning/rankings/{query_type}")
async def learning_rankings(query_type: str):
    """Get learned model rankings for a query type."""
    if not brain.learning:
        return {"error": "Self-learning engine not available"}
    available = [m.id for m in brain.models.values() if m.is_local and m.enabled]
    ranked = brain.learning.get_ranked_models(query_type, available)
    return {"query_type": query_type, "rankings": [{"model": m, "score": round(s, 3)} for m, s in ranked]}

class FeedbackRequest(BaseModel):
    model_id: str
    query_type: str = "general"
    positive: bool
    request_id: str = ""
    details: str = ""

@app.post("/api/v1/learning/feedback")
async def submit_feedback(body: FeedbackRequest):
    """Submit user feedback to improve model routing."""
    if not brain.learning:
        return {"error": "Self-learning engine not available"}
    brain.learning.record_feedback(
        model_id=body.model_id, query_type=body.query_type,
        positive=body.positive, request_id=body.request_id, details=body.details,
    )
    return {"recorded": True, "model": body.model_id, "positive": body.positive}

@app.post("/api/v1/learning/save")
async def save_learning():
    """Persist learning data to disk immediately."""
    if not brain.learning:
        return {"error": "Self-learning engine not available"}
    brain.learning.save()
    return {"saved": True}

@app.post("/api/v1/learning/reset")
async def reset_learning():
    """Reset all learning data (start fresh)."""
    if not brain.learning:
        return {"error": "Self-learning engine not available"}
    brain.learning.reset()
    return {"reset": True}


# ═══════════════════════════════════════════════════════════════
#  v4.0: Quantization & Compression Manager
# ═══════════════════════════════════════════════════════════════

@app.get("/api/v1/quantization/report")
async def quantization_report():
    """Get model space usage and compression recommendations."""
    if not brain.quantization:
        return {"error": "Quantization manager not available"}
    await brain.quantization.scan_models()
    return brain.quantization.get_space_report()

@app.get("/api/v1/quantization/models")
async def quantization_models():
    """List all models with size and quantization info."""
    if not brain.quantization:
        return {"error": "Quantization manager not available"}
    info = await brain.quantization.scan_models()
    return {
        model_id: {
            "size_gb": i.size_gb, "params": i.parameter_count,
            "quant": i.quantization, "family": i.family,
        }
        for model_id, i in info.items()
    }

@app.get("/api/v1/quantization/recommend")
async def quantization_recommend(vram_gb: float = 0):
    """Get quantization recommendations based on available VRAM."""
    if not brain.quantization:
        return {"error": "Quantization manager not available"}
    await brain.quantization.scan_models()
    recs = brain.quantization.get_compression_recommendations(vram_gb)
    return {
        "recommendations": [
            {
                "model": r.model_id,
                "current": f"{r.current_size_gb}GB ({r.current_quant})",
                "recommended": r.recommended_model,
                "new_size": f"{r.estimated_size_gb}GB",
                "savings": f"{r.savings_gb}GB ({r.savings_percent}%)",
                "quality_kept": f"{r.quality_retention*100:.0f}%",
                "reason": r.reason,
            }
            for r in recs
        ]
    }

class CompressRequest(BaseModel):
    model: str
    target_quant: str = "q4_K_M"

@app.post("/api/v1/quantization/compress")
async def compress_model(body: CompressRequest):
    """Pull a quantized version of a model to save space."""
    if not brain.quantization:
        return {"error": "Quantization manager not available"}
    result = await brain.quantization.compress_model(body.model, body.target_quant)
    if result.get("success"):
        # Re-discover after compression
        await brain.discover_ollama_models()
    return result

@app.get("/api/v1/quantization/optimal")
async def optimal_quant(params: str = "8b", vram_gb: float = 8):
    """Get the optimal quantization level for a model size and VRAM."""
    if not brain.quantization:
        return {"error": "Quantization manager not available"}
    optimal = brain.quantization.get_optimal_quant_for_vram(params, vram_gb)
    return {"params": params, "vram_gb": vram_gb, "optimal_quant": optimal}

@app.get("/api/v1/quantization/status")
async def quantization_status():
    """Show quantization level and optimization status of every installed model."""
    if not brain.quantization:
        return {"error": "Quantization manager not available"}
    await brain.quantization.scan_models()
    status = brain.quantization.get_quantization_status()
    summary = {"optimal": 0, "good": 0, "compressed": 0, "large": 0, "uncompressed": 0, "unknown": 0}
    for m in status:
        summary[m["optimization_status"]] = summary.get(m["optimization_status"], 0) + 1
    return {"total_models": len(status), "summary": summary, "models": status}

class AutoOptimizeRequest(BaseModel):
    target_quant: str = "q4_K_M"
    dry_run: bool = True  # Set to false to actually pull quantized models

@app.post("/api/v1/quantization/auto-optimize")
async def auto_optimize(body: AutoOptimizeRequest):
    """Auto-optimize all models by pulling quantized versions.
    Set dry_run=false to actually download quantized variants."""
    if not brain.quantization:
        return {"error": "Quantization manager not available"}
    result = await brain.quantization.auto_optimize(
        target_quant=body.target_quant, dry_run=body.dry_run,
    )
    if not body.dry_run:
        await brain.discover_ollama_models()
    return result


# ═══════════════════════════════════════════════════════════════
#  v4.0: Knowledge Distillation
# ═══════════════════════════════════════════════════════════════

@app.get("/api/v1/distillation/stats")
async def distillation_stats():
    """Get distillation engine status and dataset info."""
    if not brain.distillation:
        return {"error": "Distillation engine not available"}
    return brain.distillation.get_stats()

class DistillRequest(BaseModel):
    teacher_model: str
    student_model: str
    domain: str = "general"
    num_samples: int = 30

@app.post("/api/v1/distillation/start")
async def start_distillation(body: DistillRequest):
    """Start a distillation job: teacher model teaches student model."""
    if not brain.distillation:
        return {"error": "Distillation engine not available"}
    job = await brain.distillation.start_job(
        teacher_model=body.teacher_model,
        student_model=body.student_model,
        domain=body.domain,
        num_samples=body.num_samples,
    )
    return {"job_id": job.id, "status": job.status, "target": job.target_model_name}

@app.get("/api/v1/distillation/jobs")
async def distillation_jobs():
    """List all distillation jobs."""
    if not brain.distillation:
        return {"error": "Distillation engine not available"}
    return brain.distillation.get_all_jobs()

@app.get("/api/v1/distillation/jobs/{job_id}")
async def distillation_job_status(job_id: str):
    """Get status of a specific distillation job."""
    if not brain.distillation:
        return {"error": "Distillation engine not available"}
    status = brain.distillation.get_job_status(job_id)
    if not status:
        raise HTTPException(404, "Job not found")
    return status

class CollectRequest(BaseModel):
    domain: str = "general"
    num_samples: int = 20
    teacher_model: str = ""

@app.post("/api/v1/distillation/collect")
async def collect_distillation_data(body: CollectRequest):
    """Collect distillation training data from a teacher model."""
    if not brain.distillation:
        return {"error": "Distillation engine not available"}
    collected = await brain.distillation.collect_dataset(
        domain=body.domain, num_samples=body.num_samples,
        teacher_model=body.teacher_model or None,
    )
    return {"domain": body.domain, "samples_collected": collected}

@app.get("/api/v1/distillation/datasets")
async def distillation_datasets():
    """List available distillation datasets."""
    if not brain.distillation:
        return {"error": "Distillation engine not available"}
    return {
        domain: {
            "samples": len(samples),
            "avg_quality": round(sum(s.quality_score for s in samples) / len(samples), 3) if samples else 0,
            "teachers": list(set(s.teacher_model for s in samples)),
        }
        for domain, samples in brain.distillation.datasets.items()
    }


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

"""
KaasAI Neural Brain v3.0 — Local-First Unified LLM Gateway & Router
Central AI backbone — zero API keys needed. 100% functional with just Ollama.

LOCAL-FIRST ARCHITECTURE:
- 45+ local Ollama models across all categories (code, reasoning, vision, audio, embedding)
- Tiered model selection: auto-picks best model for your hardware (2GB → 48GB VRAM)
- Smart routing: LOCAL_FIRST strategy ensures local models are always preferred
- Ensemble engine: multi-model orchestration using only local models
- Local embeddings: Qwen3-Embedding, Nomic, MxBAI — no OpenAI needed

MODEL TIERS (Local):
- Tier 1 (1-4B): Qwen3 1.7B/4B, Phi4-Mini, Llama 3.2 1B/3B, Gemma 3 4B — any hardware
- Tier 2 (7-8B): Qwen3 8B, Gemma 3 12B, Qwen 2.5 Coder 7B — 4GB VRAM
- Tier 3 (14B): Phi4, Qwen3 14B, DeepSeek R1 14B — 8GB VRAM
- Tier 4 (27-32B): Qwen3 32B, Gemma 3 27B, DeepSeek R1 32B — 16-24GB VRAM

CLOUD PROVIDERS (optional, for users with API keys):
- Anthropic, OpenAI, Google, Groq, Ollama, OpenRouter, Together,
  HuggingFace, Mistral, xAI, DeepSeek, LM Studio, vLLM, Custom
"""
import os
import json
import time
import asyncio
import hashlib
import logging
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from collections import defaultdict

logger = logging.getLogger("neural-brain")


# ═══════════════════════════════════════════════════════
# Provider & Model Registry
# ═══════════════════════════════════════════════════════

class ProviderType(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    VLLM = "vllm"
    HUGGINGFACE = "huggingface"
    OPENROUTER = "openrouter"
    GROQ = "groq"
    TOGETHER = "together"
    MISTRAL = "mistral"
    XAI = "xai"
    DEEPSEEK = "deepseek"
    MINIMAX = "minimax"
    CUSTOM = "custom"


class ModelCapability(Enum):
    CHAT = "chat"
    CODE = "code"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"
    EMBEDDING = "embedding"
    LONG_CONTEXT = "long_context"
    REASONING = "reasoning"
    FAST = "fast"
    CHEAP = "cheap"
    OMNI = "omni"          # Multi-modal: text+image+audio+video
    SPEECH_IN = "speech_in"   # Can accept audio input
    SPEECH_OUT = "speech_out" # Can generate speech output


class RoutingStrategy(Enum):
    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    BEST_QUALITY = "best_quality"
    ROUND_ROBIN = "round_robin"
    FALLBACK = "fallback"
    CAPABILITY = "capability"
    LOCAL_FIRST = "local_first"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    id: str
    provider: ProviderType
    name: str
    context_window: int = 128000
    max_output: int = 4096
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    capabilities: List[ModelCapability] = field(default_factory=list)
    is_local: bool = False
    endpoint: str = ""
    api_key_env: str = ""
    enabled: bool = True
    avg_latency_ms: float = 0.0
    success_rate: float = 100.0
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    category: str = "general"  # general, code, vision, audio, reasoning, embedding


@dataclass
class ProviderConfig:
    type: ProviderType
    name: str
    base_url: str
    api_key: str = ""
    api_key_env: str = ""
    models: List[ModelConfig] = field(default_factory=list)
    enabled: bool = True
    is_local: bool = False
    max_retries: int = 3
    timeout_seconds: int = 120
    rate_limit_rpm: int = 60
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class CompletionRequest:
    messages: List[Dict[str, str]]
    model: str = ""
    provider: str = ""
    system: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    stop: List[str] = field(default_factory=list)
    stream: bool = False
    tools: List[Dict] = field(default_factory=list)
    tool_choice: str = "auto"
    response_format: Optional[Dict] = None
    routing_strategy: RoutingStrategy = RoutingStrategy.LOCAL_FIRST
    required_capabilities: List[ModelCapability] = field(default_factory=list)
    max_cost_per_request: float = 0.0
    product: str = ""
    request_id: str = ""
    user_id: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class CompletionResponse:
    id: str
    content: str
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=dict)
    cost: float = 0.0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    tool_calls: List[Dict] = field(default_factory=list)
    cached: bool = False
    request_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class EmbeddingRequest:
    texts: List[str]
    model: str = ""
    provider: str = ""
    dimensions: int = 0
    product: str = ""


@dataclass
class EmbeddingResponse:
    embeddings: List[List[float]]
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=dict)
    cost: float = 0.0
    dimensions: int = 0


# ═══════════════════════════════════════════════════════
# COMPREHENSIVE 2026 MODEL CATALOG
# Every major LLM organized by category
# ═══════════════════════════════════════════════════════

DEFAULT_MODELS = {

    # ══════════════════════════════════════════════
    # CATEGORY 1: FRONTIER TEXT/REASONING (Cloud)
    # Similar or superior to MiniMax M2.5
    # ══════════════════════════════════════════════

    # ─── Anthropic Claude ───
    "claude-opus-4-6": ModelConfig(
        id="claude-opus-4-6", provider=ProviderType.ANTHROPIC,
        name="Claude Opus 4.6 (Frontier)", context_window=200000, max_output=32000,
        cost_per_1k_input=0.015, cost_per_1k_output=0.075,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.VISION,
                      ModelCapability.FUNCTION_CALLING, ModelCapability.STREAMING,
                      ModelCapability.LONG_CONTEXT, ModelCapability.REASONING],
        api_key_env="ANTHROPIC_API_KEY", category="general",
    ),
    "claude-sonnet-4-5-20250929": ModelConfig(
        id="claude-sonnet-4-5-20250929", provider=ProviderType.ANTHROPIC,
        name="Claude Sonnet 4.5", context_window=200000, max_output=16000,
        cost_per_1k_input=0.003, cost_per_1k_output=0.015,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.VISION,
                      ModelCapability.FUNCTION_CALLING, ModelCapability.STREAMING,
                      ModelCapability.LONG_CONTEXT, ModelCapability.REASONING],
        api_key_env="ANTHROPIC_API_KEY", category="general",
    ),
    "claude-haiku-4-5-20251001": ModelConfig(
        id="claude-haiku-4-5-20251001", provider=ProviderType.ANTHROPIC,
        name="Claude Haiku 4.5 (Fast)", context_window=200000, max_output=8192,
        cost_per_1k_input=0.0008, cost_per_1k_output=0.004,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.STREAMING,
                      ModelCapability.FAST, ModelCapability.CHEAP],
        api_key_env="ANTHROPIC_API_KEY", category="general",
    ),

    # ─── OpenAI GPT ───
    "gpt-5.2": ModelConfig(
        id="gpt-5.2", provider=ProviderType.OPENAI,
        name="GPT-5.2 (Frontier)", context_window=400000, max_output=32768,
        cost_per_1k_input=0.00175, cost_per_1k_output=0.014,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.VISION,
                      ModelCapability.AUDIO, ModelCapability.FUNCTION_CALLING,
                      ModelCapability.STREAMING, ModelCapability.LONG_CONTEXT,
                      ModelCapability.REASONING, ModelCapability.OMNI],
        api_key_env="OPENAI_API_KEY", category="general",
    ),
    "gpt-4.1": ModelConfig(
        id="gpt-4.1", provider=ProviderType.OPENAI,
        name="GPT-4.1", context_window=1000000, max_output=32768,
        cost_per_1k_input=0.002, cost_per_1k_output=0.008,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.VISION,
                      ModelCapability.FUNCTION_CALLING, ModelCapability.STREAMING,
                      ModelCapability.LONG_CONTEXT],
        api_key_env="OPENAI_API_KEY", category="general",
    ),
    "gpt-4o": ModelConfig(
        id="gpt-4o", provider=ProviderType.OPENAI,
        name="GPT-4o (Omni)", context_window=128000, max_output=16384,
        cost_per_1k_input=0.0025, cost_per_1k_output=0.01,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.VISION,
                      ModelCapability.AUDIO, ModelCapability.FUNCTION_CALLING,
                      ModelCapability.STREAMING, ModelCapability.OMNI,
                      ModelCapability.SPEECH_IN, ModelCapability.SPEECH_OUT],
        api_key_env="OPENAI_API_KEY", category="general",
    ),
    "gpt-4o-mini": ModelConfig(
        id="gpt-4o-mini", provider=ProviderType.OPENAI,
        name="GPT-4o Mini (Cheap)", context_window=128000, max_output=16384,
        cost_per_1k_input=0.00015, cost_per_1k_output=0.0006,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.STREAMING,
                      ModelCapability.FAST, ModelCapability.CHEAP, ModelCapability.VISION],
        api_key_env="OPENAI_API_KEY", category="general",
    ),

    # ─── OpenAI Reasoning ───
    "o4-mini": ModelConfig(
        id="o4-mini", provider=ProviderType.OPENAI,
        name="o4-mini (Reasoning)", context_window=200000, max_output=100000,
        cost_per_1k_input=0.0011, cost_per_1k_output=0.0044,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                      ModelCapability.LONG_CONTEXT, ModelCapability.FUNCTION_CALLING],
        api_key_env="OPENAI_API_KEY", category="reasoning",
    ),
    "o3": ModelConfig(
        id="o3", provider=ProviderType.OPENAI,
        name="o3 (Deep Reasoning)", context_window=200000, max_output=100000,
        cost_per_1k_input=0.01, cost_per_1k_output=0.04,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                      ModelCapability.LONG_CONTEXT, ModelCapability.VISION],
        api_key_env="OPENAI_API_KEY", category="reasoning",
    ),

    # ─── OpenAI Codex ───
    "gpt-5.2-codex": ModelConfig(
        id="gpt-5.2-codex", provider=ProviderType.OPENAI,
        name="GPT-5.2 Codex (Code)", context_window=200000, max_output=16384,
        cost_per_1k_input=0.002, cost_per_1k_output=0.008,
        capabilities=[ModelCapability.CODE, ModelCapability.CHAT, ModelCapability.FUNCTION_CALLING,
                      ModelCapability.STREAMING],
        api_key_env="OPENAI_API_KEY", category="code",
    ),

    # ─── Google Gemini (Complete Flash + Pro Family) ───
    "gemini-3-flash": ModelConfig(
        id="gemini-3-flash-preview", provider=ProviderType.GOOGLE,
        name="Gemini 3 Flash ⚡ (NEWEST - Pro intelligence at Flash speed)", context_window=1000000, max_output=65536,
        cost_per_1k_input=0.00015, cost_per_1k_output=0.0006,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.VISION,
                      ModelCapability.AUDIO, ModelCapability.VIDEO,
                      ModelCapability.FUNCTION_CALLING, ModelCapability.STREAMING,
                      ModelCapability.LONG_CONTEXT, ModelCapability.REASONING,
                      ModelCapability.OMNI, ModelCapability.FAST],
        api_key_env="GOOGLE_API_KEY", category="general",
    ),
    "gemini-3.1-pro": ModelConfig(
        id="gemini-3.1-pro-preview", provider=ProviderType.GOOGLE,
        name="Gemini 3.1 Pro (Latest Frontier)", context_window=1000000, max_output=65536,
        cost_per_1k_input=0.00125, cost_per_1k_output=0.005,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.VISION,
                      ModelCapability.AUDIO, ModelCapability.VIDEO,
                      ModelCapability.FUNCTION_CALLING, ModelCapability.STREAMING,
                      ModelCapability.LONG_CONTEXT, ModelCapability.REASONING,
                      ModelCapability.OMNI],
        api_key_env="GOOGLE_API_KEY", category="general",
    ),
    "gemini-3.0-pro": ModelConfig(
        id="gemini-3.0-pro", provider=ProviderType.GOOGLE,
        name="Gemini 3 Pro (Frontier)", context_window=2000000, max_output=65536,
        cost_per_1k_input=0.00125, cost_per_1k_output=0.005,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.VISION,
                      ModelCapability.AUDIO, ModelCapability.VIDEO,
                      ModelCapability.FUNCTION_CALLING, ModelCapability.STREAMING,
                      ModelCapability.LONG_CONTEXT, ModelCapability.REASONING,
                      ModelCapability.OMNI, ModelCapability.SPEECH_IN],
        api_key_env="GOOGLE_API_KEY", category="general",
    ),
    "gemini-2.5-pro": ModelConfig(
        id="gemini-2.5-pro", provider=ProviderType.GOOGLE,
        name="Gemini 2.5 Pro", context_window=1000000, max_output=65536,
        cost_per_1k_input=0.00125, cost_per_1k_output=0.01,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.VISION,
                      ModelCapability.AUDIO, ModelCapability.VIDEO,
                      ModelCapability.FUNCTION_CALLING, ModelCapability.STREAMING,
                      ModelCapability.LONG_CONTEXT, ModelCapability.REASONING, ModelCapability.OMNI],
        api_key_env="GOOGLE_API_KEY", category="general",
    ),
    "gemini-2.5-flash": ModelConfig(
        id="gemini-2.5-flash", provider=ProviderType.GOOGLE,
        name="Gemini 2.5 Flash (Fast+Cheap)", context_window=1000000, max_output=65536,
        cost_per_1k_input=0.000075, cost_per_1k_output=0.0003,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.VISION,
                      ModelCapability.AUDIO, ModelCapability.VIDEO,
                      ModelCapability.STREAMING, ModelCapability.FAST, ModelCapability.CHEAP,
                      ModelCapability.LONG_CONTEXT, ModelCapability.OMNI],
        api_key_env="GOOGLE_API_KEY", category="general",
    ),
    "gemini-2.5-flash-lite": ModelConfig(
        id="gemini-2.5-flash-lite", provider=ProviderType.GOOGLE,
        name="Gemini 2.5 Flash Lite (Ultra-Cheap)", context_window=1000000, max_output=65536,
        cost_per_1k_input=0.000025, cost_per_1k_output=0.0001,
        capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING, ModelCapability.FAST,
                      ModelCapability.CHEAP, ModelCapability.LONG_CONTEXT],
        api_key_env="GOOGLE_API_KEY", category="general",
    ),
    "gemini-2.5-flash-tts": ModelConfig(
        id="gemini-2.5-flash-tts-preview", provider=ProviderType.GOOGLE,
        name="Gemini 2.5 Flash TTS (Text-to-Speech)", context_window=32000, max_output=8192,
        cost_per_1k_input=0.00015, cost_per_1k_output=0.0006,
        capabilities=[ModelCapability.CHAT, ModelCapability.AUDIO, ModelCapability.SPEECH_OUT,
                      ModelCapability.STREAMING, ModelCapability.FAST],
        api_key_env="GOOGLE_API_KEY", category="audio",
    ),
    "gemini-2.5-flash-native-audio": ModelConfig(
        id="gemini-live-2.5-flash-native-audio", provider=ProviderType.GOOGLE,
        name="Gemini 2.5 Flash Native Audio (Real-time Bidirectional)", context_window=128000, max_output=8192,
        cost_per_1k_input=0.0004, cost_per_1k_output=0.0016,
        capabilities=[ModelCapability.CHAT, ModelCapability.AUDIO, ModelCapability.SPEECH_IN,
                      ModelCapability.SPEECH_OUT, ModelCapability.STREAMING,
                      ModelCapability.FAST, ModelCapability.OMNI],
        api_key_env="GOOGLE_API_KEY", category="audio",
    ),
    "gemini-2.0-flash": ModelConfig(
        id="gemini-2.0-flash", provider=ProviderType.GOOGLE,
        name="Gemini 2.0 Flash (Retiring Mar 2026)", context_window=1000000, max_output=8192,
        cost_per_1k_input=0.0001, cost_per_1k_output=0.0004,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.VISION,
                      ModelCapability.STREAMING, ModelCapability.FAST, ModelCapability.CHEAP,
                      ModelCapability.LONG_CONTEXT],
        api_key_env="GOOGLE_API_KEY", category="general",
    ),

    # ─── xAI Grok ───
    "grok-4.1": ModelConfig(
        id="grok-4.1", provider=ProviderType.XAI,
        name="Grok 4.1 (Cheapest Frontier)", context_window=2000000, max_output=32768,
        cost_per_1k_input=0.0002, cost_per_1k_output=0.001,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.VISION,
                      ModelCapability.FUNCTION_CALLING, ModelCapability.STREAMING,
                      ModelCapability.LONG_CONTEXT, ModelCapability.REASONING,
                      ModelCapability.CHEAP],
        api_key_env="XAI_API_KEY", category="general",
    ),
    "grok-3": ModelConfig(
        id="grok-3", provider=ProviderType.XAI,
        name="Grok 3", context_window=131072, max_output=16384,
        cost_per_1k_input=0.003, cost_per_1k_output=0.015,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.VISION,
                      ModelCapability.FUNCTION_CALLING, ModelCapability.STREAMING,
                      ModelCapability.REASONING],
        api_key_env="XAI_API_KEY", category="general",
    ),

    # ─── DeepSeek ───
    "deepseek-v3.2": ModelConfig(
        id="deepseek-v3.2", provider=ProviderType.DEEPSEEK,
        name="DeepSeek V3.2 (MoE, Near-Free)", context_window=128000, max_output=16384,
        cost_per_1k_input=0.00027, cost_per_1k_output=0.0011,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.FUNCTION_CALLING,
                      ModelCapability.STREAMING, ModelCapability.REASONING, ModelCapability.CHEAP],
        api_key_env="DEEPSEEK_API_KEY", category="general",
    ),
    "deepseek-r1": ModelConfig(
        id="deepseek-r1", provider=ProviderType.DEEPSEEK,
        name="DeepSeek R1 (Reasoning)", context_window=128000, max_output=16384,
        cost_per_1k_input=0.00055, cost_per_1k_output=0.0022,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                      ModelCapability.STREAMING],
        api_key_env="DEEPSEEK_API_KEY", category="reasoning",
    ),

    # ─── Mistral ───
    "mistral-large-3": ModelConfig(
        id="mistral-large-3", provider=ProviderType.MISTRAL,
        name="Mistral Large 3", context_window=128000, max_output=16384,
        cost_per_1k_input=0.002, cost_per_1k_output=0.006,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.VISION,
                      ModelCapability.FUNCTION_CALLING, ModelCapability.STREAMING],
        api_key_env="MISTRAL_API_KEY", category="general",
    ),
    "codestral-latest": ModelConfig(
        id="codestral-latest", provider=ProviderType.MISTRAL,
        name="Codestral (Code Specialist)", context_window=256000, max_output=16384,
        cost_per_1k_input=0.0003, cost_per_1k_output=0.0009,
        capabilities=[ModelCapability.CODE, ModelCapability.CHAT, ModelCapability.STREAMING,
                      ModelCapability.FAST],
        api_key_env="MISTRAL_API_KEY", category="code",
    ),

    # ─── MiniMax ───
    "MiniMax-M2.5": ModelConfig(
        id="MiniMax-M2.5", provider=ProviderType.MINIMAX,
        name="MiniMax M2.5 (Standard 50 TPS)", context_window=200000, max_output=131072,
        cost_per_1k_input=0.00015, cost_per_1k_output=0.0012,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                      ModelCapability.FUNCTION_CALLING, ModelCapability.STREAMING,
                      ModelCapability.LONG_CONTEXT],
        api_key_env="MINIMAX_API_KEY", category="general",
    ),
    "MiniMax-M2.5-Lightning": ModelConfig(
        id="MiniMax-M2.5-Lightning", provider=ProviderType.MINIMAX,
        name="MiniMax M2.5 Lightning (100 TPS)", context_window=200000, max_output=131072,
        cost_per_1k_input=0.0003, cost_per_1k_output=0.0024,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                      ModelCapability.FUNCTION_CALLING, ModelCapability.STREAMING,
                      ModelCapability.LONG_CONTEXT, ModelCapability.FAST],
        api_key_env="MINIMAX_API_KEY", category="general",
    ),

    # ─── Groq (Ultra-Fast Inference) ───
    "llama-3.3-70b-versatile": ModelConfig(
        id="llama-3.3-70b-versatile", provider=ProviderType.GROQ,
        name="Llama 3.3 70B via Groq (Ultra-Fast)", context_window=128000, max_output=32768,
        cost_per_1k_input=0.00059, cost_per_1k_output=0.00079,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.STREAMING,
                      ModelCapability.FAST],
        api_key_env="GROQ_API_KEY", category="general",
    ),
    "deepseek-r1-distill-llama-70b": ModelConfig(
        id="deepseek-r1-distill-llama-70b", provider=ProviderType.GROQ,
        name="DeepSeek R1 70B via Groq (Fast Reasoning)", context_window=128000, max_output=16384,
        cost_per_1k_input=0.00075, cost_per_1k_output=0.00099,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                      ModelCapability.STREAMING, ModelCapability.FAST],
        api_key_env="GROQ_API_KEY", category="reasoning",
    ),
    "gemma2-9b-it": ModelConfig(
        id="gemma2-9b-it", provider=ProviderType.GROQ,
        name="Gemma 2 9B via Groq (Free Tier)", context_window=8192, max_output=4096,
        cost_per_1k_input=0.0002, cost_per_1k_output=0.0002,
        capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING, ModelCapability.FAST,
                      ModelCapability.CHEAP],
        api_key_env="GROQ_API_KEY", category="general",
    ),

    # ══════════════════════════════════════════════
    # CATEGORY 2: AUDIO / SPEECH MODELS
    # Best for voice, music, transcription
    # ══════════════════════════════════════════════

    "gpt-4o-audio": ModelConfig(
        id="gpt-4o-audio-preview", provider=ProviderType.OPENAI,
        name="GPT-4o Audio (Voice AI)", context_window=128000, max_output=16384,
        cost_per_1k_input=0.01, cost_per_1k_output=0.02,
        capabilities=[ModelCapability.CHAT, ModelCapability.AUDIO, ModelCapability.SPEECH_IN,
                      ModelCapability.SPEECH_OUT, ModelCapability.OMNI, ModelCapability.STREAMING],
        api_key_env="OPENAI_API_KEY", category="audio",
    ),
    "whisper-large-v3": ModelConfig(
        id="whisper-large-v3", provider=ProviderType.GROQ,
        name="Whisper v3 via Groq (STT, Ultra-Fast)", context_window=0, max_output=0,
        cost_per_1k_input=0.000111, cost_per_1k_output=0.0,
        capabilities=[ModelCapability.AUDIO, ModelCapability.SPEECH_IN, ModelCapability.FAST],
        api_key_env="GROQ_API_KEY", category="audio",
    ),

    # ══════════════════════════════════════════════
    # CATEGORY 3: VIDEO UNDERSTANDING MODELS
    # ══════════════════════════════════════════════

    # Gemini 3 Pro, Gemini 2.5 Pro (already listed above — best for video)
    # They accept video input natively with 1M+ token context

    # ══════════════════════════════════════════════
    # CATEGORY 4: EMBEDDING MODELS
    # ══════════════════════════════════════════════

    "text-embedding-3-small": ModelConfig(
        id="text-embedding-3-small", provider=ProviderType.OPENAI,
        name="OpenAI Embedding Small", context_window=8191, max_output=0,
        cost_per_1k_input=0.00002, cost_per_1k_output=0.0,
        capabilities=[ModelCapability.EMBEDDING, ModelCapability.CHEAP],
        api_key_env="OPENAI_API_KEY", category="embedding",
    ),
    "text-embedding-3-large": ModelConfig(
        id="text-embedding-3-large", provider=ProviderType.OPENAI,
        name="OpenAI Embedding Large", context_window=8191, max_output=0,
        cost_per_1k_input=0.00013, cost_per_1k_output=0.0,
        capabilities=[ModelCapability.EMBEDDING],
        api_key_env="OPENAI_API_KEY", category="embedding",
    ),

    # ══════════════════════════════════════════════
    # CATEGORY 5: LOCAL MODELS (Ollama — FREE)
    # Optimized for maximum efficiency on any hardware
    # Tiered: Tiny (1-3B) → Small (7-8B) → Medium (14B) → Large (27-32B)
    # ══════════════════════════════════════════════

    # ─── TIER 1: ULTRA-LIGHT (1-4B, runs on anything, <3GB VRAM) ───
    "ollama/qwen3:1.7b": ModelConfig(
        id="ollama/qwen3:1.7b", provider=ProviderType.OLLAMA,
        name="Qwen 3 1.7B (Ultra-Light, Hybrid Thinking)", context_window=40960, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING, ModelCapability.FAST,
                      ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),
    "ollama/qwen3:4b": ModelConfig(
        id="ollama/qwen3:4b", provider=ProviderType.OLLAMA,
        name="Qwen 3 4B (Light, Beats 7B Models)", context_window=40960, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                      ModelCapability.STREAMING, ModelCapability.FAST, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),
    "ollama/phi4-mini": ModelConfig(
        id="ollama/phi4-mini", provider=ProviderType.OLLAMA,
        name="Phi 4 Mini 3.8B (Microsoft, Strong STEM)", context_window=128000, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                      ModelCapability.STREAMING, ModelCapability.FAST, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),
    "ollama/llama3.2:3b": ModelConfig(
        id="ollama/llama3.2:3b", provider=ProviderType.OLLAMA,
        name="Llama 3.2 3B (Meta, Ultra-Fast)", context_window=128000, max_output=4096,
        capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING, ModelCapability.FAST,
                      ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),
    "ollama/llama3.2:1b": ModelConfig(
        id="ollama/llama3.2:1b", provider=ProviderType.OLLAMA,
        name="Llama 3.2 1B (Instant Responses, 700MB)", context_window=128000, max_output=4096,
        capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING, ModelCapability.FAST,
                      ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),
    "ollama/gemma3:4b": ModelConfig(
        id="ollama/gemma3:4b", provider=ProviderType.OLLAMA,
        name="Gemma 3 4B (Google, Vision+Text)", context_window=128000, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.VISION, ModelCapability.STREAMING,
                      ModelCapability.FAST, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),
    "ollama/granite4": ModelConfig(
        id="ollama/granite4", provider=ProviderType.OLLAMA,
        name="IBM Granite 4 3B (Hybrid Mamba, Tool Calling)", context_window=128000, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE,
                      ModelCapability.FUNCTION_CALLING, ModelCapability.STREAMING,
                      ModelCapability.CHEAP, ModelCapability.FAST],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),
    "ollama/smollm2:1.7b": ModelConfig(
        id="ollama/smollm2:1.7b", provider=ProviderType.OLLAMA,
        name="SmolLM2 1.7B (HuggingFace, Tiny+Fast)", context_window=8192, max_output=4096,
        capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING, ModelCapability.FAST,
                      ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),

    # ─── TIER 2: EFFICIENT (7-8B, sweet spot, 4-5GB VRAM) ───
    "ollama/qwen3:8b": ModelConfig(
        id="ollama/qwen3:8b", provider=ProviderType.OLLAMA,
        name="Qwen 3 8B (Best 8B Model, Hybrid Thinking)", context_window=40960, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                      ModelCapability.STREAMING, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),
    "ollama/gemma3": ModelConfig(
        id="ollama/gemma3", provider=ProviderType.OLLAMA,
        name="Gemma 3 12B (Google, Vision+Text, Strong)", context_window=128000, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.VISION,
                      ModelCapability.STREAMING, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),
    "ollama/llama3.3": ModelConfig(
        id="ollama/llama3.3", provider=ProviderType.OLLAMA,
        name="Llama 3.3 70B (Q4 runs on 48GB)", context_window=128000, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.STREAMING,
                      ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),
    "ollama/mistral": ModelConfig(
        id="ollama/mistral", provider=ProviderType.OLLAMA,
        name="Mistral 7B (Stable Classic, 4GB)", context_window=32000, max_output=4096,
        capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING, ModelCapability.FAST,
                      ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),
    "ollama/phi3:mini": ModelConfig(
        id="ollama/phi3:mini", provider=ProviderType.OLLAMA,
        name="Phi3 Mini 3.8B (Legacy, 2.2GB)", context_window=128000, max_output=4096,
        capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING, ModelCapability.FAST,
                      ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),
    "ollama/qwen2.5": ModelConfig(
        id="ollama/qwen2.5", provider=ProviderType.OLLAMA,
        name="Qwen 2.5 7B (Solid All-Around)", context_window=128000, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.STREAMING,
                      ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),
    "ollama/qwen3": ModelConfig(
        id="ollama/qwen3", provider=ProviderType.OLLAMA,
        name="Qwen 3 8B (Latest, Hybrid Thinking)", context_window=40960, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                      ModelCapability.STREAMING, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),
    "ollama/llama3": ModelConfig(
        id="ollama/llama3", provider=ProviderType.OLLAMA,
        name="Llama 3 8B (Meta Foundation)", context_window=8192, max_output=4096,
        capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING,
                      ModelCapability.CHEAP, ModelCapability.FAST],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),

    # ─── TIER 3: MEDIUM (14B, great quality, 8GB VRAM) ───
    "ollama/phi4": ModelConfig(
        id="ollama/phi4", provider=ProviderType.OLLAMA,
        name="Phi 4 14B (Microsoft, Excellent Reasoning)", context_window=200000, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                      ModelCapability.STREAMING, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),
    "ollama/qwen3:14b": ModelConfig(
        id="ollama/qwen3:14b", provider=ProviderType.OLLAMA,
        name="Qwen 3 14B (Near-Frontier Local Quality)", context_window=40960, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                      ModelCapability.STREAMING, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),
    "ollama/deepseek-r1:14b": ModelConfig(
        id="ollama/deepseek-r1:14b", provider=ProviderType.OLLAMA,
        name="DeepSeek R1 14B (Strong Reasoning, 8GB)", context_window=128000, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                      ModelCapability.STREAMING, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="reasoning",
    ),

    # ─── TIER 4: LARGE (27-32B, top local quality, 16-20GB VRAM) ───
    "ollama/gemma3:27b": ModelConfig(
        id="ollama/gemma3:27b", provider=ProviderType.OLLAMA,
        name="Gemma 3 27B (Google, Near-GPT-4 Quality)", context_window=128000, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.VISION,
                      ModelCapability.REASONING, ModelCapability.STREAMING, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),
    "ollama/qwen3:32b": ModelConfig(
        id="ollama/qwen3:32b", provider=ProviderType.OLLAMA,
        name="Qwen 3 32B (Best Open Model, Frontier-Class)", context_window=40960, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                      ModelCapability.STREAMING, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),
    "ollama/qwen2.5-coder:32b": ModelConfig(
        id="ollama/qwen2.5-coder:32b", provider=ProviderType.OLLAMA,
        name="Qwen 2.5 Coder 32B (Best Local Coding)", context_window=131072, max_output=8192,
        capabilities=[ModelCapability.CODE, ModelCapability.CHAT, ModelCapability.STREAMING,
                      ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="code",
    ),
    "ollama/deepseek-r1:32b": ModelConfig(
        id="ollama/deepseek-r1:32b", provider=ProviderType.OLLAMA,
        name="DeepSeek R1 32B (Best Local Reasoning)", context_window=128000, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                      ModelCapability.STREAMING, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="reasoning",
    ),
    "ollama/glm-4.7-flash": ModelConfig(
        id="ollama/glm-4.7-flash", provider=ProviderType.OLLAMA,
        name="GLM-4.7-Flash (30B MoE, Best-in-30B-class)", context_window=200000, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                      ModelCapability.FUNCTION_CALLING, ModelCapability.STREAMING,
                      ModelCapability.LONG_CONTEXT, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),

    # ─── Code Specialists (Local) ───
    "ollama/qwen2.5-coder:7b": ModelConfig(
        id="ollama/qwen2.5-coder:7b", provider=ProviderType.OLLAMA,
        name="Qwen 2.5 Coder 7B (Efficient Code)", context_window=131072, max_output=8192,
        capabilities=[ModelCapability.CODE, ModelCapability.CHAT, ModelCapability.STREAMING,
                      ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="code",
    ),
    "ollama/qwen2.5-coder:14b": ModelConfig(
        id="ollama/qwen2.5-coder:14b", provider=ProviderType.OLLAMA,
        name="Qwen 2.5 Coder 14B (Strong Code)", context_window=131072, max_output=8192,
        capabilities=[ModelCapability.CODE, ModelCapability.CHAT, ModelCapability.STREAMING,
                      ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="code",
    ),
    "ollama/devstral": ModelConfig(
        id="ollama/devstral", provider=ProviderType.OLLAMA,
        name="Devstral 24B (Mistral, SWE Agent)", context_window=128000, max_output=8192,
        capabilities=[ModelCapability.CODE, ModelCapability.CHAT, ModelCapability.FUNCTION_CALLING,
                      ModelCapability.STREAMING, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="code",
    ),
    "ollama/deepseek-coder-v2": ModelConfig(
        id="ollama/deepseek-coder-v2", provider=ProviderType.OLLAMA,
        name="DeepSeek Coder V2 16B (MoE)", context_window=128000, max_output=4096,
        capabilities=[ModelCapability.CODE, ModelCapability.STREAMING, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="code",
    ),
    "ollama/starcoder2:7b": ModelConfig(
        id="ollama/starcoder2:7b", provider=ProviderType.OLLAMA,
        name="StarCoder2 7B (BigCode, 600+ Languages)", context_window=16384, max_output=4096,
        capabilities=[ModelCapability.CODE, ModelCapability.STREAMING, ModelCapability.FAST,
                      ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="code",
    ),

    # ─── Reasoning Specialists (Local) ───
    "ollama/deepseek-r1:8b": ModelConfig(
        id="ollama/deepseek-r1:8b", provider=ProviderType.OLLAMA,
        name="DeepSeek R1 8B (Efficient Reasoning)", context_window=128000, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                      ModelCapability.STREAMING, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="reasoning",
    ),
    "ollama/deepseek-r1:1.5b": ModelConfig(
        id="ollama/deepseek-r1:1.5b", provider=ProviderType.OLLAMA,
        name="DeepSeek R1 1.5B (Tiny Reasoner, 1GB)", context_window=128000, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.REASONING,
                      ModelCapability.STREAMING, ModelCapability.FAST, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="reasoning",
    ),
    "ollama/phi4-reasoning": ModelConfig(
        id="ollama/phi4-reasoning", provider=ProviderType.OLLAMA,
        name="Phi 4 Reasoning 14B (Microsoft)", context_window=200000, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING,
                      ModelCapability.STREAMING, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="reasoning",
    ),

    # ─── Vision/Multimodal (Local) ───
    "ollama/llama4:scout": ModelConfig(
        id="ollama/llama4:scout", provider=ProviderType.OLLAMA,
        name="Llama 4 Scout (Vision+Text, 512K Context)", context_window=512000, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.VISION,
                      ModelCapability.STREAMING, ModelCapability.LONG_CONTEXT, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="vision",
    ),
    "ollama/qwen3-vl": ModelConfig(
        id="ollama/qwen3-vl", provider=ProviderType.OLLAMA,
        name="Qwen3-VL 8B (Best Open VLM, Vision+Video)", context_window=256000, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.VISION, ModelCapability.VIDEO,
                      ModelCapability.FUNCTION_CALLING, ModelCapability.STREAMING,
                      ModelCapability.LONG_CONTEXT, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="vision",
    ),
    "ollama/mistral-small3.1": ModelConfig(
        id="ollama/mistral-small3.1", provider=ProviderType.OLLAMA,
        name="Mistral Small 3.1 24B (Vision+Text, Strong)", context_window=128000, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.VISION, ModelCapability.FUNCTION_CALLING,
                      ModelCapability.STREAMING, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="vision",
    ),
    "ollama/llava": ModelConfig(
        id="ollama/llava", provider=ProviderType.OLLAMA,
        name="LLaVA 1.6 7B (Lightweight Vision)", context_window=4096, max_output=2048,
        capabilities=[ModelCapability.CHAT, ModelCapability.VISION, ModelCapability.STREAMING,
                      ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="vision",
    ),
    "ollama/llama3.2-vision": ModelConfig(
        id="ollama/llama3.2-vision", provider=ProviderType.OLLAMA,
        name="Llama 3.2 Vision 11B (Meta, Efficient VLM)", context_window=128000, max_output=4096,
        capabilities=[ModelCapability.CHAT, ModelCapability.VISION, ModelCapability.STREAMING,
                      ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="vision",
    ),

    # ─── Omni/Audio (Local) ───
    "ollama/minicpm-o": ModelConfig(
        id="ollama/minicpm-o", provider=ProviderType.OLLAMA,
        name="MiniCPM-o 2.6 (Omni: Vision+Audio+Speech)", context_window=32000, max_output=4096,
        capabilities=[ModelCapability.CHAT, ModelCapability.VISION, ModelCapability.AUDIO,
                      ModelCapability.SPEECH_IN, ModelCapability.SPEECH_OUT,
                      ModelCapability.OMNI, ModelCapability.STREAMING, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="audio",
    ),

    # ─── Specialist Models (Local) ───
    "ollama/glm-ocr": ModelConfig(
        id="ollama/glm-ocr", provider=ProviderType.OLLAMA,
        name="GLM-OCR 0.9B (#1 Document OCR)", context_window=8192, max_output=8192,
        capabilities=[ModelCapability.VISION, ModelCapability.STREAMING, ModelCapability.FAST,
                      ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="vision",
    ),
    "ollama/translategemma": ModelConfig(
        id="ollama/translategemma", provider=ProviderType.OLLAMA,
        name="TranslateGemma 4B (55 Languages)", context_window=128000, max_output=8192,
        capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="general",
    ),
    "ollama/functiongemma": ModelConfig(
        id="ollama/functiongemma", provider=ProviderType.OLLAMA,
        name="FunctionGemma 270M (Ultra-Tiny Tool Calling)", context_window=8192, max_output=2048,
        capabilities=[ModelCapability.FUNCTION_CALLING, ModelCapability.STREAMING,
                      ModelCapability.FAST, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="code",
    ),

    # ─── Embedding Models (Local) ───
    "ollama/qwen3-embedding": ModelConfig(
        id="ollama/qwen3-embedding", provider=ProviderType.OLLAMA,
        name="Qwen3-Embedding 0.6B (#1 MTEB Multilingual)", context_window=32000, max_output=0,
        capabilities=[ModelCapability.EMBEDDING, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="embedding",
    ),
    "ollama/nomic-embed-text": ModelConfig(
        id="ollama/nomic-embed-text", provider=ProviderType.OLLAMA,
        name="Nomic Embed Text 137M (Fast Embeddings)", context_window=8192, max_output=0,
        capabilities=[ModelCapability.EMBEDDING, ModelCapability.FAST, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="embedding",
    ),
    "ollama/mxbai-embed-large": ModelConfig(
        id="ollama/mxbai-embed-large", provider=ProviderType.OLLAMA,
        name="MxBAI Embed Large 335M (High Quality Embeddings)", context_window=512, max_output=0,
        capabilities=[ModelCapability.EMBEDDING, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:11434", category="embedding",
    ),

    # ─── LM Studio ───
    "lm-studio/default": ModelConfig(
        id="lm-studio/default", provider=ProviderType.LM_STUDIO,
        name="LM Studio Model", context_window=32000, max_output=4096,
        capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING, ModelCapability.CHEAP],
        is_local=True, endpoint="http://localhost:1234/v1", category="general",
    ),
}


# ═══════════════════════════════════════════════════════
# Quality Ranking — used by BEST_QUALITY router
# ═══════════════════════════════════════════════════════

QUALITY_RANKING = [
    "claude-opus-4-6",           # 1. Best overall reasoning
    "gemini-3.1-pro",            # 2. Latest Gemini frontier
    "gemini-3.0-pro",            # 3. Best multimodal + 2M context
    "gpt-5.2",                   # 4. Best omni-modal
    "gemini-3-flash",            # 5. Pro intelligence at Flash speed
    "grok-4.1",                  # 6. Cheapest frontier
    "o3",                        # 7. Deepest reasoning
    "claude-sonnet-4-5-20250929",# 8. Great balance
    "gemini-2.5-pro",            # 9. Strong multimodal
    "gpt-4.1",                   # 10. 1M context
    "deepseek-v3.2",             # 11. Best value reasoning
    "mistral-large-3",           # 12. Good European option
    "MiniMax-M2.5",              # 13. Great value
    "MiniMax-M2.5-Lightning",    # 14. Fast inference
    "gpt-4o",                    # 15. Proven omni-modal
    "deepseek-r1",               # 16. Open reasoning
    "llama-3.3-70b-versatile",   # 17. Fast via Groq
    "claude-haiku-4-5-20251001", # 18. Fast Anthropic
    "gemini-2.5-flash",          # 19. Best value Flash
    "gpt-4o-mini",               # 20. Cheapest OpenAI
    "gemini-2.5-flash-lite",     # 21. Ultra-cheap Google
    "gemini-2.0-flash",          # 22. Legacy
]


# ═══════════════════════════════════════════════════════
# Local Quality Ranking — used by LOCAL_FIRST router
# Best-to-good ordering for local Ollama models
# ═══════════════════════════════════════════════════════

LOCAL_QUALITY_RANKING = [
    # Tier 4: Large — best quality available locally
    "ollama/qwen3:32b",         # 1. Best open model overall
    "ollama/gemma3:27b",        # 2. Near-GPT-4 quality, vision
    "ollama/deepseek-r1:32b",   # 3. Best local reasoning
    "ollama/qwen2.5-coder:32b", # 4. Best local coding
    "ollama/glm-4.7-flash",     # 5. 30B MoE, great efficiency
    "ollama/devstral",          # 6. 24B SWE agent
    "ollama/mistral-small3.1",  # 7. 24B vision+text

    # Tier 3: Medium — great quality, fits 8GB VRAM
    "ollama/qwen3:14b",         # 8. Near-frontier quality
    "ollama/phi4",              # 9. Excellent reasoning
    "ollama/deepseek-r1:14b",   # 10. Strong reasoning
    "ollama/qwen2.5-coder:14b", # 11. Strong coding
    "ollama/phi4-reasoning",    # 12. Dedicated reasoner

    # Tier 2: Efficient — sweet spot, runs on 4GB VRAM
    "ollama/qwen3:8b",          # 13. Best 8B model
    "ollama/gemma3",            # 14. Google 12B, vision
    "ollama/qwen3",             # 15. Qwen3 default
    "ollama/qwen2.5",           # 16. Solid all-around
    "ollama/deepseek-r1:8b",    # 17. Efficient reasoning
    "ollama/qwen2.5-coder:7b",  # 18. Efficient coding
    "ollama/llama3.3",          # 19. Meta 70B (if hardware allows)

    # Tier 1: Ultra-light — instant responses, any hardware
    "ollama/qwen3:4b",          # 20. Beats 7B models
    "ollama/phi4-mini",         # 21. Strong STEM at 3.8B
    "ollama/gemma3:4b",         # 22. Vision at 4B
    "ollama/llama3.2:3b",       # 23. Ultra-fast
    "ollama/granite4",          # 24. Tool calling at 3B
    "ollama/qwen3:1.7b",        # 25. Ultra-light
    "ollama/llama3.2:1b",       # 26. Instant, 700MB
    "ollama/smollm2:1.7b",      # 27. Tiny+fast
]


# ═══════════════════════════════════════════════════════
# Prompt Cache
# ═══════════════════════════════════════════════════════

class PromptCache:
    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        self.cache: Dict[str, tuple] = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _hash(self, req: CompletionRequest) -> str:
        key = json.dumps({
            "messages": req.messages, "model": req.model, "system": req.system,
            "temperature": req.temperature, "max_tokens": req.max_tokens,
        }, sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def get(self, req: CompletionRequest) -> Optional[CompletionResponse]:
        h = self._hash(req)
        if h in self.cache:
            resp, ts = self.cache[h]
            if time.time() - ts < self.ttl:
                self.hits += 1
                resp_copy = CompletionResponse(**{k: v for k, v in asdict(resp).items()})
                resp_copy.cached = True
                return resp_copy
            else:
                del self.cache[h]
        self.misses += 1
        return None

    def set(self, req: CompletionRequest, resp: CompletionResponse):
        if req.temperature > 0 or req.stream:
            return
        h = self._hash(req)
        if len(self.cache) >= self.max_size:
            oldest = min(self.cache.items(), key=lambda x: x[1][1])
            del self.cache[oldest[0]]
        self.cache[h] = (resp, time.time())

    def stats(self) -> Dict:
        total = self.hits + self.misses
        return {"size": len(self.cache), "hits": self.hits, "misses": self.misses,
                "hit_rate": f"{self.hits/total*100:.1f}%" if total > 0 else "0%"}


# ═══════════════════════════════════════════════════════
# Usage Tracker
# ═══════════════════════════════════════════════════════

class UsageTracker:
    def __init__(self):
        self.records: List[Dict] = []
        self.by_product: Dict[str, Dict] = defaultdict(lambda: {
            "requests": 0, "tokens_in": 0, "tokens_out": 0, "cost": 0.0,
            "errors": 0, "avg_latency_ms": 0.0, "total_latency_ms": 0.0,
        })
        self.by_model: Dict[str, Dict] = defaultdict(lambda: {
            "requests": 0, "tokens_in": 0, "tokens_out": 0, "cost": 0.0,
            "errors": 0, "avg_latency_ms": 0.0, "total_latency_ms": 0.0,
        })
        self.by_provider: Dict[str, Dict] = defaultdict(lambda: {
            "requests": 0, "tokens_in": 0, "tokens_out": 0, "cost": 0.0, "errors": 0,
        })

    def record(self, req: CompletionRequest, resp: CompletionResponse):
        product = req.product or "unknown"
        model = resp.model
        provider = resp.provider
        for tracker, key in [(self.by_product, product), (self.by_model, model), (self.by_provider, provider)]:
            t = tracker[key]
            t["requests"] += 1
            t["tokens_in"] += resp.usage.get("prompt_tokens", 0)
            t["tokens_out"] += resp.usage.get("completion_tokens", 0)
            t["cost"] += resp.cost
            t["total_latency_ms"] = t.get("total_latency_ms", 0) + resp.latency_ms
            t["avg_latency_ms"] = t["total_latency_ms"] / t["requests"]
        self.records.append({
            "product": product, "model": model, "provider": provider,
            "tokens": resp.usage.get("total_tokens", 0),
            "cost": resp.cost, "latency_ms": resp.latency_ms,
            "cached": resp.cached, "timestamp": datetime.utcnow().isoformat(),
        })
        if len(self.records) > 10000:
            self.records = self.records[-5000:]

    def record_error(self, req: CompletionRequest, provider: str, model: str, error: str):
        product = req.product or "unknown"
        self.by_product[product]["errors"] += 1
        self.by_model[model]["errors"] += 1
        self.by_provider[provider]["errors"] += 1

    def get_summary(self) -> Dict:
        total_cost = sum(r["cost"] for r in self.records)
        total_tokens = sum(r["tokens"] for r in self.records)
        return {
            "total_requests": len(self.records),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "by_product": dict(self.by_product),
            "by_model": dict(self.by_model),
            "by_provider": dict(self.by_provider),
        }

    def get_product_usage(self, product: str) -> Dict:
        return dict(self.by_product.get(product, {}))


# ═══════════════════════════════════════════════════════
# Smart Router
# ═══════════════════════════════════════════════════════

class SmartRouter:
    def __init__(self, models: Dict[str, ModelConfig]):
        self.models = models

    def select_model(self, req: CompletionRequest) -> List[ModelConfig]:
        candidates = [m for m in self.models.values() if m.enabled]
        if req.required_capabilities:
            candidates = [m for m in candidates if
                         all(cap in m.capabilities for cap in req.required_capabilities)]
        if req.max_cost_per_request > 0:
            estimated_tokens = req.max_tokens + sum(len(m.get("content", "")) for m in req.messages) * 1.3
            candidates = [m for m in candidates if
                         (m.cost_per_1k_input * estimated_tokens / 1000 +
                          m.cost_per_1k_output * req.max_tokens / 1000) <= req.max_cost_per_request]
        if req.provider:
            try:
                ptype = ProviderType(req.provider)
                candidates = [m for m in candidates if m.provider == ptype]
            except ValueError:
                pass
        if not candidates:
            candidates = [m for m in self.models.values() if m.enabled]

        strategy = req.routing_strategy
        if strategy == RoutingStrategy.CHEAPEST:
            candidates.sort(key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output)
        elif strategy == RoutingStrategy.FASTEST:
            candidates.sort(key=lambda m: (m.avg_latency_ms if m.avg_latency_ms > 0 else 999999))
        elif strategy == RoutingStrategy.BEST_QUALITY:
            def quality_rank(m):
                try:
                    return QUALITY_RANKING.index(m.id)
                except ValueError:
                    return 100
            candidates.sort(key=quality_rank)
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            candidates.sort(key=lambda m: m.total_requests)
        elif strategy == RoutingStrategy.CAPABILITY:
            def cap_score(m):
                return sum(1 for cap in req.required_capabilities if cap in m.capabilities)
            candidates.sort(key=cap_score, reverse=True)
        elif strategy == RoutingStrategy.LOCAL_FIRST:
            def local_quality_rank(m):
                if not m.is_local:
                    return 1000  # Cloud models go last
                try:
                    return LOCAL_QUALITY_RANKING.index(m.id)
                except ValueError:
                    return 500  # Unknown local models before cloud
            candidates.sort(key=local_quality_rank)
        else:  # FALLBACK — also local-first by default
            def fallback_rank(m):
                if not m.is_local:
                    return 1000
                try:
                    return LOCAL_QUALITY_RANKING.index(m.id)
                except ValueError:
                    return 500
            candidates.sort(key=fallback_rank)
        return candidates


# ═══════════════════════════════════════════════════════
# Product Presets — Every Kaashmikhaa Product
# ═══════════════════════════════════════════════════════

PRODUCT_PRESETS: Dict[str, Dict] = {
    "opswatch": {
        "default_model": "ollama/qwen3:8b",
        "routing_strategy": RoutingStrategy.LOCAL_FIRST,
        "required_capabilities": [ModelCapability.CHAT, ModelCapability.REASONING],
        "max_tokens": 2048, "temperature": 0.3,
        "description": "OpsWatch Unified - Monitoring AI Analyst (Local)",
    },
    "opshiftpro": {
        "default_model": "ollama/qwen3:8b",
        "routing_strategy": RoutingStrategy.LOCAL_FIRST,
        "required_capabilities": [ModelCapability.CHAT],
        "max_tokens": 2048, "temperature": 0.4,
        "description": "OpsShiftPro - L2 Support AI Assistant (Local)",
    },
    "opshiftpro-mobile": {
        "default_model": "ollama/qwen3:4b",
        "routing_strategy": RoutingStrategy.LOCAL_FIRST,
        "required_capabilities": [ModelCapability.CHAT, ModelCapability.FAST],
        "max_tokens": 1024, "temperature": 0.3,
        "description": "OpsShiftPro Mobile - Quick AI responses (Local)",
    },
    "valluvan": {
        "default_model": "ollama/qwen3:8b",
        "routing_strategy": RoutingStrategy.LOCAL_FIRST,
        "required_capabilities": [ModelCapability.CHAT, ModelCapability.REASONING],
        "max_tokens": 4096, "temperature": 0.6,
        "description": "Valluvan Astrologer - Vedic AI predictions (Local)",
    },
    "vault-browser": {
        "default_model": "ollama/qwen3:4b",
        "routing_strategy": RoutingStrategy.LOCAL_FIRST,
        "required_capabilities": [ModelCapability.CHAT, ModelCapability.CHEAP],
        "max_tokens": 1024, "temperature": 0.3,
        "description": "Vault Browser - Privacy AI (100% local, zero data leaves device)",
    },
    "kaasai-agent": {
        "default_model": "ollama/qwen3:8b",
        "routing_strategy": RoutingStrategy.LOCAL_FIRST,
        "required_capabilities": [ModelCapability.CHAT, ModelCapability.CODE],
        "max_tokens": 4096, "temperature": 0.3,
        "description": "KaasAI Agent - Autonomous task execution (Local)",
    },
    "kaasai-ide": {
        "default_model": "ollama/qwen2.5-coder:7b",
        "routing_strategy": RoutingStrategy.LOCAL_FIRST,
        "required_capabilities": [ModelCapability.CHAT, ModelCapability.CODE],
        "max_tokens": 4096, "temperature": 0.3,
        "description": "KaasAI IDE - Multi-agent development (Local)",
    },
    "openclaw": {
        "default_model": "ollama/qwen3:8b",
        "routing_strategy": RoutingStrategy.LOCAL_FIRST,
        "required_capabilities": [ModelCapability.CHAT],
        "max_tokens": 8192, "temperature": 0.7,
        "description": "OpenClaw - Routes to best available local model automatically",
    },
}


# ═══════════════════════════════════════════════════════
# Neural Brain — Main Engine (same as v1, full code)
# ═══════════════════════════════════════════════════════

class NeuralBrain:
    def __init__(self):
        self.models: Dict[str, ModelConfig] = dict(DEFAULT_MODELS)
        self.providers: Dict[ProviderType, ProviderConfig] = {}
        self.router = SmartRouter(self.models)
        self.cache = PromptCache()
        self.usage = UsageTracker()
        self.product_presets = dict(PRODUCT_PRESETS)
        self._rate_counters: Dict[str, List[float]] = defaultdict(list)
        self._sessions: Dict[str, Any] = {}  # Connection pool: provider → aiohttp.ClientSession
        self._installed_ollama_models: set = set()  # Track what's actually installed
        self._auto_configure()

    async def _get_session(self, provider: str = "default") -> Any:
        """Reuse aiohttp sessions per provider for connection pooling."""
        import aiohttp
        session = self._sessions.get(provider)
        if session is None or session.closed:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=30, ttl_dns_cache=300)
            session = aiohttp.ClientSession(connector=connector)
            self._sessions[provider] = session
        return session

    async def close(self):
        """Cleanup all sessions on shutdown."""
        for session in self._sessions.values():
            if session and not session.closed:
                await session.close()
        self._sessions.clear()

    def _auto_configure(self):
        env_map = {
            ProviderType.ANTHROPIC: ("ANTHROPIC_API_KEY", "https://api.anthropic.com"),
            ProviderType.OPENAI: ("OPENAI_API_KEY", "https://api.openai.com/v1"),
            ProviderType.GOOGLE: ("GOOGLE_API_KEY", "https://generativelanguage.googleapis.com/v1beta"),
            ProviderType.GROQ: ("GROQ_API_KEY", "https://api.groq.com/openai/v1"),
            ProviderType.OPENROUTER: ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
            ProviderType.TOGETHER: ("TOGETHER_API_KEY", "https://api.together.xyz/v1"),
            ProviderType.HUGGINGFACE: ("HF_API_KEY", "https://api-inference.huggingface.co"),
            ProviderType.MISTRAL: ("MISTRAL_API_KEY", "https://api.mistral.ai/v1"),
            ProviderType.XAI: ("XAI_API_KEY", "https://api.x.ai/v1"),
            ProviderType.DEEPSEEK: ("DEEPSEEK_API_KEY", "https://api.deepseek.com"),
            ProviderType.MINIMAX: ("MINIMAX_API_KEY", "https://api.minimax.io/v1"),
        }
        for ptype, (env_var, base_url) in env_map.items():
            key = os.getenv(env_var, "")
            if key:
                self.configure_provider(ptype.value, api_key=key, base_url=base_url)

        # Auto-detect Ollama
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.providers[ProviderType.OLLAMA] = ProviderConfig(
            type=ProviderType.OLLAMA, name="Ollama (Local)",
            base_url=ollama_url, is_local=True,
            models=[m for m in self.models.values() if m.provider == ProviderType.OLLAMA],
        )
        # Auto-detect LM Studio
        self.providers[ProviderType.LM_STUDIO] = ProviderConfig(
            type=ProviderType.LM_STUDIO, name="LM Studio (Local)",
            base_url=os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1"),
            is_local=True,
            models=[m for m in self.models.values() if m.provider == ProviderType.LM_STUDIO],
        )

    def configure_provider(self, provider: str, api_key: str = "",
                          base_url: str = "", **kwargs) -> ProviderConfig:
        ptype = ProviderType(provider)
        config = ProviderConfig(
            type=ptype, name=ptype.value.replace("_", " ").title(),
            base_url=base_url, api_key=api_key,
            is_local=ptype in (ProviderType.OLLAMA, ProviderType.LM_STUDIO, ProviderType.VLLM),
            models=[m for m in self.models.values() if m.provider == ptype], **kwargs,
        )
        self.providers[ptype] = config
        for m in self.models.values():
            if m.provider == ptype:
                m.enabled = True
        logger.info(f"Configured provider: {ptype.value} ({len(config.models)} models)")
        return config

    def add_custom_model(self, model_id: str, provider: str, name: str,
                        endpoint: str = "", **kwargs) -> ModelConfig:
        ptype = ProviderType(provider) if provider in [p.value for p in ProviderType] else ProviderType.CUSTOM
        model = ModelConfig(id=model_id, provider=ptype, name=name, endpoint=endpoint, **kwargs)
        self.models[model_id] = model
        self.router = SmartRouter(self.models)
        return model

    def register_product(self, product_name: str, preset: Dict):
        self.product_presets[product_name] = preset

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        if req.product and req.product in self.product_presets:
            preset = self.product_presets[req.product]
            if not req.model:
                req.model = preset.get("default_model", "")
            if not req.required_capabilities:
                req.required_capabilities = preset.get("required_capabilities", [])
            if req.routing_strategy == RoutingStrategy.LOCAL_FIRST:
                req.routing_strategy = preset.get("routing_strategy", RoutingStrategy.LOCAL_FIRST)

        cached = self.cache.get(req)
        if cached:
            self.usage.record(req, cached)
            return cached

        if req.model and req.model in self.models:
            model = self.models[req.model]
            if model.enabled:
                try:
                    resp = await self._call_provider(model, req)
                    self.cache.set(req, resp)
                    self.usage.record(req, resp)
                    return resp
                except Exception as e:
                    logger.warning(f"Model {req.model} failed: {e}")
                    self.usage.record_error(req, model.provider.value, model.id, str(e))

        candidates = self.router.select_model(req)
        last_error = None
        for model in candidates[:7]:  # Try up to 7 models
            provider = self.providers.get(model.provider)
            if not provider or not provider.enabled:
                continue
            if not self._check_rate_limit(model.provider.value, provider.rate_limit_rpm):
                continue
            try:
                resp = await self._call_provider(model, req)
                self.cache.set(req, resp)
                self.usage.record(req, resp)
                model.total_requests += 1
                model.total_tokens += resp.usage.get("total_tokens", 0)
                model.total_cost += resp.cost
                model.avg_latency_ms = (model.avg_latency_ms * 0.9 + resp.latency_ms * 0.1) if model.avg_latency_ms > 0 else resp.latency_ms
                return resp
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {model.provider.value}/{model.id} failed: {e}")
                self.usage.record_error(req, model.provider.value, model.id, str(e))
                model.success_rate = max(0, model.success_rate - 5)
                continue
        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    async def complete_stream(self, req: CompletionRequest) -> AsyncGenerator[str, None]:
        req.stream = True
        model_id = req.model or "ollama/qwen3:8b"
        model = self.models.get(model_id)
        if not model:
            candidates = self.router.select_model(req)
            model = candidates[0] if candidates else list(self.models.values())[0]
        async for chunk in self._stream_provider(model, req):
            yield chunk

    async def embed(self, req: EmbeddingRequest) -> EmbeddingResponse:
        import aiohttp
        model_id = req.model

        # Local-first: try Ollama embedding models before cloud
        local_embed_models = ["ollama/qwen3-embedding", "ollama/nomic-embed-text", "ollama/mxbai-embed-large"]
        if not model_id or model_id in local_embed_models or model_id.startswith("ollama/"):
            ollama_model = model_id.replace("ollama/", "") if model_id else "qwen3-embedding"
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            try:
                session = await self._get_session("ollama")
                all_embeddings = []
                for text in req.texts:
                    resp = await session.post(
                        f"{ollama_url}/api/embed",
                        json={"model": ollama_model, "input": text},
                        timeout=aiohttp.ClientTimeout(total=60),
                    )
                    data = await resp.json()
                    if "embeddings" in data and data["embeddings"]:
                        all_embeddings.append(data["embeddings"][0])
                    elif "embedding" in data:
                        all_embeddings.append(data["embedding"])
                    else:
                        raise RuntimeError(f"Ollama embed failed: {data}")
                return EmbeddingResponse(
                    embeddings=all_embeddings,
                    model=f"ollama/{ollama_model}", provider="ollama", usage={},
                    dimensions=len(all_embeddings[0]) if all_embeddings else 0,
                )
            except Exception as e:
                if model_id and model_id.startswith("ollama/"):
                    raise
                logger.warning(f"Local embedding failed, falling back to cloud: {e}")

        # Fallback to OpenAI embeddings
        model_id = model_id or "text-embedding-3-small"
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("No local embedding model available and no OPENAI_API_KEY set. "
                             "Install an Ollama embedding model: ollama pull nomic-embed-text")
        session = await self._get_session("openai")
        resp = await session.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model_id, "input": req.texts},
            timeout=aiohttp.ClientTimeout(total=60),
        )
        data = await resp.json()
        return EmbeddingResponse(
            embeddings=[d["embedding"] for d in data.get("data", [])],
            model=model_id, provider="openai", usage=data.get("usage", {}),
            dimensions=len(data["data"][0]["embedding"]) if data.get("data") else 0,
        )

    async def _call_provider(self, model: ModelConfig, req: CompletionRequest) -> CompletionResponse:
        start = time.time()
        request_id = req.request_id or hashlib.md5(f"{time.time()}".encode()).hexdigest()[:12]
        if model.provider == ProviderType.ANTHROPIC:
            resp = await self._call_anthropic(model, req)
        elif model.provider == ProviderType.GOOGLE:
            resp = await self._call_google(model, req)
        elif model.provider in (ProviderType.OLLAMA, ProviderType.LM_STUDIO, ProviderType.VLLM):
            resp = await self._call_local(model, req)
        else:
            # OpenAI-compatible: OpenAI, Groq, Mistral, xAI, DeepSeek, MiniMax, OpenRouter, Together
            resp = await self._call_openai_compatible(model, req)
        resp.latency_ms = (time.time() - start) * 1000
        resp.request_id = request_id
        resp.cost = self._calculate_cost(model, resp.usage)
        return resp

    async def _call_anthropic(self, model: ModelConfig, req: CompletionRequest) -> CompletionResponse:
        import aiohttp
        provider = self.providers.get(ProviderType.ANTHROPIC)
        key = provider.api_key if provider else os.getenv("ANTHROPIC_API_KEY", "")
        body = {"model": model.id, "max_tokens": req.max_tokens,
                "temperature": req.temperature, "messages": req.messages}
        if req.system: body["system"] = req.system
        if req.tools: body["tools"] = req.tools
        if req.stop: body["stop_sequences"] = req.stop
        session = await self._get_session("anthropic")
        resp = await session.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json=body, timeout=aiohttp.ClientTimeout(total=120),
        )
        data = await resp.json()
        if resp.status != 200:
            raise RuntimeError(f"Anthropic error {resp.status}: {data.get('error', {}).get('message', str(data))}")
        content = ""
        tool_calls = []
        for block in data.get("content", []):
            if block["type"] == "text": content += block["text"]
            elif block["type"] == "tool_use": tool_calls.append(block)
        return CompletionResponse(
            id=data.get("id", ""), content=content, model=model.id, provider="anthropic",
            usage={"prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                   "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
                   "total_tokens": data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0)},
            finish_reason=data.get("stop_reason", "end_turn"), tool_calls=tool_calls,
        )

    async def _call_google(self, model: ModelConfig, req: CompletionRequest) -> CompletionResponse:
        import aiohttp
        key = os.getenv("GOOGLE_API_KEY", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model.id}:generateContent?key={key}"
        contents = [{"role": "user" if msg["role"] == "user" else "model",
                     "parts": [{"text": msg["content"]}]} for msg in req.messages]
        body = {"contents": contents, "generationConfig": {"temperature": req.temperature, "maxOutputTokens": req.max_tokens}}
        if req.system: body["systemInstruction"] = {"parts": [{"text": req.system}]}
        session = await self._get_session("google")
        resp = await session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=120))
        data = await resp.json()
        if "error" in data:
            raise RuntimeError(f"Google error: {data['error'].get('message', str(data))}")
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        return CompletionResponse(
            id="", content=content, model=model.id, provider="google",
            usage={"prompt_tokens": usage.get("promptTokenCount", 0),
                   "completion_tokens": usage.get("candidatesTokenCount", 0),
                   "total_tokens": usage.get("totalTokenCount", 0)},
        )

    async def _call_local(self, model: ModelConfig, req: CompletionRequest) -> CompletionResponse:
        import aiohttp
        if model.provider == ProviderType.OLLAMA:
            endpoint = model.endpoint or "http://localhost:11434"
            model_name = model.id.replace("ollama/", "")
            messages = []
            if req.system: messages.append({"role": "system", "content": req.system})
            messages.extend(req.messages)
            body = {"model": model_name, "messages": messages, "stream": False,
                    "options": {"temperature": req.temperature, "num_predict": req.max_tokens}}
            session = await self._get_session("ollama")
            resp = await session.post(f"{endpoint}/api/chat", json=body,
                                     timeout=aiohttp.ClientTimeout(total=300))
            data = await resp.json()
            return CompletionResponse(
                id="", content=data.get("message", {}).get("content", ""),
                model=model.id, provider="ollama",
                usage={"prompt_tokens": data.get("prompt_eval_count", 0),
                       "completion_tokens": data.get("eval_count", 0),
                       "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)},
            )
        else:
            return await self._call_openai_compatible(model, req)

    async def _call_openai_compatible(self, model: ModelConfig, req: CompletionRequest) -> CompletionResponse:
        import aiohttp
        provider = self.providers.get(model.provider)
        base_url = model.endpoint or (provider.base_url if provider else "")
        key = (provider.api_key if provider else "") or os.getenv(model.api_key_env, "")
        messages = []
        if req.system: messages.append({"role": "system", "content": req.system})
        messages.extend(req.messages)
        model_id = model.id.split("/")[-1] if "/" in model.id else model.id
        body = {"model": model_id, "messages": messages,
                "temperature": req.temperature, "max_tokens": req.max_tokens}
        if req.tools:
            body["tools"] = [{"type": "function", "function": t} for t in req.tools]
        headers = {"Content-Type": "application/json"}
        if key: headers["Authorization"] = f"Bearer {key}"
        if provider and provider.headers: headers.update(provider.headers)
        session = await self._get_session(model.provider.value)
        resp = await session.post(f"{base_url}/chat/completions", headers=headers, json=body,
                                 timeout=aiohttp.ClientTimeout(total=provider.timeout_seconds if provider else 120))
        data = await resp.json()
        if resp.status != 200:
            raise RuntimeError(f"{model.provider.value} error: {data}")
        choice = data["choices"][0]
        return CompletionResponse(
            id=data.get("id", ""), content=choice["message"].get("content", ""),
            model=model.id, provider=model.provider.value,
            usage=data.get("usage", {}), finish_reason=choice.get("finish_reason", "stop"),
        )

    async def _stream_provider(self, model: ModelConfig, req: CompletionRequest) -> AsyncGenerator[str, None]:
        import aiohttp
        if model.provider == ProviderType.ANTHROPIC:
            provider = self.providers.get(ProviderType.ANTHROPIC)
            key = provider.api_key if provider else os.getenv("ANTHROPIC_API_KEY", "")
            body = {"model": model.id, "max_tokens": req.max_tokens,
                    "temperature": req.temperature, "messages": req.messages, "stream": True}
            if req.system: body["system"] = req.system
            session = await self._get_session("anthropic")
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json=body,
            ) as resp:
                async for line in resp.content:
                    text = line.decode("utf-8").strip()
                    if text.startswith("data: "):
                        try:
                            event = json.loads(text[6:])
                            if event.get("type") == "content_block_delta":
                                yield event["delta"].get("text", "")
                        except json.JSONDecodeError:
                            pass

        elif model.provider == ProviderType.OLLAMA:
            # Native Ollama streaming — faster than OpenAI-compatible format
            endpoint = model.endpoint or "http://localhost:11434"
            model_name = model.id.replace("ollama/", "")
            messages = []
            if req.system: messages.append({"role": "system", "content": req.system})
            messages.extend(req.messages)
            body = {"model": model_name, "messages": messages, "stream": True,
                    "options": {"temperature": req.temperature, "num_predict": req.max_tokens}}
            session = await self._get_session("ollama")
            async with session.post(f"{endpoint}/api/chat", json=body,
                                    timeout=aiohttp.ClientTimeout(total=300)) as resp:
                async for line in resp.content:
                    text = line.decode("utf-8").strip()
                    if not text:
                        continue
                    try:
                        chunk = json.loads(text)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        pass

        else:
            provider = self.providers.get(model.provider)
            base_url = model.endpoint or (provider.base_url if provider else "https://api.openai.com/v1")
            key = (provider.api_key if provider else "") or os.getenv(model.api_key_env, "")
            messages = []
            if req.system: messages.append({"role": "system", "content": req.system})
            messages.extend(req.messages)
            model_id = model.id.split("/")[-1] if "/" in model.id else model.id
            body = {"model": model_id, "messages": messages, "temperature": req.temperature,
                    "max_tokens": req.max_tokens, "stream": True}
            headers = {"Content-Type": "application/json"}
            if key: headers["Authorization"] = f"Bearer {key}"
            session = await self._get_session(model.provider.value)
            async with session.post(f"{base_url}/chat/completions", headers=headers, json=body) as resp:
                async for line in resp.content:
                    text = line.decode("utf-8").strip()
                    if text.startswith("data: ") and text != "data: [DONE]":
                        try:
                            chunk = json.loads(text[6:])
                            delta = chunk["choices"][0].get("delta", {}).get("content", "")
                            if delta: yield delta
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass

    def _calculate_cost(self, model: ModelConfig, usage: Dict) -> float:
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        return (model.cost_per_1k_input * input_tokens / 1000 +
                model.cost_per_1k_output * output_tokens / 1000)

    def _check_rate_limit(self, provider: str, rpm: int) -> bool:
        now = time.time()
        self._rate_counters[provider] = [t for t in self._rate_counters[provider] if now - t < 60]
        if len(self._rate_counters[provider]) >= rpm:
            return False
        self._rate_counters[provider].append(now)
        return True

    async def discover_ollama_models(self) -> List[str]:
        import aiohttp
        url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        try:
            session = await self._get_session("ollama")
            resp = await session.get(f"{url}/api/tags", timeout=aiohttp.ClientTimeout(total=5))
            data = await resp.json()
            models = []
            for m in data.get("models", []):
                name = m["name"]
                model_id = f"ollama/{name}"
                self._installed_ollama_models.add(model_id)
                if model_id not in self.models:
                    self.add_custom_model(
                        model_id=model_id, provider="ollama", name=f"{name} (Local)",
                        endpoint=url, is_local=True,
                        capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING, ModelCapability.CHEAP],
                    )
                models.append(model_id)
            logger.info(f"Ollama: {len(models)} models installed")
            return models
        except Exception as e:
            logger.warning(f"Ollama discovery failed: {e}")
            return []

    async def auto_pull_models(self, models: List[str] = None) -> List[str]:
        """Auto-pull essential Ollama models that aren't installed yet."""
        import aiohttp
        url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        # Default: pull the most efficient models for each tier
        if not models:
            models = [
                "qwen3:8b",        # Best general 8B — sweet spot
                "qwen3:4b",        # Fast lightweight
                "qwen2.5-coder:7b",# Code specialist
                "deepseek-r1:8b",  # Reasoning
                "nomic-embed-text", # Embeddings
            ]

        pulled = []
        for model_name in models:
            model_id = f"ollama/{model_name}"
            if model_id in self._installed_ollama_models:
                continue
            try:
                logger.info(f"Auto-pulling model: {model_name}")
                session = await self._get_session("ollama")
                resp = await session.post(
                    f"{url}/api/pull",
                    json={"name": model_name, "stream": False},
                    timeout=aiohttp.ClientTimeout(total=1800),  # 30 min for large models
                )
                if resp.status == 200:
                    self._installed_ollama_models.add(model_id)
                    pulled.append(model_name)
                    logger.info(f"Pulled: {model_name}")
                else:
                    logger.warning(f"Failed to pull {model_name}: HTTP {resp.status}")
            except Exception as e:
                logger.warning(f"Auto-pull {model_name} failed: {e}")
        return pulled

    async def warmup_models(self, models: List[str] = None) -> List[str]:
        """Send a tiny request to pre-load models into memory."""
        if not models:
            # Warm up the most commonly used models
            models = ["ollama/qwen3:8b", "ollama/qwen3:4b"]

        warmed = []
        for model_id in models:
            if model_id not in self._installed_ollama_models and model_id in self.models:
                # Model is in catalog but we don't know if installed — skip
                continue
            try:
                req = CompletionRequest(
                    messages=[{"role": "user", "content": "hi"}],
                    model=model_id, max_tokens=1, temperature=0,
                )
                await self.complete(req)
                warmed.append(model_id)
                logger.info(f"Warmed up: {model_id}")
            except Exception:
                pass
        return warmed

    def get_status(self) -> Dict:
        return {
            "version": "3.0.0",
            "total_models": len(self.models),
            "total_providers": len(self.providers),
            "categories": {
                "general": len([m for m in self.models.values() if m.category == "general"]),
                "code": len([m for m in self.models.values() if m.category == "code"]),
                "reasoning": len([m for m in self.models.values() if m.category == "reasoning"]),
                "vision": len([m for m in self.models.values() if m.category == "vision"]),
                "audio": len([m for m in self.models.values() if m.category == "audio"]),
                "embedding": len([m for m in self.models.values() if m.category == "embedding"]),
            },
            "providers": {
                p.value: {"name": c.name, "enabled": c.enabled, "is_local": c.is_local,
                          "models": len(c.models), "base_url": c.base_url}
                for p, c in self.providers.items() if c.enabled
            },
            "models": {
                m.id: {"name": m.name, "provider": m.provider.value, "enabled": m.enabled,
                       "is_local": m.is_local, "category": m.category,
                       "context_window": m.context_window,
                       "cost_input": m.cost_per_1k_input, "cost_output": m.cost_per_1k_output,
                       "requests": m.total_requests,
                       "capabilities": [c.value for c in m.capabilities]}
                for m in self.models.values() if m.enabled
            },
            "products": {k: v.get("description", "") for k, v in self.product_presets.items()},
            "cache": self.cache.stats(),
            "usage_summary": self.usage.get_summary(),
        }

    def list_models(self, provider: str = None, capability: str = None,
                    local_only: bool = False, category: str = None) -> List[Dict]:
        models = list(self.models.values())
        if provider: models = [m for m in models if m.provider.value == provider]
        if capability:
            cap = ModelCapability(capability)
            models = [m for m in models if cap in m.capabilities]
        if local_only: models = [m for m in models if m.is_local]
        if category: models = [m for m in models if m.category == category]
        return [{"id": m.id, "name": m.name, "provider": m.provider.value,
                 "is_local": m.is_local, "category": m.category,
                 "cost": m.cost_per_1k_input + m.cost_per_1k_output,
                 "capabilities": [c.value for c in m.capabilities]} for m in models]


# Singleton
brain = NeuralBrain()

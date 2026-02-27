"""
NEXUS Model Router
===================
Dynamic model selection across 15+ providers.
Extends JARVIS's HuggingFace-only approach to be truly provider-agnostic.

Routing strategies:
- fastest:      Lowest latency model
- cheapest:     Lowest cost per token
- best_quality: Highest capability score
- capability:   Match model capabilities to task type
- adaptive:     Self-learning route based on historical performance
- fallback:     Try providers in order until one succeeds
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from nexus.core.config import NexusConfig, RoutingStrategy


@dataclass
class ModelProfile:
    """Profile of a model's capabilities and performance."""
    name: str
    provider: str
    capabilities: list[str] = field(default_factory=list)
    max_tokens: int = 4096
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    avg_latency_ms: float = 1000
    quality_score: float = 0.8  # 0-1
    context_window: int = 128000
    supports_tools: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True

    # Adaptive stats
    total_requests: int = 0
    total_successes: int = 0
    total_failures: int = 0
    avg_quality: float = 0.8
    last_used: float = 0


# Default model profiles for major providers
DEFAULT_MODELS: dict[str, ModelProfile] = {
    "claude-opus-4-6": ModelProfile(
        name="claude-opus-4-6", provider="anthropic",
        capabilities=["reasoning", "code", "analysis", "creative", "vision", "math"],
        cost_per_1k_input=0.015, cost_per_1k_output=0.075,
        quality_score=0.98, context_window=200000,
        supports_vision=True,
    ),
    "claude-sonnet-4-6": ModelProfile(
        name="claude-sonnet-4-6", provider="anthropic",
        capabilities=["reasoning", "code", "analysis", "creative", "vision"],
        cost_per_1k_input=0.003, cost_per_1k_output=0.015,
        quality_score=0.93, context_window=200000,
        supports_vision=True,
    ),
    "claude-haiku-4-5": ModelProfile(
        name="claude-haiku-4-5", provider="anthropic",
        capabilities=["general", "code", "fast"],
        cost_per_1k_input=0.0008, cost_per_1k_output=0.004,
        quality_score=0.85, avg_latency_ms=400, context_window=200000,
    ),
    "gpt-4o": ModelProfile(
        name="gpt-4o", provider="openai",
        capabilities=["reasoning", "code", "vision", "creative"],
        cost_per_1k_input=0.005, cost_per_1k_output=0.015,
        quality_score=0.92, context_window=128000,
        supports_vision=True,
    ),
    "gpt-4o-mini": ModelProfile(
        name="gpt-4o-mini", provider="openai",
        capabilities=["general", "code", "fast"],
        cost_per_1k_input=0.00015, cost_per_1k_output=0.0006,
        quality_score=0.82, avg_latency_ms=500, context_window=128000,
    ),
    "gemini-2.0-flash": ModelProfile(
        name="gemini-2.0-flash", provider="google",
        capabilities=["general", "code", "fast", "vision"],
        cost_per_1k_input=0.0001, cost_per_1k_output=0.0004,
        quality_score=0.85, avg_latency_ms=300, context_window=1000000,
        supports_vision=True,
    ),
    "deepseek-r1": ModelProfile(
        name="deepseek-r1", provider="deepseek",
        capabilities=["reasoning", "math", "code"],
        cost_per_1k_input=0.00055, cost_per_1k_output=0.0022,
        quality_score=0.90, context_window=64000,
    ),
    "llama-3.3-70b": ModelProfile(
        name="llama-3.3-70b", provider="groq",
        capabilities=["general", "code", "fast"],
        cost_per_1k_input=0.00059, cost_per_1k_output=0.00079,
        quality_score=0.87, avg_latency_ms=200, context_window=128000,
    ),
    "qwen-2.5-72b": ModelProfile(
        name="qwen-2.5-72b", provider="together",
        capabilities=["general", "code", "reasoning"],
        cost_per_1k_input=0.0012, cost_per_1k_output=0.0012,
        quality_score=0.86, context_window=128000,
    ),
    "mistral-large": ModelProfile(
        name="mistral-large", provider="mistral",
        capabilities=["general", "code", "multilingual"],
        cost_per_1k_input=0.002, cost_per_1k_output=0.006,
        quality_score=0.88, context_window=128000,
    ),
}


class ModelRouter:
    """
    Intelligent model selection and request routing.

    Selects the optimal model based on:
    - Task requirements (type, complexity)
    - Routing strategy (fastest, cheapest, best, adaptive)
    - Historical performance data
    - Provider availability
    - Cost constraints
    """

    def __init__(self, config: NexusConfig):
        self.config = config
        self.models = dict(DEFAULT_MODELS)
        self._performance_log: list[dict] = []

    async def select_model(
        self,
        task: str = "",
        task_type: str = "general",
        strategy: RoutingStrategy | None = None,
        max_cost_per_1k: float | None = None,
        requires_vision: bool = False,
        requires_tools: bool = False,
        min_quality: float = 0.0,
    ) -> str:
        """Select the best model for a task."""
        strategy = strategy or self.config.routing_strategy

        # Filter available models
        candidates = list(self.models.values())

        # Apply hard filters
        if requires_vision:
            candidates = [m for m in candidates if m.supports_vision]
        if requires_tools:
            candidates = [m for m in candidates if m.supports_tools]
        if max_cost_per_1k:
            candidates = [m for m in candidates if m.cost_per_1k_input <= max_cost_per_1k]
        if min_quality:
            candidates = [m for m in candidates if m.quality_score >= min_quality]

        # Filter by provider availability
        candidates = [
            m for m in candidates
            if m.provider in self.config.providers and self.config.providers[m.provider].enabled
        ]

        if not candidates:
            return self.config.default_model

        # Apply routing strategy
        if strategy == RoutingStrategy.FASTEST:
            selected = min(candidates, key=lambda m: m.avg_latency_ms)
        elif strategy == RoutingStrategy.CHEAPEST:
            selected = min(candidates, key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output)
        elif strategy == RoutingStrategy.BEST_QUALITY:
            selected = max(candidates, key=lambda m: m.quality_score)
        elif strategy == RoutingStrategy.CAPABILITY:
            selected = self._select_by_capability(candidates, task_type)
        elif strategy == RoutingStrategy.ADAPTIVE:
            selected = self._select_adaptive(candidates, task_type)
        else:
            selected = candidates[0]

        return selected.name

    def _select_by_capability(self, candidates: list[ModelProfile], task_type: str) -> ModelProfile:
        """Select model based on capability match."""
        scored = []
        for model in candidates:
            score = model.quality_score
            if task_type in model.capabilities:
                score += 0.2
            if "general" in model.capabilities:
                score += 0.05
            scored.append((score, model))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else candidates[0]

    def _select_adaptive(self, candidates: list[ModelProfile], task_type: str) -> ModelProfile:
        """Select model based on historical performance for this task type."""
        scored = []
        for model in candidates:
            # Base quality score
            score = model.quality_score

            # Bonus for capability match
            if task_type in model.capabilities:
                score += 0.1

            # Adaptive: weight by historical success rate
            if model.total_requests > 0:
                success_rate = model.total_successes / model.total_requests
                score = score * 0.5 + success_rate * 0.3 + model.avg_quality * 0.2

            scored.append((score, model))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else candidates[0]

    async def generate(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        **kwargs,
    ) -> dict:
        """
        Generate a response from a model.
        Routes to the appropriate provider adapter.
        """
        profile = self.models.get(model)
        if not profile:
            # Try to find by partial match
            for name, p in self.models.items():
                if model in name:
                    profile = p
                    break

        provider_name = profile.provider if profile else "anthropic"
        provider_config = self.config.providers.get(provider_name)

        if not provider_config:
            return {"content": f"Provider '{provider_name}' not configured", "tool_calls": []}

        start_time = time.time()

        try:
            # Import provider adapters dynamically to avoid circular imports
            from nexus.models.providers import ProviderManager
            result = await ProviderManager.call_provider(
                provider_name=provider_name,
                provider_config=provider_config,
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )

            # Update adaptive stats
            if profile:
                profile.total_requests += 1
                profile.total_successes += 1
                profile.last_used = time.time()
                latency = (time.time() - start_time) * 1000
                profile.avg_latency_ms = (profile.avg_latency_ms + latency) / 2

            return result

        except Exception as e:
            if profile:
                profile.total_requests += 1
                profile.total_failures += 1

            # Fallback
            return {"content": f"Generation error: {e}", "tool_calls": []}

    @property
    def available_models(self) -> list[str]:
        """List of available model names."""
        return list(self.models.keys())

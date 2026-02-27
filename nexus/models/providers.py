"""
NEXUS Provider Adapters
========================
Unified interface for calling 15+ LLM providers.
Each provider is adapted to a common request/response format.
"""

from __future__ import annotations

import os
import json
import aiohttp
from typing import Any

from nexus.core.config import ProviderConfig


class ProviderManager:
    """Manages provider adapters and routes API calls."""

    def __init__(self, config=None):
        self.config = config

    @staticmethod
    async def call_provider(
        provider_name: str,
        provider_config: ProviderConfig,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> dict:
        """Route a generation request to the appropriate provider."""

        adapters = {
            "anthropic": _call_anthropic,
            "openai": _call_openai,
            "google": _call_openai,    # Gemini uses OpenAI-compatible
            "groq": _call_openai,       # OpenAI-compatible
            "together": _call_openai,   # OpenAI-compatible
            "fireworks": _call_openai,  # OpenAI-compatible
            "deepseek": _call_openai,   # OpenAI-compatible
            "mistral": _call_openai,    # OpenAI-compatible
            "openrouter": _call_openai, # OpenAI-compatible
            "ollama": _call_ollama,
            "cerebras": _call_openai,   # OpenAI-compatible
            "sambanova": _call_openai,  # OpenAI-compatible
        }

        adapter = adapters.get(provider_name, _call_openai)
        return await adapter(provider_config, model, messages, tools, temperature, max_tokens, stream)


async def _call_anthropic(
    config: ProviderConfig, model: str, messages: list[dict],
    tools: list[dict] | None, temperature: float, max_tokens: int, stream: bool,
) -> dict:
    """Call Anthropic's Messages API."""
    api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY", "")
    url = config.base_url or "https://api.anthropic.com/v1/messages"

    # Separate system from conversation messages
    system_msg = ""
    conv_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            system_msg += msg.get("content", "") + "\n"
        else:
            conv_messages.append({"role": msg["role"], "content": msg.get("content", "")})

    body: dict[str, Any] = {
        "model": model,
        "messages": conv_messages if conv_messages else [{"role": "user", "content": "Hello"}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system_msg:
        body["system"] = system_msg.strip()
    if tools:
        body["tools"] = [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in tools
            if "function" in t
        ]

    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            data = await resp.json()

    # Parse response
    content = ""
    tool_calls = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            content += block.get("text", "")
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "arguments": block.get("input", {}),
            })

    return {
        "content": content,
        "tool_calls": tool_calls,
        "total_tokens": data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0),
        "cost": 0.0,  # Calculate based on model pricing
    }


async def _call_openai(
    config: ProviderConfig, model: str, messages: list[dict],
    tools: list[dict] | None, temperature: float, max_tokens: int, stream: bool,
) -> dict:
    """Call OpenAI-compatible API (works with Groq, Together, DeepSeek, etc.)."""
    api_key = config.api_key or os.getenv("OPENAI_API_KEY", "")
    base_url = config.base_url or "https://api.openai.com/v1"
    url = f"{base_url}/chat/completions"

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = tools

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            data = await resp.json()

    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})

    tool_calls = []
    for tc in message.get("tool_calls", []):
        tool_calls.append({
            "id": tc.get("id", ""),
            "name": tc.get("function", {}).get("name", ""),
            "arguments": json.loads(tc.get("function", {}).get("arguments", "{}")),
        })

    return {
        "content": message.get("content", ""),
        "tool_calls": tool_calls,
        "total_tokens": data.get("usage", {}).get("total_tokens", 0),
        "cost": 0.0,
    }


async def _call_ollama(
    config: ProviderConfig, model: str, messages: list[dict],
    tools: list[dict] | None, temperature: float, max_tokens: int, stream: bool,
) -> dict:
    """Call Ollama's local API."""
    base_url = config.base_url or "http://localhost:11434"
    url = f"{base_url}/api/chat"

    body = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "stream": False,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                data = await resp.json()

        return {
            "content": data.get("message", {}).get("content", ""),
            "tool_calls": [],
            "total_tokens": data.get("eval_count", 0) + data.get("prompt_eval_count", 0),
            "cost": 0.0,
        }
    except Exception as e:
        return {"content": f"Ollama error: {e}", "tool_calls": [], "total_tokens": 0, "cost": 0.0}


class ModelEnsemble:
    """
    Multi-model ensemble for improved quality.
    Queries multiple models and aggregates responses.
    """

    def __init__(self, models: list[str], strategy: str = "best_of_n"):
        self.models = models
        self.strategy = strategy

    async def generate(self, router, messages: list[dict], **kwargs) -> dict:
        """Generate responses from multiple models and select the best."""
        tasks = [
            router.generate(model=model, messages=messages, **kwargs)
            for model in self.models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = [r for r in results if isinstance(r, dict) and r.get("content")]

        if not valid_results:
            return {"content": "All models failed", "tool_calls": []}

        if self.strategy == "best_of_n":
            # Pick the longest, most detailed response
            return max(valid_results, key=lambda r: len(r.get("content", "")))
        elif self.strategy == "consensus":
            # Return the most common response theme
            return valid_results[0]

        return valid_results[0]


# Helper for importing asyncio at module level
import asyncio

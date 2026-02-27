"""
Neural Brain API — Basic Usage Examples

Zero API keys needed. Just run Ollama + Neural Brain.

Prerequisites:
    pip install requests
    ollama pull qwen3:8b        # Best 8B general model
    python -m uvicorn api.main:app --port 8200
"""
import requests

BASE = "http://localhost:8200"


def chat(message: str, **kwargs) -> str:
    """Simple chat helper — auto-routes to best local model."""
    resp = requests.post(f"{BASE}/api/v1/chat/completions", json={
        "messages": [{"role": "user", "content": message}],
        **kwargs,
    })
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ═══ 1. Basic Chat (auto-routes to best LOCAL model) ═══
print("=== Basic Chat (Local-First, No API Key) ===")
print(chat("What is the capital of France?"))


# ═══ 2. Pick a Specific Local Model ═══
print("\n=== Specific Model: Qwen3 8B ===")
print(chat("Explain recursion simply", model="ollama/qwen3:8b"))


# ═══ 3. Routing Strategies ═══
print("\n=== Local-First (default) ===")
print(chat("Hello world", routing_strategy="local_first"))

print("\n=== Best Quality (picks largest available) ===")
print(chat("Compare microservices vs monolith architecture",
           routing_strategy="best_quality"))

print("\n=== Fastest Response ===")
print(chat("What is 2+2?", routing_strategy="fastest"))


# ═══ 4. Ensemble (Smart Multi-Model Routing — All Local) ═══
print("\n=== Ensemble: Code Task → routes to coding model ===")
print(chat("Write a Python function to find prime numbers",
           model="ensemble/smart"))

print("\n=== Ensemble: Reasoning → routes to reasoning model ===")
print(chat("Why do economies experience boom and bust cycles?",
           model="ensemble/smart"))

print("\n=== Ensemble: Consensus (2 models + judge) ===")
print(chat("Should I use Redis or Memcached for session storage?",
           model="ensemble/consensus"))


# ═══ 5. Capability-Based Routing ═══
print("\n=== Route by Capability: Reasoning ===")
print(chat("Analyze this problem step by step: If all roses are flowers...",
           required_capabilities=["reasoning"]))


# ═══ 6. Product Preset (Local-First) ═══
print("\n=== Register & Use Product Preset ===")
requests.post(f"{BASE}/api/products", json={
    "product_name": "my-chatbot",
    "default_model": "ollama/qwen3:8b",
    "routing_strategy": "local_first",
    "temperature": 0.4,
    "max_tokens": 1024,
    "description": "My custom chatbot — runs 100% locally"
})
print(chat("Help me plan my day", product="my-chatbot"))


# ═══ 7. Local Embeddings (No OpenAI Key Needed) ═══
print("\n=== Local Embeddings ===")
resp = requests.post(f"{BASE}/api/v1/embeddings", json={
    "texts": ["Hello world", "Neural Brain is awesome"],
})
data = resp.json()
print(f"Embeddings: {len(data['data'])} vectors, {len(data['data'][0]['embedding'])} dimensions")


# ═══ 8. System Status ═══
print("\n=== System Status ===")
status = requests.get(f"{BASE}/api/status").json()
print(f"Version: {status['version']}")
print(f"Models: {status['total_models']}")
print(f"Providers: {status['total_providers']}")
print(f"Categories: {status['categories']}")

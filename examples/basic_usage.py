"""
Neural Brain API — Basic Usage Examples

Prerequisites:
    pip install requests
    # Start Neural Brain: python -m uvicorn api.main:app --port 8200
"""
import requests

BASE = "http://localhost:8200"


def chat(message: str, **kwargs) -> str:
    """Simple chat helper."""
    resp = requests.post(f"{BASE}/api/v1/chat/completions", json={
        "messages": [{"role": "user", "content": message}],
        **kwargs,
    })
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ═══ 1. Basic Chat (auto-routes to best available) ═══
print("=== Basic Chat ===")
print(chat("What is the capital of France?"))


# ═══ 2. Specific Model ═══
print("\n=== Specific Model ===")
print(chat("Explain recursion", model="ollama/phi4"))


# ═══ 3. Routing Strategies ═══
print("\n=== Cheapest Model ===")
print(chat("Hello world", routing_strategy="cheapest"))

print("\n=== Best Quality ===")
print(chat("Compare microservices vs monolith architecture",
           routing_strategy="best_quality"))

print("\n=== Fastest Response ===")
print(chat("What is 2+2?", routing_strategy="fastest"))


# ═══ 4. Ensemble (Smart Auto-Routing) ═══
print("\n=== Ensemble: Code Task ===")
print(chat("Write a Python function to find prime numbers",
           model="ensemble/smart"))

print("\n=== Ensemble: Reasoning Task ===")
print(chat("Why do economies experience boom and bust cycles?",
           model="ensemble/smart"))

print("\n=== Ensemble: Consensus (2 models + judge) ===")
print(chat("Should I use Redis or Memcached for session storage?",
           model="ensemble/consensus"))


# ═══ 5. Capability-Based Routing ═══
print("\n=== Route by Capability ===")
print(chat("Analyze this problem step by step: If all roses are flowers...",
           required_capabilities=["reasoning"]))


# ═══ 6. Cost-Capped Request ═══
print("\n=== Cost-Capped ($0.01 max) ===")
print(chat("Summarize the benefits of exercise",
           max_cost_per_request=0.01))


# ═══ 7. Product Preset ═══
print("\n=== Register & Use Product Preset ===")
requests.post(f"{BASE}/api/products", json={
    "product_name": "my-chatbot",
    "default_model": "ollama/phi4",
    "routing_strategy": "fallback",
    "temperature": 0.4,
    "max_tokens": 1024,
    "description": "My custom chatbot"
})
print(chat("Help me plan my day", product="my-chatbot"))


# ═══ 8. System Status ═══
print("\n=== System Status ===")
status = requests.get(f"{BASE}/api/status").json()
print(f"Models: {status['total_models']}")
print(f"Providers: {status['total_providers']}")
print(f"Categories: {status['categories']}")

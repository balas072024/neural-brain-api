"""
Neural Brain API — Ensemble Engine Examples

The ensemble engine classifies your query and routes it to
specialist models. It supports multiple orchestration modes.

Prerequisites:
    pip install requests
    # Start Neural Brain: python -m uvicorn api.main:app --port 8200
"""
import requests

BASE = "http://localhost:8200"


# ═══ 1. Query Classification ═══
queries = [
    "Write a Python function to sort a linked list",
    "Why did the Roman Empire fall?",
    "Calculate the derivative of x^3 + 2x",
    "Translate 'good morning' to Tamil",
    "Write a poem about the ocean",
    "What's in this image?",
]

print("=== Query Classification ===")
for q in queries:
    resp = requests.get(f"{BASE}/api/v1/ensemble/classify", params={"text": q}).json()
    print(f"  [{resp['query_type']:>12}] (conf={resp['confidence']:.0%}) -> {resp['specialist']}")
    print(f"                Query: {q[:60]}")


# ═══ 2. Ensemble Modes ═══

# Smart (default): classify -> route to specialist
print("\n=== Smart Mode ===")
resp = requests.post(f"{BASE}/api/v1/ensemble", json={
    "messages": [{"role": "user", "content": "Write a Redis caching decorator in Python"}],
    "mode": "smart",
}).json()
print(f"Model: {resp['model_used']}")
print(f"Type:  {resp['query_type']} (conf={resp['confidence']:.0%})")
print(f"Time:  {resp['latency_ms']:.0f}ms")

# Consensus: 2 models answer, judge picks best
print("\n=== Consensus Mode ===")
resp = requests.post(f"{BASE}/api/v1/ensemble", json={
    "messages": [{"role": "user", "content": "PostgreSQL vs MongoDB for an e-commerce app"}],
    "mode": "consensus",
}).json()
print(f"Model: {resp['model_used']}")
print(f"Models consulted: {resp['models_consulted']}")
print(f"Verified: {resp['verified']}")

# Chain: specialist -> refiner improves
print("\n=== Chain Mode ===")
resp = requests.post(f"{BASE}/api/v1/ensemble", json={
    "messages": [{"role": "user", "content": "Explain the CAP theorem with real-world examples"}],
    "mode": "chain",
}).json()
print(f"Model: {resp['model_used']}")
print(f"Verified: {resp['verified']}")

# With verification
print("\n=== Smart + Verify ===")
resp = requests.post(f"{BASE}/api/v1/ensemble", json={
    "messages": [{"role": "user", "content": "What are the tax implications of stock options?"}],
    "mode": "smart",
    "verify": True,
}).json()
print(f"Verified: {resp['verified']}")
if resp['verification_model']:
    print(f"Verified by: {resp['verification_model']}")


# ═══ 3. Ensemble Stats ═══
print("\n=== Ensemble Stats ===")
stats = requests.get(f"{BASE}/api/v1/ensemble/stats").json()
print(f"Total queries:   {stats['total_queries']}")
print(f"Single calls:    {stats['single_calls']}")
print(f"Ensemble calls:  {stats['ensemble_calls']}")
print(f"By type:         {stats['by_type']}")

"""
Neural Brain API — Ensemble Engine Examples (100% Local)

The ensemble engine classifies your query and routes it to the
best LOCAL specialist model. No API keys needed.

Query types: code, reasoning, math, vision, translation, creative, ocr, general

Prerequisites:
    pip install requests
    ollama pull qwen3:8b qwen2.5-coder:7b deepseek-r1:8b
    python -m uvicorn api.main:app --port 8200
"""
import requests

BASE = "http://localhost:8200"


# ═══ 1. Query Classification (instant, no model call) ═══
queries = [
    "Write a Python function to sort a linked list",
    "Why did the Roman Empire fall?",
    "Calculate the derivative of x^3 + 2x",
    "Translate 'good morning' to Tamil",
    "Write a poem about the ocean",
    "What's in this image?",
    "Extract text from this receipt",
    "What should I cook for dinner?",
]

print("=== Query Classification (All Routes to Local Models) ===")
for q in queries:
    resp = requests.get(f"{BASE}/api/v1/ensemble/classify", params={"text": q}).json()
    print(f"  [{resp['query_type']:>12}] (conf={resp['confidence']:.0%}) → {resp['specialist']}")
    print(f"                Query: {q[:60]}")


# ═══ 2. Ensemble Modes ═══

# Smart (default): classify → route to best local specialist
print("\n=== Smart Mode (Auto-Routes to Best Local Model) ===")
resp = requests.post(f"{BASE}/api/v1/ensemble", json={
    "messages": [{"role": "user", "content": "Write a Redis caching decorator in Python"}],
    "mode": "smart",
}).json()
print(f"Model: {resp['model_used']}")
print(f"Type:  {resp['query_type']} (conf={resp['confidence']:.0%})")
print(f"Time:  {resp['latency_ms']:.0f}ms")

# Fastest: use the fastest local judge model
print("\n=== Fastest Mode ===")
resp = requests.post(f"{BASE}/api/v1/ensemble", json={
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "mode": "fastest",
}).json()
print(f"Model: {resp['model_used']} ({resp['latency_ms']:.0f}ms)")

# Strongest: use largest available local model
print("\n=== Strongest Mode ===")
resp = requests.post(f"{BASE}/api/v1/ensemble", json={
    "messages": [{"role": "user", "content": "Explain quantum entanglement"}],
    "mode": "strongest",
}).json()
print(f"Model: {resp['model_used']} ({resp['latency_ms']:.0f}ms)")

# Consensus: 2 local models answer, local judge picks best
print("\n=== Consensus Mode (2 Models + Judge) ===")
resp = requests.post(f"{BASE}/api/v1/ensemble", json={
    "messages": [{"role": "user", "content": "PostgreSQL vs MongoDB for an e-commerce app"}],
    "mode": "consensus",
}).json()
print(f"Model: {resp['model_used']}")
print(f"Models consulted: {resp['models_consulted']}")
print(f"Verified: {resp['verified']}")

# Chain: specialist answers → refiner improves
print("\n=== Chain Mode (Specialist → Refiner) ===")
resp = requests.post(f"{BASE}/api/v1/ensemble", json={
    "messages": [{"role": "user", "content": "Explain the CAP theorem with real-world examples"}],
    "mode": "chain",
}).json()
print(f"Model: {resp['model_used']}")
print(f"Verified: {resp['verified']}")


# ═══ 3. Ensemble Stats ═══
print("\n=== Ensemble Stats ===")
stats = requests.get(f"{BASE}/api/v1/ensemble/stats").json()
print(f"Total queries:   {stats['total_queries']}")
print(f"Single calls:    {stats['single_calls']}")
print(f"Ensemble calls:  {stats['ensemble_calls']}")
print(f"By type:         {stats['by_type']}")
print(f"Specialists:     {stats['specialist_models']}")

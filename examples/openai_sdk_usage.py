"""
Neural Brain API — OpenAI SDK Compatibility (100% Local)

Neural Brain is a drop-in replacement for OpenAI's API.
Point the OpenAI SDK at Neural Brain and it just works —
all running locally with Ollama, no API keys needed.

Prerequisites:
    pip install openai
    ollama pull qwen3:8b
    python -m uvicorn api.main:app --port 8200
"""
from openai import OpenAI

# Point OpenAI SDK at Neural Brain — no API key needed!
client = OpenAI(
    base_url="http://localhost:8200/api/v1",
    api_key="not-needed"  # 100% local, no auth required
)


# ═══ Standard Chat (routes to best local model) ═══
print("=== Chat ===")
response = client.chat.completions.create(
    model="ensemble/smart",  # Auto-classifies and routes locally
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what a load balancer does."}
    ],
    temperature=0.7,
    max_tokens=1024,
)
print(response.choices[0].message.content)


# ═══ Specific Local Model ═══
print("\n=== Specific Model ===")
response = client.chat.completions.create(
    model="ollama/qwen3:8b",
    messages=[{"role": "user", "content": "What is the Fibonacci sequence?"}],
)
print(response.choices[0].message.content[:200])


# ═══ Streaming (native Ollama streaming) ═══
print("\n=== Streaming ===")
stream = client.chat.completions.create(
    model="ensemble/smart",
    messages=[{"role": "user", "content": "Write a haiku about programming"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()


# ═══ Local Embeddings (no OpenAI key needed) ═══
print("\n=== Local Embeddings ===")
embeddings = client.embeddings.create(
    model="ollama/nomic-embed-text",  # Runs locally
    input=["Neural Brain is awesome", "LLM gateway routing"]
)
print(f"Embedding dimensions: {len(embeddings.data[0].embedding)}")

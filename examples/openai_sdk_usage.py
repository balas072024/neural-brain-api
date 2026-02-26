"""
Neural Brain API — OpenAI SDK Compatibility

Neural Brain is a drop-in replacement for OpenAI's API.
Point the OpenAI SDK at Neural Brain and it just works.

Prerequisites:
    pip install openai
    # Start Neural Brain: python -m uvicorn api.main:app --port 8200
"""
from openai import OpenAI

# Point OpenAI SDK at Neural Brain
client = OpenAI(
    base_url="http://localhost:8200/api/v1",
    api_key="not-needed"  # Neural Brain handles auth per-provider
)


# ═══ Standard Chat ═══
response = client.chat.completions.create(
    model="ensemble/smart",  # Auto-routes to best model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what a load balancer does."}
    ],
    temperature=0.7,
    max_tokens=1024,
)
print(response.choices[0].message.content)


# ═══ Streaming ═══
stream = client.chat.completions.create(
    model="ensemble/smart",
    messages=[{"role": "user", "content": "Write a haiku about programming"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()


# ═══ Embeddings ═══
# Note: requires OpenAI API key configured in Neural Brain
embeddings = client.embeddings.create(
    model="text-embedding-3-small",
    input=["Neural Brain is awesome", "LLM gateway routing"]
)
print(f"Embedding dimensions: {len(embeddings.data[0].embedding)}")

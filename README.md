# 🧠 Neural Brain API

**Unified LLM Gateway & Smart Router — 84+ models, 15 providers, one API.**

Drop-in OpenAI-compatible API that routes your requests to the best model for the job. Local-first, cost-aware, with built-in ensemble orchestration.

```
POST /api/v1/chat/completions  ←  same as OpenAI
```

Neural Brain handles the rest: provider failover, cost optimization, specialist routing, caching, and usage tracking.

---

## Why Neural Brain?

| Problem | Neural Brain Solution |
|---|---|
| Locked into one provider | 15 providers, automatic failover |
| Paying too much | Cost-aware routing, local model priority |
| Wrong model for the task | Ensemble engine classifies query → routes to specialist |
| No visibility into costs | Per-request cost tracking, per-product usage dashboards |
| Complex multi-provider setup | One config, one endpoint, all providers |

## Quick Start

```bash
# Clone
git clone https://github.com/AKBala/neural-brain-api.git
cd neural-brain-api

# Install
pip install -r requirements.txt

# Configure (add your API keys)
cp .env.example .env

# Run
python -m uvicorn api.main:app --host 0.0.0.0 --port 8200
```

That's it. Hit `http://localhost:8200/api/v1/chat/completions` with any OpenAI-compatible client.

### Zero-Config Local Mode

No API keys? No problem. Install [Ollama](https://ollama.com), pull a model, and Neural Brain auto-discovers it:

```bash
ollama pull phi4
python -m uvicorn api.main:app --port 8200
# Neural Brain auto-detects local models — no config needed
```

---

## Features

### 🔀 Smart Routing

Route requests by strategy:

```python
import requests

# Cheapest available model
requests.post("http://localhost:8200/api/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Hello"}],
    "routing_strategy": "cheapest"
})

# Best quality (ranked by benchmark performance)
requests.post("http://localhost:8200/api/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Analyze this architecture..."}],
    "routing_strategy": "best_quality"
})

# Fastest response time
requests.post("http://localhost:8200/api/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Quick question"}],
    "routing_strategy": "fastest"
})
```

**Available strategies:** `cheapest`, `fastest`, `best_quality`, `round_robin`, `fallback`, `capability`

### 🎯 Ensemble Engine

Multi-model orchestration that classifies your query and routes to the right specialist:

```python
# Ensemble auto-routes: code → code model, reasoning → reasoning model
requests.post("http://localhost:8200/api/v1/chat/completions", json={
    "model": "ensemble/smart",
    "messages": [{"role": "user", "content": "Write a Python FastAPI endpoint"}]
})
# → Routes to code specialist (Qwen Coder, Codestral, etc.)

# Consensus mode: multiple models answer, judge picks best
requests.post("http://localhost:8200/api/v1/chat/completions", json={
    "model": "ensemble/consensus",
    "messages": [{"role": "user", "content": "Should I use PostgreSQL or MongoDB?"}]
})
```

**Ensemble modes:**
| Mode | What it does | Best for |
|---|---|---|
| `smart` | Classify → route to specialist | Default, fast |
| `consensus` | 2 models answer → judge picks best | Important decisions |
| `chain` | Specialist answers → refiner improves | Quality-critical tasks |
| `fastest` | Smallest available model | Latency-sensitive |
| `strongest` | Largest available model | Complex reasoning |

**Query classification:** The engine detects code, reasoning, math, vision, translation, OCR, creative, and general queries — each routed to purpose-built models.

### 📦 Product Presets

Register your apps with pre-configured routing:

```python
# Register a product
requests.post("http://localhost:8200/api/products", json={
    "product_name": "my-chatbot",
    "default_model": "claude-sonnet-4-5-20250929",
    "routing_strategy": "fallback",
    "required_capabilities": ["chat", "function_calling"],
    "max_tokens": 2048,
    "temperature": 0.4
})

# Use it — automatically applies all presets
requests.post("http://localhost:8200/api/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Help me debug this"}],
    "product": "my-chatbot"
})
```

### 💰 Usage Tracking

Per-request cost calculation, per-model and per-product breakdowns:

```bash
# Overall usage
GET /api/usage

# Per-product usage
GET /api/usage/my-chatbot

# Cache hit rates
GET /api/cache/stats
```

### 🔌 Provider Support

| Provider | Models | Type | Key Required |
|---|---|---|---|
| **Ollama** | 20+ local models | Local | No |
| **LM Studio** | Any GGUF model | Local | No |
| **Anthropic** | Claude Opus/Sonnet/Haiku | Cloud | Yes |
| **OpenAI** | GPT-5.2, GPT-4o, o3/o4 | Cloud | Yes |
| **Google** | Gemini 3, 2.5 Pro/Flash | Cloud | Yes |
| **Groq** | Llama, DeepSeek (ultra-fast) | Cloud | Yes |
| **DeepSeek** | V3.2, R1 | Cloud | Yes |
| **Mistral** | Large 3, Codestral | Cloud | Yes |
| **xAI** | Grok 4.1, Grok 3 | Cloud | Yes |
| **MiniMax** | M2.5, M2.5 Lightning | Cloud | Yes |
| **OpenRouter** | 100+ models | Cloud | Yes |
| **Together** | Open-source models | Cloud | Yes |
| **HuggingFace** | Inference API | Cloud | Yes |
| **vLLM** | Self-hosted | Local | No |
| **Custom** | Any OpenAI-compatible | Either | Varies |

### Model Categories

Neural Brain organizes 84+ models into 6 categories:

- **General** — Chat, instruction following, general tasks
- **Code** — Code generation, debugging, review
- **Reasoning** — Complex analysis, math, logic
- **Vision** — Image understanding, OCR, diagrams
- **Audio** — Speech-to-text, text-to-speech, voice
- **Embedding** — Vector embeddings for search/RAG

---

## API Reference

### Chat Completion (OpenAI-compatible)

```
POST /api/v1/chat/completions
```

```json
{
  "messages": [{"role": "user", "content": "Hello"}],
  "model": "claude-sonnet-4-5-20250929",
  "temperature": 0.7,
  "max_tokens": 4096,
  "stream": false,
  "routing_strategy": "fallback",
  "required_capabilities": ["chat", "code"],
  "product": "my-app",
  "max_cost_per_request": 0.05
}
```

### Ensemble

```
POST /api/v1/ensemble
```

```json
{
  "messages": [{"role": "user", "content": "Explain quantum computing"}],
  "mode": "consensus",
  "verify": true
}
```

### Embeddings

```
POST /api/v1/embeddings
```

```json
{
  "texts": ["Hello world", "Neural Brain is cool"],
  "model": "text-embedding-3-small"
}
```

### Management

```
GET  /health                          # Health check
GET  /api/status                      # Full system status
GET  /api/models                      # List all models
GET  /api/models?category=code        # Filter by category
GET  /api/models?local_only=true      # Local models only
GET  /api/providers                   # List providers
POST /api/providers                   # Configure provider
POST /api/providers/ollama/discover   # Auto-discover Ollama models
GET  /api/products                    # List product presets
POST /api/products                    # Register product
GET  /api/usage                       # Usage summary
GET  /api/cache/stats                 # Cache statistics
WS   /ws                              # WebSocket chat
```

---

## Configuration

Copy `.env.example` to `.env` and add keys for providers you want:

```bash
# Local (always works, no key needed)
OLLAMA_URL=http://localhost:11434

# Add any cloud providers you want
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
GROQ_API_KEY=gsk_...
```

You only need keys for providers you want to use. Local models (Ollama, LM Studio) work without any configuration.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              Your Application                │
│         (any OpenAI-compatible client)       │
└──────────────────┬──────────────────────────┘
                   │
         POST /api/v1/chat/completions
                   │
┌──────────────────▼──────────────────────────┐
│            Neural Brain API                  │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐  │
│  │  Cache   │ │  Router  │ │   Ensemble   │  │
│  │         │ │          │ │   Engine     │  │
│  │ SHA-256 │ │ Strategy │ │ Classify →   │  │
│  │ TTL 1hr │ │ Selector │ │ Route →      │  │
│  │         │ │          │ │ (Verify)     │  │
│  └─────────┘ └──────────┘ └──────────────┘  │
│  ┌──────────────────────────────────────┐    │
│  │          Usage Tracker               │    │
│  │   Per-request cost · Per-product     │    │
│  │   Per-model · Per-provider           │    │
│  └──────────────────────────────────────┘    │
└──────────────────┬──────────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───▼───┐   ┌─────▼─────┐  ┌────▼────┐
│ Local │   │   Cloud   │  │  Cloud  │
│Ollama │   │ Anthropic │  │  OpenAI │  ... (15 providers)
│LMStudio│  │  Google   │  │  Groq   │
└───────┘   └───────────┘  └─────────┘
```

---

## Use with OpenClaw / LiteLLM / Any Client

Neural Brain is OpenAI-compatible. Point any client at it:

```python
# OpenAI Python SDK
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8200/api/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="ensemble/smart",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

```bash
# cURL
curl http://localhost:8200/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "ensemble/smart", "messages": [{"role": "user", "content": "Hello"}]}'
```

---

## Roadmap

- [ ] Persistent usage analytics (SQLite/PostgreSQL)
- [ ] Web dashboard for monitoring
- [ ] Plugin system for custom routing logic
- [ ] Rate limiting per API key
- [ ] Prompt template registry
- [ ] A/B testing across models
- [ ] Webhook notifications on errors/budget alerts

---

## License

MIT — use it however you want.

---

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

Built by [Balamurugan](https://github.com/AKBala) — part of the Kaashmikhaa AI ecosystem.

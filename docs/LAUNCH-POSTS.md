# Neural Brain API — Launch Posts
## Ready to copy-paste. One per platform.

---

## 1. HACKER NEWS (Show HN)

**Title:** Show HN: Neural Brain API – Open-source LLM gateway with smart routing across 84+ models

**Post body:**

Hey HN,

I built an open-source unified LLM gateway that sits between your app and 15+ providers (Ollama, Anthropic, OpenAI, Google, Groq, DeepSeek, Mistral, xAI, etc).

One endpoint. OpenAI-compatible. Drop-in replacement.

What makes it different from LiteLLM or OpenRouter:

- Ensemble Engine — classifies your query (code/reasoning/math/vision/translation) and routes to the right specialist model automatically. Ask a code question, gets routed to Qwen Coder. Ask a reasoning question, gets routed to DeepSeek R1. No config needed.

- Consensus mode — two models answer independently, a third judges and picks the best response. Useful for high-stakes decisions.

- Cost-aware routing — set max_cost_per_request: 0.01 and it picks the cheapest model that can handle the job. Or use routing_strategy: "fastest" or "best_quality".

- Product presets — register your apps with default models, temperature, routing strategy. Your chatbot gets one config, your code assistant gets another.

- Local-first — auto-discovers Ollama models. Works fully offline with zero API keys.

Built this while working as an Application Support Manager — started as an internal tool to route AI requests across our different products, then realized it might be useful to others.

Stack: Python, FastAPI, aiohttp. 6 dependencies total.

GitHub: https://github.com/balas072024/neural-brain-api

Happy to answer questions about the routing logic or ensemble architecture.

---

## 2. REDDIT — r/LocalLLaMA

**Title:** I built an open-source LLM gateway that auto-routes queries to the right local model — code goes to Qwen Coder, reasoning to DeepSeek R1, vision to LLaVA. Works with Ollama out of the box.

**Post body:**

Been running multiple Ollama models on my RTX 4060 and got tired of manually switching between them depending on the task. So I built Neural Brain API — a unified gateway that classifies your query and routes it to the right specialist.

How it works:

You hit one endpoint: POST /api/v1/chat/completions (OpenAI-compatible)

The ensemble engine figures out what you're asking:
- Code question -> routes to Qwen Coder / Codestral / DeepSeek Coder
- Reasoning -> DeepSeek R1 / Phi4 Reasoning
- Math -> DeepSeek R1
- Vision -> Mistral Small 3.1 / Qwen3-VL / LLaVA
- Translation -> TranslateGemma
- OCR -> GLM-OCR
- General -> Phi4 / Gemma 3 / Qwen 2.5

It also supports:
- Consensus mode (2 models answer, third judges)
- Chain mode (specialist answers, refiner improves)
- Fallback chains (if primary model fails, tries next)
- Auto-discovery of your Ollama models
- Cloud providers as fallback (Anthropic, OpenAI, Google, Groq, etc)
- Usage tracking with per-request cost calculation

No API keys needed for local-only setup. Just have Ollama running.

GitHub: https://github.com/balas072024/neural-brain-api

Feedback welcome — especially on the query classification logic. It's keyword + pattern based right now, would love to make it smarter.

---

## 3. REDDIT — r/selfhosted

**Title:** Neural Brain API — self-hosted LLM gateway that routes requests across Ollama, Anthropic, OpenAI, Google and 12 more providers. One endpoint, automatic failover, cost tracking.

**Post body:**

Just open-sourced my LLM gateway. It's a single Python service that gives you one unified API for all your AI models — local and cloud.

The problem it solves:

If you self-host Ollama AND use cloud APIs, you end up with different endpoints, different auth, different request formats, and no unified cost tracking. Neural Brain sits in front of everything and gives you one OpenAI-compatible endpoint.

What it does:

- Routes to 84+ models across 15 providers
- Auto-discovers your Ollama models (no config needed)
- Automatic failover — if Ollama is slow, falls back to cloud
- Smart routing: cheapest, fastest, best quality, round-robin, capability-based
- Ensemble engine that classifies queries and picks the right specialist model
- Caching (SHA-256 hashed, 1hr TTL) — identical requests served from cache
- Per-request cost tracking with product-level breakdowns
- WebSocket support for streaming

Deployment:

pip install -r requirements.txt
cp .env.example .env
python -m uvicorn api.main:app --port 8200

Docker Compose included (Neural Brain + Ollama in one stack).

Stack: Python, FastAPI, aiohttp. Minimal dependencies.

GitHub: https://github.com/balas072024/neural-brain-api

---

## 4. REDDIT — r/MachineLearning [D]

**Title:** [D] Open-sourced an LLM ensemble gateway that classifies queries and routes to specialist models — looking for feedback on the classification approach

**Post body:**

I've been working on a multi-model orchestration system and wanted to get feedback from this community on the approach.

The core idea: Instead of sending every request to one model, classify the query type (code, reasoning, math, vision, translation, creative, general) and route to a specialist model that's best suited for that task.

Current classification approach:
- Keyword matching with weighted scores per category
- Regex pattern matching for structural indicators (code blocks, math expressions, translation requests)
- Category weights (e.g., OCR keywords weighted 2.5x since they're very specific, creative keywords weighted 1.0x since they overlap with general)
- Confidence threshold — low confidence triggers verification step

Ensemble modes:
1. Smart — classify then route to top specialist
2. Consensus — 2 specialists answer, judge model picks best
3. Chain — specialist answers, refiner model improves output

What I'd love feedback on:
- Is keyword/pattern classification sufficient or should I use a small classifier model?
- Better heuristics for distinguishing reasoning vs general queries?
- Is consensus mode worth the latency trade-off?

The whole system is open source: https://github.com/balas072024/neural-brain-api

Ensemble engine code is in core/ensemble.py if you want to look at the classification logic directly.

---

## 5. TWITTER / X

Post 1 (Launch):

Just open-sourced Neural Brain API

An LLM gateway that routes your requests to the right model automatically.

Code question? Routes to Qwen Coder
Reasoning? Routes to DeepSeek R1
Vision? Routes to LLaVA

84+ models. 15 providers. One endpoint.
OpenAI-compatible. Local-first.

https://github.com/balas072024/neural-brain-api

Post 2 (Thread reply):

What makes it different:

1/ Ensemble Engine — classifies your query and picks the specialist. No manual model switching.

2/ Consensus mode — 2 models answer, a 3rd judges. For when you need to be right.

3/ Cost-aware routing — set a budget per request and it finds the cheapest model that works.

4/ Product presets — your chatbot, code tool, and support bot each get their own config.

5/ Auto-discovers Ollama models. Zero config for local setups.

6/ 6 dependencies. FastAPI + aiohttp. That's basically it.

Built this from a real production need — was managing AI routing across multiple products and got tired of the spaghetti.

MIT licensed. PRs welcome.

---

## 6. LINKEDIN

I just open-sourced Neural Brain API — a unified LLM gateway I've been building.

The problem: if you use multiple AI models (local Ollama models + cloud APIs like Anthropic, OpenAI, Google), you end up with different endpoints, different authentication, different request formats, and zero visibility into what anything costs.

Neural Brain sits in front of all of them and gives you one endpoint. OpenAI-compatible, so any existing client works without changes.

The interesting part is the Ensemble Engine — it classifies incoming queries (code, reasoning, math, vision, translation) and routes them to specialist models automatically. You can also run consensus mode where two models answer independently and a third picks the best response.

Some numbers:
- 84+ models supported
- 15 providers (Ollama, Anthropic, OpenAI, Google, Groq, DeepSeek, Mistral, xAI, and more)
- 6 routing strategies (cheapest, fastest, best quality, round-robin, fallback, capability-based)
- Per-request cost tracking with product-level breakdowns
- 6 Python dependencies total

This came from a real production need — managing AI routing across multiple applications as an Application Support Manager.

MIT licensed. Available on GitHub: https://github.com/balas072024/neural-brain-api

If you're building with LLMs and tired of managing multiple provider integrations, give it a look.

#OpenSource #AI #LLM #Python #MachineLearning #DevTools

---

## 7. DEV.TO

**Title:** I Built an Open-Source LLM Gateway That Routes Queries to the Right Model Automatically
**Tags:** ai, python, opensource, llm

The Problem:

I run multiple AI models — local ones via Ollama and cloud APIs from Anthropic, OpenAI, and Google. Every time I build something, I deal with different endpoints, different auth, different formats, no cost visibility, and manually picking which model to use.

So I built Neural Brain API — a unified gateway that handles all of this.

One endpoint. OpenAI-compatible. The ensemble engine figures out what type of query it is and routes to a specialist. Code goes to Qwen Coder. Reasoning goes to DeepSeek R1. Vision to LLaVA.

Features:
- 84+ models across 15 providers
- 6 routing strategies
- 8 query types classified by the ensemble engine
- Consensus mode (2 models answer, third judges)
- Chain mode (specialist answers, refiner improves)
- Built-in caching, rate limiting, usage tracking
- Auto-discovers Ollama models (zero config)
- Docker Compose included
- 6 Python dependencies
- MIT licensed

GitHub: https://github.com/balas072024/neural-brain-api

Feedback welcome — especially on the query classification approach.

---

## POSTING SCHEDULE

| Day | Platform | Post # |
|-----|----------|--------|
| Day 1 (Today) | Reddit r/LocalLLaMA | #2 |
| Day 1 (Today) | Twitter/X | #5 |
| Day 2 | Reddit r/selfhosted | #3 |
| Day 2 | LinkedIn | #6 |
| Day 3 | Hacker News (Show HN) | #1 |
| Day 3 | Dev.to | #7 |
| Day 5 | Reddit r/MachineLearning | #4 |

TIPS:
- Post Reddit between 9-11am EST
- HN between 8-10am EST weekdays
- Reply to EVERY comment in first 2 hours
- Good questions from comments = add answer to README

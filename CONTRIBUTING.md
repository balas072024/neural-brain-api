# Contributing to Neural Brain API

Thanks for your interest in contributing!

## Quick Start

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Test locally: `python -m uvicorn api.main:app --port 8200`
5. Submit a PR

## What We Need

- **New provider adapters** — Add support for more LLM providers
- **Better routing strategies** — Smarter model selection algorithms
- **Ensemble improvements** — Better query classification, new orchestration modes
- **Documentation** — Usage examples, tutorials, guides
- **Tests** — Unit tests, integration tests
- **Dashboard** — Web UI for monitoring usage and costs

## Code Style

- Python 3.10+
- Type hints on all functions
- Docstrings on public methods
- Keep dependencies minimal

## Adding a New Provider

1. Add `ProviderType` enum value in `core/brain.py`
2. Add model configs to `DEFAULT_MODELS`
3. Add provider URL to `_auto_configure()`
4. If non-OpenAI-compatible, add `_call_yourprovider()` method
5. Update `.env.example` with the new API key variable
6. Update README provider table

## Adding a New Ensemble Mode

1. Add mode handler in `core/ensemble.py` → `complete()` method
2. Add to `ENSEMBLE_MODEL_IDS` in `api/main.py`
3. Document in README

## Questions?

Open an issue. We're friendly.

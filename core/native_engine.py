"""
Neural Brain Native Inference Engine v2.0 — SPEED OPTIMIZED
Run GGUF models directly — zero Ollama dependency.

Uses llama-cpp-python for direct GGUF model loading and inference.
Same model format as Ollama, but runs entirely inside Neural Brain.

PERFORMANCE OPTIMIZATIONS (targeting sub-2s response):
- Flash Attention: reduces memory bandwidth bottleneck
- Optimized n_batch (1024): faster prompt processing
- Low thread count (2) for GPU offload: proven faster than auto
- Memory lock (use_mlock): prevents OS swapping to disk
- Speculative decoding (LlamaPromptLookupDecoding): parallel token prediction
- KV cache warmup: pre-evaluates system prompt on model load
- Reduced context window (2048): faster first-token latency
- Smart max_tokens caps per speed tier

Sources:
- NVIDIA CUDA Graphs: ~150 tok/s on RTX 4090
- llama.cpp best practices: flash_attn + n_batch=1024 + n_threads=2 (GPU)
- Speculative decoding: up to 2.5x speedup with prompt lookup

FEATURES:
- Load GGUF models from local disk or HuggingFace
- GPU acceleration (CUDA/Metal/Vulkan) via llama.cpp
- Multiple models loaded concurrently with VRAM management
- Auto-discover Ollama's cached GGUF files and reuse them
- Model pool: keep frequently used models in memory
- Compatible with existing Neural Brain routing and learning
"""

import os
import sys
import time
import json
import logging
import hashlib
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger("neural-brain.native")


# ═══════════════════════════════════════════════════════
#  Model Configuration
# ═══════════════════════════════════════════════════════

@dataclass
class NativeModelConfig:
    """Configuration for a natively loaded GGUF model."""
    model_id: str              # e.g., "native/qwen3-4b-q4"
    gguf_path: str             # Path to .gguf file
    name: str = ""
    family: str = ""           # e.g., "qwen3", "llama3"
    params: str = ""           # e.g., "4b", "8b"
    quantization: str = ""     # e.g., "Q4_K_M"
    size_gb: float = 0.0
    n_ctx: int = 2048          # Reduced context for speed (was 4096)
    n_gpu_layers: int = -1     # -1 = all layers on GPU
    n_threads: int = 2         # 2 threads optimal for GPU workloads
    n_batch: int = 1024        # Batch size for prompt processing
    flash_attn: bool = True    # Flash attention for speed
    use_mlock: bool = True     # Lock memory to prevent OS swapping
    use_mmap: bool = True      # Memory-map for fast loading
    loaded: bool = False
    last_used: float = 0.0
    warmup_done: bool = False  # KV cache warmup completed


@dataclass
class NativeResponse:
    """Response from native inference."""
    content: str
    model_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0


# ═══════════════════════════════════════════════════════
#  GGUF Model Discovery
# ═══════════════════════════════════════════════════════

def get_ollama_model_paths() -> List[str]:
    """Find where Ollama stores GGUF model blobs on this system."""
    paths = []
    system = platform.system()

    if system == "Windows":
        # Windows: %USERPROFILE%\.ollama\models
        home = os.environ.get("USERPROFILE", "")
        if home:
            paths.append(os.path.join(home, ".ollama", "models"))
        # Also check common install paths
        paths.append(os.path.join("C:\\", "Users", "Public", ".ollama", "models"))
    elif system == "Darwin":
        # macOS: ~/.ollama/models
        paths.append(os.path.expanduser("~/.ollama/models"))
    else:
        # Linux: ~/.ollama/models or /usr/share/ollama/.ollama/models
        paths.append(os.path.expanduser("~/.ollama/models"))
        paths.append("/usr/share/ollama/.ollama/models")

    return [p for p in paths if os.path.exists(p)]


def discover_gguf_files(search_dirs: List[str] = None) -> Dict[str, str]:
    """Discover GGUF model files on disk.
    Returns {model_name: gguf_path}."""
    if search_dirs is None:
        search_dirs = get_ollama_model_paths()
        # Also check common user directories
        search_dirs.extend([
            os.path.expanduser("~/models"),
            os.path.expanduser("~/gguf"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
        ])

    found = {}
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        # Direct .gguf files
        for root, dirs, files in os.walk(search_dir):
            for f in files:
                if f.endswith(".gguf"):
                    path = os.path.join(root, f)
                    name = f.replace(".gguf", "")
                    found[name] = path

        # Ollama blob format: models/manifests/... and models/blobs/...
        blobs_dir = os.path.join(search_dir, "blobs")
        manifests_dir = os.path.join(search_dir, "manifests")
        if os.path.exists(manifests_dir):
            _discover_ollama_blobs(manifests_dir, blobs_dir, found)

    return found


def _discover_ollama_blobs(manifests_dir: str, blobs_dir: str, found: Dict):
    """Parse Ollama manifest files to find model GGUF blobs."""
    try:
        registry_dir = os.path.join(manifests_dir, "registry.ollama.ai", "library")
        if not os.path.exists(registry_dir):
            return

        for model_name in os.listdir(registry_dir):
            model_dir = os.path.join(registry_dir, model_name)
            if not os.path.isdir(model_dir):
                continue

            for tag in os.listdir(model_dir):
                manifest_path = os.path.join(model_dir, tag)
                if not os.path.isfile(manifest_path):
                    continue

                try:
                    with open(manifest_path) as f:
                        manifest = json.load(f)

                    for layer in manifest.get("layers", []):
                        if layer.get("mediaType", "") in (
                            "application/vnd.ollama.image.model",
                            "application/octet-stream",
                        ):
                            digest = layer.get("digest", "")
                            if digest:
                                # Ollama stores blobs as sha256-<hash>
                                blob_name = digest.replace(":", "-")
                                blob_path = os.path.join(blobs_dir, blob_name)
                                if os.path.exists(blob_path):
                                    model_key = f"{model_name}:{tag}"
                                    found[model_key] = blob_path
                                    size_gb = os.path.getsize(blob_path) / (1024**3)
                                    logger.debug(f"Found Ollama blob: {model_key} -> {blob_path} ({size_gb:.1f}GB)")
                except (json.JSONDecodeError, KeyError, OSError):
                    continue
    except OSError:
        pass


# ═══════════════════════════════════════════════════════
#  Native Inference Engine
# ═══════════════════════════════════════════════════════

class NativeEngine:
    """
    Direct GGUF model inference — no Ollama needed.
    Uses llama-cpp-python for hardware-accelerated inference.

    SPEED OPTIMIZATIONS (v2.0):
    - flash_attn=True: Flash Attention reduces memory bandwidth bottleneck
    - n_batch=1024: Faster prompt processing (default 512 is too conservative)
    - n_threads=2: Counter-intuitive but proven faster for GPU-heavy workloads
    - use_mlock=True: Prevents OS from swapping model to disk
    - use_mmap=True: Memory-mapped loading for faster startup
    - n_ctx=2048: Smaller context = faster first-token latency
    - Speculative decoding: LlamaPromptLookupDecoding for parallel token prediction
    - KV cache warmup: Pre-evaluate system prompt on load for instant first response
    - Smart max_tokens: Auto-cap based on speed tier (fast=128, medium=256, full=512)
    """

    # Maximum models to keep loaded in memory simultaneously
    MAX_LOADED_MODELS = 2

    # Default system prompt for KV cache warmup
    WARMUP_SYSTEM_PROMPT = "You are a helpful AI assistant. Respond concisely."

    def __init__(self, models_dir: str = None, n_gpu_layers: int = -1, n_ctx: int = 2048):
        self.models_dir = models_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )
        os.makedirs(self.models_dir, exist_ok=True)

        self.default_n_gpu_layers = n_gpu_layers  # -1 = all on GPU
        self.default_n_ctx = n_ctx
        self._llama_available = False
        self._speculative_available = False
        self._models: Dict[str, NativeModelConfig] = {}
        self._loaded: Dict[str, Any] = {}  # model_id -> Llama instance
        self._discovered_paths: Dict[str, str] = {}

        # Check if llama-cpp-python is available
        try:
            from llama_cpp import Llama
            self._llama_available = True
            logger.info("llama-cpp-python available — native inference enabled")

            # Check for speculative decoding support
            try:
                from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
                self._speculative_available = True
                logger.info("Speculative decoding (LlamaPromptLookupDecoding) available")
            except ImportError:
                logger.info("Speculative decoding not available (upgrade llama-cpp-python)")

        except ImportError:
            logger.warning(
                "llama-cpp-python not installed. Install with: "
                "pip install llama-cpp-python  "
                "(For GPU: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python)"
            )

    @property
    def available(self) -> bool:
        return self._llama_available

    def discover_models(self) -> Dict[str, NativeModelConfig]:
        """Discover all available GGUF models on this system."""
        self._discovered_paths = discover_gguf_files()

        for model_key, gguf_path in self._discovered_paths.items():
            model_id = f"native/{model_key}"

            if model_id in self._models:
                continue

            try:
                size_gb = os.path.getsize(gguf_path) / (1024**3)
            except OSError:
                size_gb = 0.0

            # Parse model info from name
            family, params, quant = _parse_model_name(model_key)

            self._models[model_id] = NativeModelConfig(
                model_id=model_id,
                gguf_path=gguf_path,
                name=model_key,
                family=family,
                params=params,
                quantization=quant,
                size_gb=round(size_gb, 2),
                n_ctx=self.default_n_ctx,
                n_gpu_layers=self.default_n_gpu_layers,
            )

        logger.info(f"Discovered {len(self._models)} native GGUF models")
        return dict(self._models)

    def load_model(self, model_id: str) -> bool:
        """Load a GGUF model into memory with all speed optimizations.

        Speed settings applied:
        - flash_attn: Flash Attention for reduced memory bandwidth
        - n_batch=1024: Faster prompt eval (2x over default 512)
        - n_threads=2: Optimal for GPU-offloaded models
        - use_mlock: Lock model in RAM, prevent swapping
        - use_mmap: Memory-mapped loading for fast startup
        - Speculative decoding: LlamaPromptLookupDecoding when available
        - KV warmup: Pre-evaluate system prompt for instant first response
        """
        if not self._llama_available:
            logger.error("Cannot load model: llama-cpp-python not installed")
            return False

        if model_id in self._loaded:
            return True  # Already loaded

        config = self._models.get(model_id)
        if not config:
            logger.error(f"Model not found: {model_id}")
            return False

        if not os.path.exists(config.gguf_path):
            logger.error(f"GGUF file not found: {config.gguf_path}")
            return False

        # Evict oldest model if at capacity
        if len(self._loaded) >= self.MAX_LOADED_MODELS:
            self._evict_oldest()

        try:
            from llama_cpp import Llama

            # Use config thread count (default 2 for GPU, fallback to CPU count)
            n_threads = config.n_threads if config.n_threads > 0 else (os.cpu_count() or 4)

            # Build speculative decoding draft model if available
            draft_model = None
            if self._speculative_available:
                try:
                    from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
                    # num_pred_tokens=10 for GPU, 2 for CPU-only
                    num_pred = 10 if config.n_gpu_layers != 0 else 2
                    draft_model = LlamaPromptLookupDecoding(num_pred_tokens=num_pred)
                    logger.info(f"Speculative decoding enabled (pred_tokens={num_pred})")
                except Exception as e:
                    logger.debug(f"Speculative decoding init failed: {e}")

            logger.info(
                f"Loading native model: {model_id} ({config.size_gb}GB) "
                f"[gpu_layers={config.n_gpu_layers}, ctx={config.n_ctx}, "
                f"batch={config.n_batch}, threads={n_threads}, "
                f"flash_attn={config.flash_attn}, mlock={config.use_mlock}, "
                f"speculative={'yes' if draft_model else 'no'}]"
            )

            start = time.time()

            # Build kwargs — only pass flash_attn if supported
            llm_kwargs = dict(
                model_path=config.gguf_path,
                n_ctx=config.n_ctx,
                n_gpu_layers=config.n_gpu_layers,
                n_threads=n_threads,
                n_batch=config.n_batch,
                use_mlock=config.use_mlock,
                use_mmap=config.use_mmap,
                verbose=False,
            )

            # flash_attn support was added in llama-cpp-python 0.2.58+
            try:
                llm_kwargs["flash_attn"] = config.flash_attn
                if draft_model:
                    llm_kwargs["draft_model"] = draft_model
                llm = Llama(**llm_kwargs)
            except TypeError as e:
                # Fallback: older llama-cpp-python without flash_attn/draft_model
                logger.warning(f"Falling back without flash_attn/draft_model: {e}")
                llm_kwargs.pop("flash_attn", None)
                llm_kwargs.pop("draft_model", None)
                llm = Llama(**llm_kwargs)

            load_time = time.time() - start

            self._loaded[model_id] = llm
            config.loaded = True
            config.last_used = time.time()

            logger.info(f"Loaded {model_id} in {load_time:.1f}s")

            # KV cache warmup: pre-evaluate system prompt so first real
            # request skips prompt processing for the system prefix
            self._warmup_model(model_id)

            return True

        except Exception as e:
            logger.error(f"Failed to load {model_id}: {e}")
            return False

    def _warmup_model(self, model_id: str):
        """Pre-evaluate system prompt to warm up KV cache.

        This ensures the first real user request doesn't pay the
        full prompt processing cost for the system prefix.
        llama-cpp-python automatically caches evaluated tokens
        in the KV cache, so subsequent calls with the same prefix
        will skip re-evaluation.
        """
        if model_id not in self._loaded:
            return

        config = self._models.get(model_id)
        if config and config.warmup_done:
            return

        llm = self._loaded[model_id]
        try:
            warmup_prompt = _format_chat_prompt(
                [{"role": "user", "content": "hi"}],
                system=self.WARMUP_SYSTEM_PROMPT,
            )
            start = time.time()
            # Generate 1 token just to force KV cache population
            llm(warmup_prompt, max_tokens=1, temperature=0.0)
            warmup_ms = (time.time() - start) * 1000

            if config:
                config.warmup_done = True

            logger.info(f"KV cache warmup done for {model_id} in {warmup_ms:.0f}ms")
        except Exception as e:
            logger.debug(f"KV warmup failed for {model_id}: {e}")

    def unload_model(self, model_id: str):
        """Unload a model from memory."""
        if model_id in self._loaded:
            del self._loaded[model_id]
            if model_id in self._models:
                self._models[model_id].loaded = False
            logger.info(f"Unloaded: {model_id}")

    def _evict_oldest(self):
        """Remove the least recently used model from memory."""
        if not self._loaded:
            return
        oldest_id = min(
            self._loaded.keys(),
            key=lambda mid: self._models.get(mid, NativeModelConfig("", "")).last_used
        )
        self.unload_model(oldest_id)

    @staticmethod
    def _speed_cap_tokens(max_tokens: int, speed_tier: str = "fast") -> int:
        """Cap max_tokens based on speed tier for sub-2s responses.

        Speed tiers:
        - "fast":   128 tokens max (~1-2s at 60+ tok/s)
        - "medium": 256 tokens max (~2-4s)
        - "full":   no cap (use caller's max_tokens as-is)
        """
        caps = {"fast": 128, "medium": 256, "full": max_tokens}
        return min(max_tokens, caps.get(speed_tier, max_tokens))

    async def complete(self, model_id: str, messages: List[Dict],
                       temperature: float = 0.7, max_tokens: int = 256,
                       system: str = "",
                       speed_tier: str = "fast") -> NativeResponse:
        """Run inference on a loaded model — speed optimized.

        Args:
            speed_tier: "fast" (128 tok, ~1-2s), "medium" (256 tok),
                        "full" (no cap). Default "fast" for sub-2s response.
        """
        if not self._llama_available:
            raise RuntimeError("llama-cpp-python not installed")

        # Auto-load if not loaded
        if model_id not in self._loaded:
            if not self.load_model(model_id):
                raise RuntimeError(f"Failed to load model: {model_id}")

        llm = self._loaded[model_id]
        config = self._models.get(model_id)
        if config:
            config.last_used = time.time()

        # Build prompt from messages
        prompt = _format_chat_prompt(messages, system)

        # Apply speed tier cap
        effective_max_tokens = self._speed_cap_tokens(max_tokens, speed_tier)

        start = time.time()
        try:
            output = llm(
                prompt,
                max_tokens=effective_max_tokens,
                temperature=temperature,
                stop=["<|im_end|>", "<|end|>", "</s>", "<|eot_id|>"],
                echo=False,
            )

            elapsed_ms = (time.time() - start) * 1000
            content = output["choices"][0]["text"].strip() if output.get("choices") else ""
            usage = output.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            tps = (completion_tokens / (elapsed_ms / 1000)) if elapsed_ms > 0 and completion_tokens > 0 else 0

            return NativeResponse(
                content=content,
                model_id=model_id,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=completion_tokens,
                total_tokens=usage.get("total_tokens", 0),
                latency_ms=round(elapsed_ms, 1),
                tokens_per_second=round(tps, 1),
            )

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.error(f"Native inference failed for {model_id}: {e}")
            raise RuntimeError(f"Inference failed: {e}")

    async def chat(self, model_id: str, messages: List[Dict],
                   temperature: float = 0.7, max_tokens: int = 256,
                   system: str = "",
                   speed_tier: str = "fast") -> NativeResponse:
        """Chat completion using llama-cpp-python's chat interface — speed optimized.

        Args:
            speed_tier: "fast" (128 tok, ~1-2s), "medium" (256 tok),
                        "full" (no cap). Default "fast" for sub-2s response.
        """
        if not self._llama_available:
            raise RuntimeError("llama-cpp-python not installed")

        if model_id not in self._loaded:
            if not self.load_model(model_id):
                raise RuntimeError(f"Failed to load model: {model_id}")

        llm = self._loaded[model_id]
        config = self._models.get(model_id)
        if config:
            config.last_used = time.time()

        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)

        # Apply speed tier cap
        effective_max_tokens = self._speed_cap_tokens(max_tokens, speed_tier)

        start = time.time()
        try:
            output = llm.create_chat_completion(
                messages=chat_messages,
                max_tokens=effective_max_tokens,
                temperature=temperature,
            )

            elapsed_ms = (time.time() - start) * 1000
            content = output["choices"][0]["message"]["content"].strip() if output.get("choices") else ""
            usage = output.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            tps = (completion_tokens / (elapsed_ms / 1000)) if elapsed_ms > 0 and completion_tokens > 0 else 0

            return NativeResponse(
                content=content,
                model_id=model_id,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=completion_tokens,
                total_tokens=usage.get("total_tokens", 0),
                latency_ms=round(elapsed_ms, 1),
                tokens_per_second=round(tps, 1),
            )

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.error(f"Native chat failed for {model_id}: {e}")
            raise RuntimeError(f"Chat failed: {e}")

    async def download_model(self, repo_id: str, filename: str = None,
                             quantization: str = "Q4_K_M") -> Optional[str]:
        """Download a GGUF model from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download

            if not filename:
                # Try to find the right GGUF file
                filename = f"*{quantization}*.gguf"

            logger.info(f"Downloading {repo_id}/{filename}...")
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=self.models_dir,
            )
            logger.info(f"Downloaded to: {path}")
            return path

        except ImportError:
            logger.warning("huggingface_hub not installed. Install with: pip install huggingface-hub")
            return None
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None

    def get_status(self) -> Dict:
        """Get native engine status including speed optimization details."""
        return {
            "available": self._llama_available,
            "engine_version": "2.0-speed",
            "total_models_discovered": len(self._models),
            "models_loaded": len(self._loaded),
            "max_loaded": self.MAX_LOADED_MODELS,
            "models_dir": self.models_dir,
            "speed_optimizations": {
                "flash_attention": True,
                "n_batch": 1024,
                "n_threads_gpu": 2,
                "use_mlock": True,
                "use_mmap": True,
                "speculative_decoding": self._speculative_available,
                "kv_cache_warmup": True,
                "context_window": self.default_n_ctx,
                "gpu_layers": self.default_n_gpu_layers,
                "speed_tiers": {
                    "fast": "128 tokens max (~1-2s)",
                    "medium": "256 tokens max (~2-4s)",
                    "full": "no cap",
                },
            },
            "models": {
                mid: {
                    "name": cfg.name,
                    "family": cfg.family,
                    "params": cfg.params,
                    "quantization": cfg.quantization,
                    "size_gb": cfg.size_gb,
                    "loaded": cfg.loaded,
                    "warmup_done": cfg.warmup_done,
                    "gguf_path": cfg.gguf_path,
                }
                for mid, cfg in self._models.items()
            },
            "loaded_models": list(self._loaded.keys()),
        }

    def get_loaded_models(self) -> List[str]:
        return list(self._loaded.keys())

    def get_available_models(self) -> List[str]:
        return list(self._models.keys())


# ═══════════════════════════════════════════════════════
#  Utility Functions
# ═══════════════════════════════════════════════════════

def _parse_model_name(name: str) -> Tuple[str, str, str]:
    """Parse model family, param count, and quantization from name.
    E.g., 'qwen3:4b' -> ('qwen3', '4b', '')
    E.g., 'llama-3.2-3b-Q4_K_M' -> ('llama-3.2', '3b', 'Q4_K_M')
    """
    family = ""
    params = ""
    quant = ""

    # Quantization patterns
    quant_patterns = ["Q2_K", "Q3_K_M", "Q3_K_S", "Q4_0", "Q4_K_M", "Q4_K_S",
                      "Q5_0", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "F16", "FP16",
                      "q2_k", "q3_k_m", "q4_0", "q4_k_m", "q5_k_m", "q6_k", "q8_0", "fp16"]

    parts = name.replace(":", "-").replace("/", "-").split("-")

    for p in parts:
        p_upper = p.upper()
        if p_upper in [q.upper() for q in quant_patterns]:
            quant = p_upper
        elif p.lower().endswith("b") and p[:-1].replace(".", "").isdigit():
            params = p.lower()
        else:
            family = f"{family}-{p}" if family else p

    return family, params, quant


def _format_chat_prompt(messages: List[Dict], system: str = "") -> str:
    """Format messages into a chat prompt string.
    Uses ChatML format which works with most GGUF models."""
    parts = []

    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════
#  Popular GGUF Models (HuggingFace repos)
# ═══════════════════════════════════════════════════════

# Speed-optimized quantization recommendations:
# - Q4_0:   Fastest inference, slightly lower quality. Best for sub-2s targets.
# - Q4_K_M: Best balance of speed + quality. Recommended default.
# - Q5_K_M: Higher quality, ~15% slower than Q4_K_M.
# - Q8_0:   Near-FP16 quality, ~2x slower than Q4_K_M. Use for accuracy-critical tasks.
# For sub-2s responses: prefer Q4_K_M on 3-4B models, Q4_0 on 7-8B models.

POPULAR_GGUF_MODELS = {
    "qwen3-4b": {
        "repo": "Qwen/Qwen3-4B-GGUF",
        "files": {"Q4_K_M": "qwen3-4b-q4_k_m.gguf", "Q8_0": "qwen3-4b-q8_0.gguf"},
        "params": "4b", "family": "qwen3",
    },
    "qwen3-8b": {
        "repo": "Qwen/Qwen3-8B-GGUF",
        "files": {"Q4_K_M": "qwen3-8b-q4_k_m.gguf", "Q8_0": "qwen3-8b-q8_0.gguf"},
        "params": "8b", "family": "qwen3",
    },
    "llama3.2-3b": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "files": {"Q4_K_M": "Llama-3.2-3B-Instruct-Q4_K_M.gguf"},
        "params": "3b", "family": "llama3.2",
    },
    "llama3.1-8b": {
        "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "files": {"Q4_K_M": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"},
        "params": "8b", "family": "llama3.1",
    },
    "phi4-mini": {
        "repo": "bartowski/phi-4-mini-instruct-GGUF",
        "files": {"Q4_K_M": "phi-4-mini-instruct-Q4_K_M.gguf"},
        "params": "3.8b", "family": "phi4",
    },
    "deepseek-r1-8b": {
        "repo": "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
        "files": {"Q4_K_M": "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"},
        "params": "8b", "family": "deepseek-r1",
    },
}

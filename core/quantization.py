"""
Neural Brain Model Quantization & Compression Manager v1.0
Optimize model storage and VRAM usage through intelligent quantization.

QUANTIZATION STRATEGY:
- Auto-detect available quantization levels from Ollama model info
- Prefer smaller quantized variants when quality threshold is met
- Track model sizes and recommend space-saving alternatives
- Pull quantized variants automatically (e.g., Q4_K_M instead of FP16)
- VRAM-aware: pick the best quantization level for your hardware

COMPRESSION LEVELS (Ollama GGUF quantizations):
- Q2_K:    ~2-bit, ~40% size, quality loss notable    → Ultra-low VRAM
- Q3_K_M:  ~3-bit, ~50% size, acceptable for chat     → Low VRAM
- Q4_0:    ~4-bit, ~55% size, good quality             → Standard
- Q4_K_M:  ~4-bit, ~60% size, best 4-bit quality      → Recommended default
- Q5_K_M:  ~5-bit, ~65% size, near-original quality    → High quality
- Q6_K:    ~6-bit, ~75% size, minimal quality loss     → Premium
- Q8_0:    ~8-bit, ~85% size, essentially lossless     → Maximum quality
- FP16:    ~100% size, original quality                → Full precision
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("neural-brain.quantization")


# ═══════════════════════════════════════════════════════
#  Quantization Levels & Model Size Estimates
# ═══════════════════════════════════════════════════════

@dataclass
class QuantizationInfo:
    """Information about a specific quantization variant."""
    level: str           # e.g., "q4_K_M", "q8_0", "fp16"
    bits_per_weight: float
    quality_retention: float  # 0-1, how much quality vs FP16
    size_ratio: float        # Size relative to FP16
    recommended_for: str     # Description of use case

QUANTIZATION_LEVELS = {
    "q2_K":   QuantizationInfo("q2_K", 2.0, 0.65, 0.25, "Ultra-low VRAM, basic chat only"),
    "q3_K_M": QuantizationInfo("q3_K_M", 3.0, 0.78, 0.38, "Low VRAM, acceptable quality"),
    "q4_0":   QuantizationInfo("q4_0", 4.0, 0.85, 0.50, "Standard quantization"),
    "q4_K_M": QuantizationInfo("q4_K_M", 4.5, 0.90, 0.55, "Best 4-bit quality (recommended)"),
    "q5_0":   QuantizationInfo("q5_0", 5.0, 0.92, 0.62, "Good balance"),
    "q5_K_M": QuantizationInfo("q5_K_M", 5.5, 0.95, 0.67, "Near-original quality"),
    "q6_K":   QuantizationInfo("q6_K", 6.0, 0.97, 0.75, "Minimal quality loss"),
    "q8_0":   QuantizationInfo("q8_0", 8.0, 0.99, 0.85, "Essentially lossless"),
    "fp16":   QuantizationInfo("fp16", 16.0, 1.0, 1.0, "Full precision (original)"),
}

# Priority order: best quality-to-size ratio first
QUANTIZATION_PREFERENCE = ["q4_K_M", "q5_K_M", "q4_0", "q6_K", "q3_K_M", "q8_0", "q5_0", "q2_K", "fp16"]

# Estimated FP16 model sizes (GB) for common parameter counts
MODEL_PARAM_SIZES = {
    "0.5b": 1.0, "0.6b": 1.2, "0.9b": 1.8, "1b": 2.0, "1.7b": 3.4,
    "3b": 6.0, "3.8b": 7.6, "4b": 8.0, "7b": 14.0, "8b": 16.0,
    "12b": 24.0, "14b": 28.0, "24b": 48.0, "27b": 54.0, "32b": 64.0, "70b": 140.0,
}

# VRAM requirements per quantization for common sizes
VRAM_REQUIREMENTS_GB = {
    # (params, quant) → approximate VRAM needed
    "1b": {"q4_K_M": 0.8, "q8_0": 1.5, "fp16": 2.0},
    "3b": {"q4_K_M": 2.0, "q8_0": 3.5, "fp16": 6.0},
    "4b": {"q4_K_M": 2.8, "q8_0": 4.5, "fp16": 8.0},
    "7b": {"q4_K_M": 4.5, "q8_0": 8.0, "fp16": 14.0},
    "8b": {"q4_K_M": 5.0, "q8_0": 9.0, "fp16": 16.0},
    "14b": {"q4_K_M": 8.5, "q8_0": 15.0, "fp16": 28.0},
    "27b": {"q4_K_M": 16.0, "q8_0": 28.0, "fp16": 54.0},
    "32b": {"q4_K_M": 19.0, "q8_0": 33.0, "fp16": 64.0},
    "70b": {"q4_K_M": 40.0, "q8_0": 75.0, "fp16": 140.0},
}


@dataclass
class ModelSizeInfo:
    """Size and quantization info for an installed model."""
    model_id: str
    size_bytes: int = 0
    size_gb: float = 0.0
    parameter_count: str = ""  # e.g., "8b", "14b"
    quantization: str = ""     # e.g., "q4_K_M", "fp16"
    family: str = ""           # e.g., "qwen3", "phi4"
    format: str = ""           # e.g., "gguf"


@dataclass
class CompressionRecommendation:
    """A recommendation to compress/quantize a model."""
    model_id: str
    current_size_gb: float
    current_quant: str
    recommended_model: str
    recommended_quant: str
    estimated_size_gb: float
    savings_gb: float
    savings_percent: float
    quality_retention: float
    reason: str


# ═══════════════════════════════════════════════════════
#  Quantization Manager
# ═══════════════════════════════════════════════════════

class QuantizationManager:
    """Manages model quantization, compression, and space optimization."""

    def __init__(self, brain=None):
        self.brain = brain
        self._model_info: Dict[str, ModelSizeInfo] = {}
        self._available_vram_gb: float = 0.0
        self._total_disk_usage_gb: float = 0.0

    async def scan_models(self) -> Dict[str, ModelSizeInfo]:
        """Scan all installed Ollama models and collect size/quantization info."""
        import aiohttp
        url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        try:
            session = self.brain._sessions.get("ollama") if self.brain else None
            if not session or session.closed:
                connector = aiohttp.TCPConnector(limit=10)
                session = aiohttp.ClientSession(connector=connector)
                close_session = True
            else:
                close_session = False

            # Get list of installed models with sizes
            resp = await session.get(f"{url}/api/tags", timeout=aiohttp.ClientTimeout(total=10))
            data = await resp.json()

            self._model_info.clear()
            self._total_disk_usage_gb = 0.0

            for model in data.get("models", []):
                name = model.get("name", "")
                size_bytes = model.get("size", 0)
                size_gb = size_bytes / (1024 ** 3)
                self._total_disk_usage_gb += size_gb

                # Parse model details
                info = ModelSizeInfo(
                    model_id=f"ollama/{name}",
                    size_bytes=size_bytes,
                    size_gb=round(size_gb, 2),
                )

                # Detect parameter count and quantization from model details
                details = model.get("details", {})
                info.family = details.get("family", "")
                info.format = details.get("format", "")
                info.parameter_count = details.get("parameter_size", "")
                info.quantization = details.get("quantization_level", "")

                self._model_info[info.model_id] = info

            if close_session:
                await session.close()

            logger.info(f"Scanned {len(self._model_info)} models, "
                       f"total disk: {self._total_disk_usage_gb:.1f} GB")
            return dict(self._model_info)

        except Exception as e:
            logger.warning(f"Model scan failed: {e}")
            return {}

    async def get_model_details(self, model_name: str) -> Dict:
        """Get detailed info about a specific model including quantization."""
        import aiohttp
        url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        try:
            session = self.brain._sessions.get("ollama") if self.brain else None
            if not session or session.closed:
                connector = aiohttp.TCPConnector(limit=10)
                session = aiohttp.ClientSession(connector=connector)
                close_session = True
            else:
                close_session = False

            clean_name = model_name.replace("ollama/", "")
            resp = await session.post(
                f"{url}/api/show",
                json={"name": clean_name},
                timeout=aiohttp.ClientTimeout(total=10),
            )
            data = await resp.json()

            if close_session:
                await session.close()

            return data
        except Exception as e:
            logger.warning(f"Failed to get model details for {model_name}: {e}")
            return {}

    def get_compression_recommendations(self, max_vram_gb: float = 0) -> List[CompressionRecommendation]:
        """Analyze installed models and recommend quantized alternatives."""
        recommendations = []

        for model_id, info in self._model_info.items():
            # Skip already-compressed or tiny models
            if info.size_gb < 1.0:
                continue

            current_quant = info.quantization.lower() if info.quantization else "unknown"

            # If already well-quantized (q4_K_M or below), skip
            if current_quant in ("q4_k_m", "q4_0", "q3_k_m", "q2_k"):
                continue

            # Recommend q4_K_M as default compression target
            target_quant = "q4_K_M"
            quant_info = QUANTIZATION_LEVELS.get(target_quant)
            if not quant_info:
                continue

            estimated_new_size = info.size_gb * quant_info.size_ratio
            savings = info.size_gb - estimated_new_size

            if savings < 0.5:  # Only recommend if saving at least 500MB
                continue

            # Build recommended model name with quantization tag
            clean_name = model_id.replace("ollama/", "")
            base_name = clean_name.split(":")[0] if ":" in clean_name else clean_name
            param_size = info.parameter_count.lower() if info.parameter_count else ""

            if param_size:
                recommended_model = f"{base_name}:{param_size}-{target_quant.lower()}"
            else:
                recommended_model = f"{base_name}:{target_quant.lower()}"

            recommendations.append(CompressionRecommendation(
                model_id=model_id,
                current_size_gb=round(info.size_gb, 2),
                current_quant=current_quant,
                recommended_model=recommended_model,
                recommended_quant=target_quant,
                estimated_size_gb=round(estimated_new_size, 2),
                savings_gb=round(savings, 2),
                savings_percent=round((savings / info.size_gb) * 100, 1),
                quality_retention=quant_info.quality_retention,
                reason=f"Switch from {current_quant or 'fp16'} to {target_quant} — "
                       f"save {savings:.1f}GB with {quant_info.quality_retention*100:.0f}% quality retention",
            ))

        # Sort by potential savings (biggest savings first)
        recommendations.sort(key=lambda r: r.savings_gb, reverse=True)
        return recommendations

    async def compress_model(self, model_name: str, target_quant: str = "q4_K_M") -> Dict:
        """Pull a quantized version of a model to replace the larger one."""
        import aiohttp
        url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        clean_name = model_name.replace("ollama/", "")
        base_name = clean_name.split(":")[0] if ":" in clean_name else clean_name

        # Determine the quantized model tag
        quantized_name = f"{base_name}:{target_quant.lower()}"

        try:
            session = self.brain._sessions.get("ollama") if self.brain else None
            if not session or session.closed:
                connector = aiohttp.TCPConnector(limit=10)
                session = aiohttp.ClientSession(connector=connector)
                close_session = True
            else:
                close_session = False

            logger.info(f"Pulling quantized model: {quantized_name}")
            resp = await session.post(
                f"{url}/api/pull",
                json={"name": quantized_name, "stream": False},
                timeout=aiohttp.ClientTimeout(total=3600),  # 1 hour for large models
            )

            if close_session:
                await session.close()

            if resp.status == 200:
                logger.info(f"Successfully pulled {quantized_name}")
                return {
                    "success": True,
                    "original": clean_name,
                    "quantized": quantized_name,
                    "target_quant": target_quant,
                }
            else:
                error_text = await resp.text()
                return {
                    "success": False,
                    "error": f"Pull failed with status {resp.status}: {error_text}",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def create_quantized_model(self, source_model: str, target_name: str,
                                     quantization: str = "q4_K_M") -> Dict:
        """Create a new quantized model from an existing one using Ollama Modelfile."""
        import aiohttp
        url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        clean_source = source_model.replace("ollama/", "")

        # Create a Modelfile that references the source model
        modelfile = f"""FROM {clean_source}
PARAMETER num_ctx 8192
TEMPLATE \"\"\"{{{{ .Prompt }}}}\"\"\"
"""

        try:
            session = self.brain._sessions.get("ollama") if self.brain else None
            if not session or session.closed:
                connector = aiohttp.TCPConnector(limit=10)
                session = aiohttp.ClientSession(connector=connector)
                close_session = True
            else:
                close_session = False

            logger.info(f"Creating quantized model: {target_name} from {clean_source}")
            resp = await session.post(
                f"{url}/api/create",
                json={
                    "name": target_name,
                    "modelfile": modelfile,
                    "stream": False,
                    "quantize": quantization,
                },
                timeout=aiohttp.ClientTimeout(total=3600),
            )
            result = await resp.json() if resp.status == 200 else {}

            if close_session:
                await session.close()

            if resp.status == 200:
                return {"success": True, "model": target_name, "quantization": quantization}
            else:
                return {"success": False, "error": str(result)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_optimal_quant_for_vram(self, param_count: str, available_vram_gb: float) -> str:
        """Given a model's parameter count and available VRAM, pick best quantization."""
        vram_reqs = VRAM_REQUIREMENTS_GB.get(param_count.lower(), {})

        # Try from highest quality to lowest, pick first that fits
        quant_by_quality = ["fp16", "q8_0", "q6_K", "q5_K_M", "q5_0", "q4_K_M", "q4_0", "q3_K_M", "q2_K"]
        for quant in quant_by_quality:
            required = vram_reqs.get(quant)
            if required and required <= available_vram_gb:
                return quant
            # Estimate if not in table
            if quant in QUANTIZATION_LEVELS:
                estimated_fp16 = MODEL_PARAM_SIZES.get(param_count.lower(), 0)
                estimated_vram = estimated_fp16 * QUANTIZATION_LEVELS[quant].size_ratio * 1.2  # 20% overhead
                if estimated_vram <= available_vram_gb:
                    return quant

        return "q2_K"  # Absolute minimum

    def get_space_report(self) -> Dict:
        """Get a comprehensive report on model storage and optimization opportunities."""
        recommendations = self.get_compression_recommendations()
        total_potential_savings = sum(r.savings_gb for r in recommendations)

        # Group by family
        by_family = {}
        for model_id, info in self._model_info.items():
            family = info.family or "unknown"
            if family not in by_family:
                by_family[family] = {"count": 0, "total_gb": 0.0, "models": []}
            by_family[family]["count"] += 1
            by_family[family]["total_gb"] += info.size_gb
            by_family[family]["models"].append({
                "id": model_id,
                "size_gb": round(info.size_gb, 2),
                "params": info.parameter_count,
                "quant": info.quantization,
            })

        return {
            "total_models": len(self._model_info),
            "total_disk_gb": round(self._total_disk_usage_gb, 2),
            "potential_savings_gb": round(total_potential_savings, 2),
            "recommendations_count": len(recommendations),
            "recommendations": [
                {
                    "model": r.model_id,
                    "current_size_gb": r.current_size_gb,
                    "recommended": r.recommended_model,
                    "new_size_gb": r.estimated_size_gb,
                    "savings_gb": r.savings_gb,
                    "savings_percent": r.savings_percent,
                    "quality_kept": f"{r.quality_retention*100:.0f}%",
                    "reason": r.reason,
                }
                for r in recommendations[:10]
            ],
            "by_family": {k: {"count": v["count"], "total_gb": round(v["total_gb"], 2)}
                         for k, v in by_family.items()},
            "quantization_levels": {
                k: {"bits": v.bits_per_weight, "quality": f"{v.quality_retention*100:.0f}%",
                    "size_ratio": f"{v.size_ratio*100:.0f}%", "use_case": v.recommended_for}
                for k, v in QUANTIZATION_LEVELS.items()
            },
        }

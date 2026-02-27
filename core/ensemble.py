"""
Neural Brain Ensemble Engine v2.0
Smart multi-model orchestrator — 100% local-first.

Routes queries to the best available local specialist model
based on content analysis. Supports any hardware tier from
2GB VRAM laptops to 48GB+ workstations.

Auto-detects available models and picks the best one you can run.
"""

import asyncio
import time
import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("neural-brain.ensemble")


# ═══════════════════════════════════════════════
#  QUERY CLASSIFICATION
# ═══════════════════════════════════════════════

class QueryType:
    CODE = "code"
    REASONING = "reasoning"
    VISION = "vision"
    TRANSLATION = "translation"
    OCR = "ocr"
    GENERAL = "general"
    CREATIVE = "creative"
    MATH = "math"


# Keywords and patterns for classification
CLASSIFICATION_RULES = {
    QueryType.CODE: {
        "keywords": [
            "code", "function", "class", "def ", "import ", "variable", "bug", "fix",
            "python", "javascript", "typescript", "react", "node", "api", "endpoint",
            "database", "sql", "query", "git", "deploy", "docker", "npm", "pip",
            "error", "exception", "debug", "compile", "syntax", "algorithm",
            "html", "css", "json", "yaml", "xml", "regex", "bash", "shell",
            "async", "await", "promise", "callback", "loop", "array", "list",
            "dict", "object", "string", "integer", "float", "boolean",
            "file", "read", "write", "create", "delete", "update", "script",
            "install", "package", "library", "framework", "module", "component",
            "test", "unittest", "pytest", "jest", "build", "webpack", "vite",
            "```", "console.log", "print(", "return ", "if ", "for ", "while ",
        ],
        "patterns": [
            r"```\w+", r"def \w+", r"class \w+", r"import \w+", r"from \w+ import",
            r"function \w+", r"const \w+", r"let \w+", r"var \w+",
            r"\w+\.\w+\(", r"pip install", r"npm install",
        ],
        "weight": 1.5,
    },
    QueryType.REASONING: {
        "keywords": [
            "why", "explain", "reason", "think", "analyze", "compare", "evaluate",
            "pros and cons", "trade-off", "best approach", "strategy", "plan",
            "logic", "argument", "evidence", "proof", "deduce", "infer",
            "step by step", "break down", "deep dive", "implications",
            "what if", "scenario", "hypothesis", "cause", "effect",
            "philosophy", "ethics", "moral", "decision", "critical",
        ],
        "patterns": [
            r"why (does|is|are|do|did|would|should|can|could)",
            r"explain (how|why|what)",
            r"what (are|is) the (best|difference|reason|cause)",
            r"how (does|do|would|should|can|could) .+ (work|compare|differ)",
        ],
        "weight": 1.2,
    },
    QueryType.MATH: {
        "keywords": [
            "calculate", "compute", "solve", "equation", "formula", "integral",
            "derivative", "matrix", "vector", "probability", "statistics",
            "algebra", "geometry", "calculus", "trigonometry", "logarithm",
            "sum", "product", "factorial", "permutation", "combination",
        ],
        "patterns": [
            r"\d+\s*[\+\-\*\/\^]\s*\d+",
            r"solve\s+(for|the)",
            r"(what|find)\s+is\s+\d+",
        ],
        "weight": 1.3,
    },
    QueryType.TRANSLATION: {
        "keywords": [
            "translate", "translation", "tamil", "arabic", "hindi", "french",
            "spanish", "german", "chinese", "japanese", "korean", "language",
            "localize", "i18n", "multilingual",
        ],
        "patterns": [
            r"translate .+ (to|into|from) \w+",
            r"(say|write|how to say) .+ in \w+",
        ],
        "weight": 2.0,
    },
    QueryType.CREATIVE: {
        "keywords": [
            "write", "story", "poem", "creative", "imagine", "fiction",
            "blog", "article", "essay", "letter", "email draft",
            "marketing", "slogan", "tagline", "pitch", "presentation",
        ],
        "patterns": [
            r"write (a|an|me|the) \w+",
            r"create (a|an) (story|poem|blog|article)",
        ],
        "weight": 1.0,
    },
    QueryType.VISION: {
        "keywords": [
            "image", "picture", "photo", "screenshot", "diagram", "chart",
            "look at", "what do you see", "describe this", "visual",
        ],
        "patterns": [r"(look|see|analyze|describe) (this|the) (image|picture|photo|screenshot)"],
        "weight": 2.0,
    },
    QueryType.OCR: {
        "keywords": [
            "ocr", "extract text", "read text from", "scan document",
            "handwriting", "receipt", "invoice text",
        ],
        "patterns": [r"(extract|read|get) (the )?text (from|in)"],
        "weight": 2.5,
    },
}

# ═══════════════════════════════════════════════
#  MODEL SPECIALISTS — 100% Local-First Arsenal
#  Tiered: Large → Medium → Small → Tiny fallbacks
#  Auto-selects best model available on your hardware
# ═══════════════════════════════════════════════

SPECIALIST_MODELS = {
    QueryType.CODE: [
        "ollama/qwen2.5-coder:32b",       # Best local coding (24GB VRAM)
        "ollama/devstral",                 # SWE agent (16GB VRAM)
        "ollama/qwen2.5-coder:14b",       # Strong code (8GB VRAM)
        "ollama/qwen3:8b",                # Good code + reasoning (4GB VRAM)
        "ollama/qwen2.5-coder:7b",        # Efficient code (4GB VRAM)
        "ollama/qwen3:4b",                # Tiny but capable
        "ollama/phi4-mini",               # STEM strong at 3.8B
    ],
    QueryType.REASONING: [
        "ollama/deepseek-r1:32b",          # Best local reasoning (24GB)
        "ollama/qwen3:32b",                # Frontier-class (20GB)
        "ollama/deepseek-r1:14b",          # Strong reasoning (8GB)
        "ollama/phi4-reasoning",           # Dedicated reasoner (8GB)
        "ollama/phi4",                     # Excellent at 14B
        "ollama/qwen3:8b",                # Hybrid thinking
        "ollama/deepseek-r1:8b",           # Efficient reasoning
        "ollama/qwen3:4b",                # Tiny but thinks
    ],
    QueryType.MATH: [
        "ollama/deepseek-r1:32b",          # Best math reasoning
        "ollama/qwen3:32b",                # Strong math
        "ollama/deepseek-r1:14b",          # Good math
        "ollama/phi4-reasoning",           # STEM specialist
        "ollama/phi4",                     # Strong STEM
        "ollama/deepseek-r1:8b",           # Efficient
        "ollama/phi4-mini",               # STEM at 3.8B
        "ollama/qwen3:4b",                # Tiny math
    ],
    QueryType.GENERAL: [
        "ollama/qwen3:32b",                # Best open model
        "ollama/gemma3:27b",               # Near-GPT-4
        "ollama/glm-4.7-flash",            # 30B MoE
        "ollama/qwen3:14b",               # Near-frontier
        "ollama/phi4",                     # Excellent 14B
        "ollama/qwen3:8b",                # Best 8B
        "ollama/gemma3",                   # Google 12B
        "ollama/qwen3:4b",                # Beats 7B models
        "ollama/phi4-mini",               # 3.8B solid
        "ollama/llama3.2:3b",              # Ultra-fast
    ],
    QueryType.CREATIVE: [
        "ollama/qwen3:32b",                # Best creative quality
        "ollama/gemma3:27b",               # Strong creative
        "ollama/mistral-small3.1",         # Good writing
        "ollama/qwen3:14b",               # Good creative
        "ollama/phi4",                     # Solid
        "ollama/qwen3:8b",                # Efficient
        "ollama/gemma3",                   # Google
        "ollama/qwen3:4b",                # Tiny creative
    ],
    QueryType.VISION: [
        "ollama/mistral-small3.1",         # 24B vision+text
        "ollama/qwen3-vl",                # Best open VLM
        "ollama/llama3.2-vision",          # 11B efficient VLM
        "ollama/gemma3:27b",               # 27B with vision
        "ollama/gemma3",                   # 12B with vision
        "ollama/gemma3:4b",               # 4B with vision
        "ollama/llava",                    # Lightweight
        "ollama/llama4:scout",             # 512K context
    ],
    QueryType.TRANSLATION: [
        "ollama/translategemma",           # 55 languages specialist
        "ollama/qwen3:32b",                # Strong multilingual
        "ollama/glm-4.7-flash",            # Good multilingual
        "ollama/qwen3:8b",                # Decent multilingual
        "ollama/gemma3",                   # 140+ languages
        "ollama/qwen3:4b",                # Basic multilingual
    ],
    QueryType.OCR: [
        "ollama/glm-ocr",                 # #1 document OCR (0.9B!)
        "ollama/qwen3-vl",                # Vision+text extraction
        "ollama/mistral-small3.1",         # Strong vision
        "ollama/llama3.2-vision",          # Efficient VLM
        "ollama/gemma3:4b",               # Light vision
    ],
}

JUDGE_MODELS = [
    "ollama/qwen3:8b",                    # Best 8B judge
    "ollama/phi4",                         # Strong reasoning judge
    "ollama/qwen3:4b",                    # Tiny but good judge
    "ollama/phi4-mini",                   # Fallback tiny judge
    "ollama/phi3:mini",                    # Legacy fallback
]


@dataclass
class ClassificationResult:
    query_type: str
    confidence: float
    scores: Dict[str, float]
    specialist: str
    fallbacks: List[str]


@dataclass
class EnsembleResult:
    content: str
    model_used: str
    query_type: str
    confidence: float
    latency_ms: float
    models_consulted: List[str]
    verified: bool = False
    verification_model: str = ""


class EnsembleEngine:
    """Smart multi-model orchestrator."""

    def __init__(self, brain):
        self.brain = brain
        self.stats = {
            "total_queries": 0,
            "by_type": {},
            "by_model": {},
            "avg_latency_ms": 0,
            "ensemble_calls": 0,
            "single_calls": 0,
        }

    def classify_query(self, messages: List[Dict], system: str = "") -> ClassificationResult:
        user_text = ""
        has_image = False
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_text = content.lower()
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                user_text = part.get("text", "").lower()
                            elif part.get("type") in ("image", "image_url"):
                                has_image = True
                break

        full_text = f"{system.lower()} {user_text}"
        scores = {}
        for qtype, rules in CLASSIFICATION_RULES.items():
            score = 0.0
            for kw in rules["keywords"]:
                if kw in full_text:
                    score += 1.0
            for pattern in rules["patterns"]:
                if re.search(pattern, full_text, re.IGNORECASE):
                    score += 2.0
            score *= rules["weight"]
            scores[qtype] = score

        if has_image:
            scores[QueryType.VISION] = max(scores.get(QueryType.VISION, 0), 10.0)

        if all(s == 0 for s in scores.values()):
            best_type = QueryType.GENERAL
            confidence = 0.5
        else:
            best_type = max(scores, key=scores.get)
            total = sum(scores.values())
            confidence = scores[best_type] / total if total > 0 else 0.5

        specialists = SPECIALIST_MODELS.get(best_type, SPECIALIST_MODELS[QueryType.GENERAL])
        available = [m for m in specialists if m in self.brain.models and self.brain.models[m].enabled]
        if not available:
            available = [m.id for m in self.brain.models.values()
                        if m.provider.value == "ollama" and m.enabled]

        specialist = available[0] if available else "ollama/phi4:latest"
        fallbacks = available[1:4] if len(available) > 1 else []

        return ClassificationResult(
            query_type=best_type, confidence=confidence, scores=scores,
            specialist=specialist, fallbacks=fallbacks,
        )

    async def complete(self, messages: List[Dict], system: str = "",
                       temperature: float = 0.7, max_tokens: int = 8192,
                       mode: str = "smart", verify: bool = False, **kwargs) -> EnsembleResult:
        start = time.time()
        self.stats["total_queries"] += 1
        classification = self.classify_query(messages, system)
        qtype = classification.query_type
        self.stats["by_type"][qtype] = self.stats["by_type"].get(qtype, 0) + 1

        logger.info(f"[Ensemble] '{qtype}' (conf={classification.confidence:.0%}) → {classification.specialist}")

        if mode == "fastest":
            return await self._single_model(
                JUDGE_MODELS[0], messages, system, temperature, max_tokens,
                qtype, classification.confidence, start)
        if mode == "strongest":
            for m in ["ollama/qwen3:32b", "ollama/gemma3:27b", "ollama/deepseek-r1:32b",
                       "ollama/qwen2.5-coder:32b", "ollama/glm-4.7-flash",
                       "ollama/qwen3:14b", "ollama/phi4", "ollama/qwen3:8b"]:
                if m in self.brain.models:
                    return await self._single_model(
                        m, messages, system, temperature, max_tokens,
                        qtype, classification.confidence, start)
        if mode == "consensus":
            return await self._consensus(classification, messages, system, temperature, max_tokens, start)
        if mode == "chain":
            return await self._chain(classification, messages, system, temperature, max_tokens, start)

        result = await self._smart_route(classification, messages, system, temperature, max_tokens, start)
        if verify and result.content and classification.confidence < 0.7:
            result = await self._verify(result, messages, system, start)
        return result

    async def _single_model(self, model_id, messages, system, temperature, max_tokens,
                            qtype, confidence, start) -> EnsembleResult:
        try:
            resp = await self._call_model(model_id, messages, system, temperature, max_tokens)
            return EnsembleResult(content=resp, model_used=model_id, query_type=qtype,
                                confidence=confidence, latency_ms=(time.time()-start)*1000,
                                models_consulted=[model_id])
        except Exception as e:
            return EnsembleResult(content=f"Model {model_id} failed: {e}", model_used="none",
                                query_type=qtype, confidence=0,
                                latency_ms=(time.time()-start)*1000, models_consulted=[model_id])

    async def _smart_route(self, classification, messages, system, temperature, max_tokens, start) -> EnsembleResult:
        self.stats["single_calls"] += 1
        models_tried = []
        for model_id in [classification.specialist] + classification.fallbacks:
            models_tried.append(model_id)
            try:
                resp = await self._call_model(model_id, messages, system, temperature, max_tokens)
                if resp and resp.strip():
                    self.stats["by_model"][model_id] = self.stats["by_model"].get(model_id, 0) + 1
                    return EnsembleResult(content=resp, model_used=model_id,
                                        query_type=classification.query_type,
                                        confidence=classification.confidence,
                                        latency_ms=(time.time()-start)*1000,
                                        models_consulted=models_tried)
            except Exception as e:
                logger.warning(f"[Ensemble] {model_id} failed: {e}")
                continue
        return EnsembleResult(
            content="All local models unavailable. Check Ollama is running.",
            model_used="none", query_type=classification.query_type, confidence=0,
            latency_ms=(time.time()-start)*1000, models_consulted=models_tried)

    async def _consensus(self, classification, messages, system, temperature, max_tokens, start) -> EnsembleResult:
        self.stats["ensemble_calls"] += 1
        models_consulted = []
        primary_model = classification.specialist
        primary_resp = ""
        try:
            primary_resp = await self._call_model(primary_model, messages, system, temperature, max_tokens)
            models_consulted.append(primary_model)
        except Exception as e:
            logger.warning(f"[Ensemble] Primary {primary_model} failed: {e}")

        secondary_model = classification.fallbacks[0] if classification.fallbacks else None
        secondary_resp = ""
        if secondary_model and secondary_model != primary_model:
            try:
                secondary_resp = await self._call_model(secondary_model, messages, system, temperature, max_tokens)
                models_consulted.append(secondary_model)
            except Exception as e:
                logger.warning(f"[Ensemble] Secondary {secondary_model} failed: {e}")

        if not secondary_resp:
            return EnsembleResult(content=primary_resp or "No response.", model_used=primary_model,
                                query_type=classification.query_type, confidence=classification.confidence,
                                latency_ms=(time.time()-start)*1000, models_consulted=models_consulted)

        judge_model = self._get_judge_model(exclude=[primary_model, secondary_model])
        judge_prompt = [{"role": "user", "content": f"""You are a quality judge. Two AI models answered the same question.
Pick the BEST answer or synthesize both into an improved response.

QUESTION: {self._get_last_user_message(messages)}

ANSWER A ({primary_model}):
{primary_resp[:3000]}

ANSWER B ({secondary_model}):
{secondary_resp[:3000]}

Provide the best possible answer directly. Do NOT mention judging or two answers."""}]
        try:
            final = await self._call_model(judge_model, judge_prompt, "", 0.3, max_tokens)
            models_consulted.append(f"judge:{judge_model}")
            return EnsembleResult(
                content=final, model_used=f"ensemble({primary_model}+{secondary_model}→{judge_model})",
                query_type=classification.query_type, confidence=classification.confidence,
                latency_ms=(time.time()-start)*1000, models_consulted=models_consulted,
                verified=True, verification_model=judge_model)
        except Exception:
            return EnsembleResult(content=primary_resp, model_used=primary_model,
                                query_type=classification.query_type, confidence=classification.confidence,
                                latency_ms=(time.time()-start)*1000, models_consulted=models_consulted)

    async def _chain(self, classification, messages, system, temperature, max_tokens, start) -> EnsembleResult:
        self.stats["ensemble_calls"] += 1
        models_consulted = []
        primary_model = classification.specialist
        try:
            primary_resp = await self._call_model(primary_model, messages, system, temperature, max_tokens)
            models_consulted.append(primary_model)
        except Exception:
            return await self._smart_route(classification, messages, system, temperature, max_tokens, start)

        refiner = self._get_judge_model(exclude=[primary_model])
        refine_prompt = [{"role": "user", "content": f"""Review and improve this response. Fix errors, add missing details, improve clarity.
If already excellent, return as-is.

QUESTION: {self._get_last_user_message(messages)}

RESPONSE TO IMPROVE:
{primary_resp[:4000]}

Provide the improved response directly."""}]
        try:
            refined = await self._call_model(refiner, refine_prompt, "", 0.3, max_tokens)
            models_consulted.append(f"refiner:{refiner}")
            return EnsembleResult(content=refined, model_used=f"chain({primary_model}→{refiner})",
                                query_type=classification.query_type, confidence=classification.confidence,
                                latency_ms=(time.time()-start)*1000, models_consulted=models_consulted,
                                verified=True, verification_model=refiner)
        except Exception:
            return EnsembleResult(content=primary_resp, model_used=primary_model,
                                query_type=classification.query_type, confidence=classification.confidence,
                                latency_ms=(time.time()-start)*1000, models_consulted=models_consulted)

    async def _verify(self, result, messages, system, start) -> EnsembleResult:
        verifier = self._get_judge_model(exclude=[result.model_used])
        verify_prompt = [{"role": "user", "content": f"""Check this response for accuracy. If correct, return exactly. If errors, fix them.

QUESTION: {self._get_last_user_message(messages)}

RESPONSE: {result.content[:4000]}

Return verified response directly."""}]
        try:
            verified = await self._call_model(verifier, verify_prompt, "", 0.2, 4096)
            result.verified = True
            result.verification_model = verifier
            result.content = verified
            result.models_consulted.append(f"verifier:{verifier}")
            result.latency_ms = (time.time() - start) * 1000
        except Exception:
            pass
        return result

    async def _call_model(self, model_id, messages, system, temperature, max_tokens) -> str:
        from core.brain import CompletionRequest
        # Sanitize messages for Ollama — convert 'developer' role to 'system'
        clean_msgs = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "developer":
                role = "system"
            elif role not in ("system", "user", "assistant", "tool"):
                role = "user"
            # Only include text content (skip image_url etc for text models)
            content = msg.get("content", "")
            if isinstance(content, list):
                # Extract text parts only
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = "\n".join(text_parts) if text_parts else ""
            if content:
                clean_msgs.append({"role": role, "content": content})
        if not clean_msgs:
            clean_msgs = [{"role": "user", "content": "hello"}]
        logger.info(f"[CALL] {model_id} with {len(clean_msgs)} msgs (cleaned from {len(messages)})")
        req = CompletionRequest(messages=clean_msgs, model=model_id, system=system,
                               temperature=temperature, max_tokens=max_tokens)
        resp = await self.brain.complete(req)
        return resp.content

    def _get_judge_model(self, exclude=None) -> str:
        exclude = exclude or []
        for m in JUDGE_MODELS:
            if m not in exclude and m in self.brain.models:
                return m
        for m in self.brain.models.values():
            if m.is_local and m.id not in exclude:
                return m.id
        return JUDGE_MODELS[0]

    def _get_last_user_message(self, messages) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content[:1000]
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            return part.get("text", "")[:1000]
        return ""

    def get_stats(self) -> Dict:
        return {**self.stats,
                "specialist_models": {q: m[:2] for q, m in SPECIALIST_MODELS.items()},
                "judge_models": JUDGE_MODELS}

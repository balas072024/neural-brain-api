"""
Neural Brain Self-Learning Engine v1.0
Adaptive model routing that learns from every request.

SELF-LEARNING LOOP:
1. Track every request: model, query_type, latency, success/fail, user feedback
2. Compute rolling performance scores per model per query type
3. Adjust routing weights: boost high-performers, demote failures
4. Persist learning data to disk — survives restarts
5. Auto-update ensemble specialist rankings based on observed performance

The system gets smarter with every query — no manual tuning needed.
"""

import os
import json
import time
import math
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("neural-brain.learning")

# ═══════════════════════════════════════════════════════
#  Performance Record — one per model per query type
# ═══════════════════════════════════════════════════════

@dataclass
class ModelPerformance:
    """Rolling performance metrics for a model on a specific query type."""
    model_id: str
    query_type: str
    total_requests: int = 0
    successes: int = 0
    failures: int = 0
    avg_latency_ms: float = 0.0
    avg_quality_score: float = 0.5  # 0.0 = worst, 1.0 = best
    positive_feedback: int = 0
    negative_feedback: int = 0
    total_tokens: int = 0
    ema_latency_ms: float = 0.0    # Exponential moving average
    ema_quality: float = 0.5       # Exponential moving average
    routing_weight: float = 1.0    # Adaptive weight for routing (higher = preferred)
    last_used: float = 0.0
    last_failure: float = 0.0
    consecutive_failures: int = 0


# ═══════════════════════════════════════════════════════
#  Self-Learning Engine
# ═══════════════════════════════════════════════════════

class SelfLearningEngine:
    """
    Learns which models perform best for each query type.
    Automatically adjusts routing weights based on observed performance.
    """

    # EMA decay factors
    LATENCY_DECAY = 0.15   # Recent latency matters more
    QUALITY_DECAY = 0.2    # Quality changes slowly
    WEIGHT_DECAY = 0.1     # Weight adjustments are gradual

    # Scoring weights for composite routing score
    SCORE_WEIGHTS = {
        "success_rate": 0.30,
        "quality": 0.30,
        "latency": 0.20,
        "feedback": 0.20,
    }

    # Minimum requests before a model's score is trusted
    MIN_REQUESTS_FOR_TRUST = 5

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
        os.makedirs(self.data_dir, exist_ok=True)
        self._data_file = os.path.join(self.data_dir, "learning_data.json")
        self._feedback_file = os.path.join(self.data_dir, "feedback_log.jsonl")

        # model_id -> query_type -> ModelPerformance
        self.performance: Dict[str, Dict[str, ModelPerformance]] = defaultdict(dict)

        # Global model rankings (computed from performance data)
        self.rankings: Dict[str, List[str]] = {}  # query_type -> [model_ids ordered by score]

        # Learning event log (recent events for debugging)
        self._recent_events: List[Dict] = []

        self._load()

    # ─── Core Learning Methods ───

    def record_request(self, model_id: str, query_type: str, latency_ms: float,
                       success: bool, tokens: int = 0, quality_score: float = -1):
        """Record the outcome of a request for learning."""
        perf = self._get_or_create(model_id, query_type)
        perf.total_requests += 1
        perf.last_used = time.time()
        perf.total_tokens += tokens

        if success:
            perf.successes += 1
            perf.consecutive_failures = 0

            # Update latency EMA
            if perf.ema_latency_ms == 0:
                perf.ema_latency_ms = latency_ms
            else:
                perf.ema_latency_ms = (
                    (1 - self.LATENCY_DECAY) * perf.ema_latency_ms +
                    self.LATENCY_DECAY * latency_ms
                )
            perf.avg_latency_ms = (
                (perf.avg_latency_ms * (perf.successes - 1) + latency_ms) / perf.successes
            )

            # Update quality EMA if provided
            if quality_score >= 0:
                if perf.ema_quality == 0.5 and perf.total_requests <= 1:
                    perf.ema_quality = quality_score
                else:
                    perf.ema_quality = (
                        (1 - self.QUALITY_DECAY) * perf.ema_quality +
                        self.QUALITY_DECAY * quality_score
                    )
                perf.avg_quality_score = perf.ema_quality
        else:
            perf.failures += 1
            perf.last_failure = time.time()
            perf.consecutive_failures += 1

        # Recompute routing weight
        perf.routing_weight = self._compute_weight(perf)

        # Update rankings for this query type
        self._update_rankings(query_type)

        # Log event
        self._recent_events.append({
            "model": model_id, "query_type": query_type,
            "latency_ms": round(latency_ms, 1), "success": success,
            "weight": round(perf.routing_weight, 3),
            "timestamp": time.time(),
        })
        if len(self._recent_events) > 500:
            self._recent_events = self._recent_events[-250:]

    def record_feedback(self, model_id: str, query_type: str, positive: bool,
                        request_id: str = "", details: str = ""):
        """Record user feedback (thumbs up/down) for a model response."""
        perf = self._get_or_create(model_id, query_type)

        if positive:
            perf.positive_feedback += 1
            # Boost quality score slightly on positive feedback
            perf.ema_quality = min(1.0, perf.ema_quality + 0.02)
        else:
            perf.negative_feedback += 1
            # Lower quality score on negative feedback
            perf.ema_quality = max(0.0, perf.ema_quality - 0.05)

        perf.avg_quality_score = perf.ema_quality
        perf.routing_weight = self._compute_weight(perf)
        self._update_rankings(query_type)

        # Persist feedback to log
        try:
            with open(self._feedback_file, "a") as f:
                f.write(json.dumps({
                    "model": model_id, "query_type": query_type,
                    "positive": positive, "request_id": request_id,
                    "details": details, "timestamp": time.time(),
                }) + "\n")
        except Exception:
            pass

    def get_ranked_models(self, query_type: str, available_models: List[str] = None) -> List[Tuple[str, float]]:
        """Get models ranked by learned performance for a query type.
        Returns [(model_id, score), ...] sorted best-first."""
        if query_type not in self.rankings or not self.rankings[query_type]:
            return []

        ranked = []
        for model_id in self.rankings[query_type]:
            if available_models and model_id not in available_models:
                continue
            perf = self.performance.get(model_id, {}).get(query_type)
            score = perf.routing_weight if perf else 0.5
            ranked.append((model_id, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def get_best_model(self, query_type: str, available_models: List[str] = None) -> Optional[str]:
        """Get the single best model for a query type based on learned performance."""
        ranked = self.get_ranked_models(query_type, available_models)
        return ranked[0][0] if ranked else None

    def should_explore(self, model_id: str, query_type: str) -> bool:
        """Should we try this model to gather more data? (Exploration vs exploitation)."""
        perf = self.performance.get(model_id, {}).get(query_type)
        if not perf:
            return True  # Never tried — always explore
        if perf.total_requests < self.MIN_REQUESTS_FOR_TRUST:
            return True  # Not enough data — explore
        # Occasionally explore even well-known models (5% chance)
        import random
        return random.random() < 0.05

    # ─── Weight Computation ───

    def _compute_weight(self, perf: ModelPerformance) -> float:
        """Compute composite routing weight from all performance metrics."""
        if perf.total_requests == 0:
            return 0.5  # Neutral — no data yet

        # 1. Success rate score (0-1)
        success_rate = perf.successes / perf.total_requests if perf.total_requests > 0 else 0.5

        # Penalize consecutive failures heavily
        if perf.consecutive_failures >= 3:
            success_rate *= 0.3
        elif perf.consecutive_failures >= 2:
            success_rate *= 0.6

        # 2. Quality score (0-1) — from EMA
        quality = perf.ema_quality

        # 3. Latency score (0-1) — lower is better, sigmoid normalization
        if perf.ema_latency_ms > 0:
            # 500ms = 0.73, 1000ms = 0.5, 2000ms = 0.27, 5000ms = 0.07
            latency_score = 1.0 / (1.0 + math.exp((perf.ema_latency_ms - 1000) / 500))
        else:
            latency_score = 0.5  # Unknown

        # 4. Feedback score (0-1)
        total_feedback = perf.positive_feedback + perf.negative_feedback
        if total_feedback > 0:
            feedback_score = perf.positive_feedback / total_feedback
        else:
            feedback_score = 0.5  # Neutral

        # Composite score
        weight = (
            self.SCORE_WEIGHTS["success_rate"] * success_rate +
            self.SCORE_WEIGHTS["quality"] * quality +
            self.SCORE_WEIGHTS["latency"] * latency_score +
            self.SCORE_WEIGHTS["feedback"] * feedback_score
        )

        # Confidence adjustment: less data = weight closer to 0.5
        confidence = min(1.0, perf.total_requests / (self.MIN_REQUESTS_FOR_TRUST * 2))
        weight = 0.5 + (weight - 0.5) * confidence

        # Recency penalty: demote models not used recently (>24h)
        if perf.last_used > 0:
            hours_since = (time.time() - perf.last_used) / 3600
            if hours_since > 24:
                weight *= max(0.7, 1.0 - (hours_since - 24) / 168)  # Gradual decay over a week

        return max(0.01, min(1.0, weight))

    def _update_rankings(self, query_type: str):
        """Recompute rankings for a query type."""
        models_for_type = []
        for model_id, types in self.performance.items():
            if query_type in types:
                perf = types[query_type]
                models_for_type.append((model_id, perf.routing_weight))

        models_for_type.sort(key=lambda x: x[1], reverse=True)
        self.rankings[query_type] = [m[0] for m in models_for_type]

    def _get_or_create(self, model_id: str, query_type: str) -> ModelPerformance:
        if query_type not in self.performance[model_id]:
            self.performance[model_id][query_type] = ModelPerformance(
                model_id=model_id, query_type=query_type
            )
        return self.performance[model_id][query_type]

    # ─── Persistence ───

    def save(self):
        """Persist learning data to disk."""
        try:
            data = {}
            for model_id, types in self.performance.items():
                data[model_id] = {}
                for qtype, perf in types.items():
                    data[model_id][qtype] = asdict(perf)

            with open(self._data_file, "w") as f:
                json.dump({
                    "version": "1.0",
                    "performance": data,
                    "rankings": self.rankings,
                    "saved_at": time.time(),
                }, f, indent=2)
            logger.info(f"Learning data saved: {sum(len(t) for t in self.performance.values())} records")
        except Exception as e:
            logger.warning(f"Failed to save learning data: {e}")

    def _load(self):
        """Load learning data from disk."""
        if not os.path.exists(self._data_file):
            return
        try:
            with open(self._data_file) as f:
                raw = json.load(f)

            for model_id, types in raw.get("performance", {}).items():
                for qtype, perf_data in types.items():
                    perf = ModelPerformance(**perf_data)
                    self.performance[model_id][qtype] = perf

            self.rankings = raw.get("rankings", {})
            total = sum(len(t) for t in self.performance.values())
            logger.info(f"Loaded learning data: {total} records from {len(self.performance)} models")
        except Exception as e:
            logger.warning(f"Failed to load learning data: {e}")

    # ─── Analytics & Insights ───

    def get_insights(self) -> Dict:
        """Get learning insights and current model rankings."""
        total_records = sum(len(t) for t in self.performance.values())
        total_requests = sum(
            perf.total_requests
            for types in self.performance.values()
            for perf in types.values()
        )
        total_feedback = sum(
            perf.positive_feedback + perf.negative_feedback
            for types in self.performance.values()
            for perf in types.values()
        )

        # Best model per query type
        best_per_type = {}
        for qtype, ranked_models in self.rankings.items():
            if ranked_models:
                model_id = ranked_models[0]
                perf = self.performance.get(model_id, {}).get(qtype)
                best_per_type[qtype] = {
                    "model": model_id,
                    "weight": round(perf.routing_weight, 3) if perf else 0,
                    "requests": perf.total_requests if perf else 0,
                    "success_rate": round(perf.successes / perf.total_requests, 3) if perf and perf.total_requests > 0 else 0,
                    "avg_latency_ms": round(perf.avg_latency_ms, 1) if perf else 0,
                }

        # Top performers overall
        all_models = []
        for model_id, types in self.performance.items():
            avg_weight = sum(p.routing_weight for p in types.values()) / len(types)
            total_reqs = sum(p.total_requests for p in types.values())
            all_models.append({
                "model": model_id,
                "avg_weight": round(avg_weight, 3),
                "total_requests": total_reqs,
                "query_types": list(types.keys()),
            })
        all_models.sort(key=lambda x: x["avg_weight"], reverse=True)

        return {
            "learning_active": True,
            "total_records": total_records,
            "total_requests_tracked": total_requests,
            "total_feedback": total_feedback,
            "models_tracked": len(self.performance),
            "query_types_tracked": len(self.rankings),
            "best_per_type": best_per_type,
            "top_models": all_models[:10],
            "recent_events": self._recent_events[-20:],
            "rankings": {k: v[:5] for k, v in self.rankings.items()},
        }

    def get_model_report(self, model_id: str) -> Dict:
        """Detailed performance report for a specific model."""
        types = self.performance.get(model_id, {})
        if not types:
            return {"model": model_id, "data": "no_data"}

        report = {"model": model_id, "query_types": {}}
        for qtype, perf in types.items():
            report["query_types"][qtype] = {
                "total_requests": perf.total_requests,
                "successes": perf.successes,
                "failures": perf.failures,
                "success_rate": round(perf.successes / perf.total_requests, 3) if perf.total_requests > 0 else 0,
                "avg_latency_ms": round(perf.avg_latency_ms, 1),
                "ema_latency_ms": round(perf.ema_latency_ms, 1),
                "quality_score": round(perf.avg_quality_score, 3),
                "positive_feedback": perf.positive_feedback,
                "negative_feedback": perf.negative_feedback,
                "routing_weight": round(perf.routing_weight, 3),
                "total_tokens": perf.total_tokens,
            }
        return report

    def reset(self):
        """Reset all learning data (start fresh)."""
        self.performance.clear()
        self.rankings.clear()
        self._recent_events.clear()
        if os.path.exists(self._data_file):
            os.remove(self._data_file)
        logger.info("Learning data reset")

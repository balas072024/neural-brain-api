"""
NEXUS Reflection Engine
========================
Self-improvement through generate-reflect-refine cycles.
Implements the Reflector-Evaluator-Meta Agent pattern.

Architecture:
1. Generate: Agent produces initial output
2. Critique: Critic agent evaluates quality
3. Refine: Original agent improves based on feedback
4. Evaluate: Score quality improvement
5. Learn: Store experience for future tasks

Inspired by Reflexion, LATS, and HealthFlow patterns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from nexus.core.config import NexusConfig


@dataclass
class ReflectionResult:
    """Result of a reflection cycle."""
    quality_score: float = 0.0
    feedback: str = ""
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    refined_output: str | None = None
    cycles_used: int = 0
    improvement_delta: float = 0.0
    duration_ms: float = 0


@dataclass
class QualityRubric:
    """Evaluation rubric for quality assessment."""
    correctness: float = 0.0      # Factual accuracy
    completeness: float = 0.0     # Covers all aspects
    clarity: float = 0.0          # Clear and well-structured
    relevance: float = 0.0        # Addresses the actual task
    actionability: float = 0.0    # Practical and usable

    @property
    def overall(self) -> float:
        scores = [self.correctness, self.completeness, self.clarity, self.relevance, self.actionability]
        return sum(scores) / len(scores) if scores else 0.0


class ReflectionEngine:
    """
    Self-improvement engine using reflection cycles.

    The engine follows the Generate → Critique → Refine pattern:
    1. An agent generates output
    2. A critic agent evaluates the output
    3. If below quality threshold, the original agent refines
    4. Repeat until quality threshold is met or max cycles reached
    5. Store experience for future improvement
    """

    def __init__(self, config: NexusConfig):
        self.config = config
        self._reflection_history: list[ReflectionResult] = []
        self._quality_trends: list[float] = []

    def attach(self, agent):
        """Attach reflection capability to an agent."""
        agent._reflection_engine = self

    async def reflect(
        self,
        task: str,
        result: str,
        model_router=None,
        rubric: QualityRubric | None = None,
        context: dict | None = None,
    ) -> ReflectionResult:
        """
        Run a reflection cycle on a task result.

        Uses a critic prompt to evaluate quality and provide feedback.
        """
        start_time = time.time()
        reflection = ReflectionResult()

        # Generate critique
        critique = await self._critique(task, result, model_router)
        reflection.feedback = critique.get("feedback", "")
        reflection.strengths = critique.get("strengths", [])
        reflection.weaknesses = critique.get("weaknesses", [])
        reflection.suggestions = critique.get("suggestions", [])

        # Score quality
        reflection.quality_score = self._score_quality(task, result, critique)

        # If below threshold, generate refined output
        if reflection.quality_score < self.config.reflection_quality_threshold:
            refined = await self._refine(task, result, reflection, model_router)
            if refined:
                reflection.refined_output = refined
                # Re-score
                new_score = self._score_quality(task, refined, critique)
                reflection.improvement_delta = new_score - reflection.quality_score
                reflection.quality_score = new_score

        reflection.duration_ms = (time.time() - start_time) * 1000
        reflection.cycles_used = 1

        self._reflection_history.append(reflection)
        self._quality_trends.append(reflection.quality_score)

        return reflection

    async def _critique(self, task: str, result: str, model_router=None) -> dict:
        """Generate a structured critique of the result."""
        critique_prompt = (
            "You are a quality evaluator. Critically review this output.\n\n"
            f"ORIGINAL TASK: {task}\n\n"
            f"OUTPUT TO REVIEW:\n{result}\n\n"
            "Evaluate and respond in this exact JSON format:\n"
            "{\n"
            '  "feedback": "Overall assessment in 2-3 sentences",\n'
            '  "strengths": ["strength1", "strength2"],\n'
            '  "weaknesses": ["weakness1", "weakness2"],\n'
            '  "suggestions": ["specific improvement 1", "specific improvement 2"],\n'
            '  "scores": {\n'
            '    "correctness": 0.0-1.0,\n'
            '    "completeness": 0.0-1.0,\n'
            '    "clarity": 0.0-1.0,\n'
            '    "relevance": 0.0-1.0\n'
            "  }\n"
            "}"
        )

        if model_router:
            response = await model_router.generate(
                model=self.config.default_model,
                messages=[{"role": "user", "content": critique_prompt}],
                temperature=0.2,  # Low temperature for analytical tasks
            )
            try:
                # Try to parse JSON from response
                content = response.get("content", "")
                # Find JSON in response
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    import json
                    return json.loads(content[start:end])
            except (json.JSONDecodeError, ValueError):
                pass

        return {
            "feedback": "Unable to generate critique",
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
            "scores": {},
        }

    async def _refine(
        self, task: str, result: str, reflection: ReflectionResult, model_router=None
    ) -> str | None:
        """Refine the output based on reflection feedback."""
        refine_prompt = (
            "You produced an output that needs improvement. "
            "Refine it based on the feedback below.\n\n"
            f"ORIGINAL TASK: {task}\n\n"
            f"YOUR PREVIOUS OUTPUT:\n{result}\n\n"
            f"FEEDBACK: {reflection.feedback}\n"
            f"WEAKNESSES: {', '.join(reflection.weaknesses)}\n"
            f"SUGGESTIONS: {', '.join(reflection.suggestions)}\n\n"
            "Produce an improved version that addresses all the feedback:"
        )

        if model_router:
            response = await model_router.generate(
                model=self.config.default_model,
                messages=[{"role": "user", "content": refine_prompt}],
                temperature=0.5,
            )
            return response.get("content", "")

        return None

    def _score_quality(self, task: str, result: str, critique: dict) -> float:
        """Score quality based on critique and heuristics."""
        # Use scores from critique if available
        scores = critique.get("scores", {})
        if scores:
            values = [v for v in scores.values() if isinstance(v, (int, float))]
            if values:
                return sum(values) / len(values)

        # Heuristic scoring
        score = 0.5  # Base score

        # Length check (not too short, not too long)
        result_len = len(result)
        if result_len > 100:
            score += 0.1
        if result_len > 500:
            score += 0.1

        # Task relevance (simple keyword overlap)
        task_words = set(task.lower().split())
        result_words = set(result.lower().split())
        overlap = len(task_words.intersection(result_words))
        if overlap > 0:
            score += min(0.2, overlap * 0.05)

        # Penalize error indicators
        error_indicators = ["error", "failed", "unable", "cannot", "sorry"]
        if any(ei in result.lower() for ei in error_indicators):
            score -= 0.2

        return max(0.0, min(1.0, score))

    async def meta_reflect(self) -> dict:
        """
        Meta-level reflection: Analyze reflection patterns over time.
        Returns insights about quality trends and systematic issues.
        """
        if not self._reflection_history:
            return {"insights": [], "avg_quality": 0.0}

        avg_quality = sum(r.quality_score for r in self._reflection_history) / len(self._reflection_history)

        # Find common weaknesses
        all_weaknesses: dict[str, int] = {}
        for r in self._reflection_history:
            for w in r.weaknesses:
                all_weaknesses[w] = all_weaknesses.get(w, 0) + 1

        common_weaknesses = sorted(all_weaknesses.items(), key=lambda x: x[1], reverse=True)[:5]

        # Quality trend
        recent_quality = [r.quality_score for r in self._reflection_history[-10:]]
        trend = "improving" if len(recent_quality) > 1 and recent_quality[-1] > recent_quality[0] else "stable"

        return {
            "avg_quality": avg_quality,
            "total_reflections": len(self._reflection_history),
            "common_weaknesses": common_weaknesses,
            "quality_trend": trend,
            "avg_improvement_delta": sum(r.improvement_delta for r in self._reflection_history) / len(self._reflection_history),
        }

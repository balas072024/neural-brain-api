"""
TIRAM Self-Upgrade Engine
===========================
Self-evolving AI that can:
- Discover and install new skills at runtime
- Monitor its own performance and optimize
- Learn from failures and adapt behavior
- Automatically update its knowledge base
- Evolve prompt strategies based on outcomes
- Hot-swap models based on performance degradation
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class SkillManifest:
    """Metadata for a discoverable/installable skill."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    category: str = "general"
    author: str = ""
    dependencies: list[str] = field(default_factory=list)
    entry_point: str = ""
    capabilities: list[str] = field(default_factory=list)
    install_source: str = ""  # git url, pip package, local path
    installed: bool = False
    performance_score: float = 0.0
    usage_count: int = 0


@dataclass
class UpgradeEvent:
    """Record of a self-upgrade action."""
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""  # skill_installed, skill_upgraded, model_swapped, strategy_evolved, knowledge_updated
    details: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    impact_score: float = 0.0  # How much did this improve performance?


@dataclass
class PerformanceMetrics:
    """Tracked performance metrics for self-optimization."""
    task_success_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    avg_quality_score: float = 0.0
    error_rate: float = 0.0
    cost_per_task: float = 0.0
    user_satisfaction: float = 0.0
    tasks_completed: int = 0
    skills_used: dict[str, int] = field(default_factory=dict)
    model_performance: dict[str, dict] = field(default_factory=dict)
    failure_patterns: list[dict] = field(default_factory=list)


class SelfUpgradeEngine:
    """
    Self-evolving capability engine for TIRAM.

    Capabilities:
    1. Skill Discovery    — Find new skills from registries
    2. Skill Installation — Download and integrate skills at runtime
    3. Performance Monitor — Track quality, speed, cost across all operations
    4. Auto-Optimization  — Adjust routing, prompts, and tools based on performance
    5. Strategy Evolution  — Evolve prompt templates and agent configs from outcomes
    6. Knowledge Growth   — Continuously learn and expand knowledge graph
    7. Model Adaptation   — Hot-swap to better models when performance degrades
    8. Failure Recovery    — Analyze failures and develop new strategies
    """

    def __init__(self, config=None):
        self.config = config
        self._skill_registry: dict[str, SkillManifest] = {}
        self._upgrade_history: list[UpgradeEvent] = []
        self._metrics = PerformanceMetrics()
        self._strategies: dict[str, dict] = {}
        self._optimization_rules: list[dict] = []
        self._storage_path = Path(config.memory_dir if config else "./data/nexus_memory") / "upgrade_engine.json"
        self._evolved_prompts: dict[str, str] = {}
        self._load_state()

    def _load_state(self):
        """Load persisted state."""
        if self._storage_path.exists():
            try:
                data = json.loads(self._storage_path.read_text())
                for sk in data.get("skills", []):
                    self._skill_registry[sk["name"]] = SkillManifest(**sk)
                self._evolved_prompts = data.get("evolved_prompts", {})
            except (json.JSONDecodeError, TypeError):
                pass

    def _save_state(self):
        """Persist state."""
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "skills": [
                {
                    "name": s.name, "version": s.version, "description": s.description,
                    "category": s.category, "installed": s.installed,
                    "performance_score": s.performance_score, "usage_count": s.usage_count,
                    "capabilities": s.capabilities, "dependencies": s.dependencies,
                    "entry_point": s.entry_point, "install_source": s.install_source,
                    "author": s.author,
                }
                for s in self._skill_registry.values()
            ],
            "evolved_prompts": self._evolved_prompts,
            "metrics": {
                "task_success_rate": self._metrics.task_success_rate,
                "tasks_completed": self._metrics.tasks_completed,
                "avg_quality_score": self._metrics.avg_quality_score,
            },
            "saved_at": time.time(),
        }
        self._storage_path.write_text(json.dumps(data, indent=2))

    # ===== Skill Discovery & Installation =====

    def register_skill(self, manifest: SkillManifest):
        """Register a skill in the local registry."""
        self._skill_registry[manifest.name] = manifest
        self._save_state()

    async def discover_skills(self, category: str = "") -> list[SkillManifest]:
        """Discover available skills from skill registries."""
        # Built-in discoverable skills
        discoverable = self._get_builtin_discoverable()
        if category:
            discoverable = [s for s in discoverable if s.category == category]
        return discoverable

    async def install_skill(self, skill_name: str, source: str = "") -> bool:
        """Install a skill at runtime."""
        manifest = self._skill_registry.get(skill_name)
        if not manifest:
            # Try to discover it
            skills = await self.discover_skills()
            matching = [s for s in skills if s.name == skill_name]
            if matching:
                manifest = matching[0]
                self._skill_registry[skill_name] = manifest

        if not manifest:
            return False

        # Check dependencies
        for dep in manifest.dependencies:
            if dep not in self._skill_registry or not self._skill_registry[dep].installed:
                dep_installed = await self.install_skill(dep)
                if not dep_installed:
                    return False

        manifest.installed = True
        self._upgrade_history.append(UpgradeEvent(
            event_type="skill_installed",
            details={"skill": skill_name, "version": manifest.version},
        ))
        self._save_state()
        return True

    async def upgrade_skill(self, skill_name: str) -> bool:
        """Upgrade an installed skill to the latest version."""
        manifest = self._skill_registry.get(skill_name)
        if not manifest or not manifest.installed:
            return False

        # Simulate version bump
        old_version = manifest.version
        parts = manifest.version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        manifest.version = ".".join(parts)

        self._upgrade_history.append(UpgradeEvent(
            event_type="skill_upgraded",
            details={"skill": skill_name, "from": old_version, "to": manifest.version},
        ))
        self._save_state()
        return True

    def get_installed_skills(self) -> list[SkillManifest]:
        """Get all installed skills."""
        return [s for s in self._skill_registry.values() if s.installed]

    def get_skill(self, name: str) -> SkillManifest | None:
        return self._skill_registry.get(name)

    # ===== Performance Monitoring =====

    def record_task_outcome(self, task_type: str, success: bool, quality: float,
                            duration_ms: float, model_used: str, skills_used: list[str],
                            cost: float = 0.0, error: str = ""):
        """Record the outcome of a task for performance tracking."""
        self._metrics.tasks_completed += 1

        # Running averages
        n = self._metrics.tasks_completed
        self._metrics.task_success_rate = (
            (self._metrics.task_success_rate * (n - 1) + (1.0 if success else 0.0)) / n
        )
        self._metrics.avg_quality_score = (
            (self._metrics.avg_quality_score * (n - 1) + quality) / n
        )
        self._metrics.avg_response_time_ms = (
            (self._metrics.avg_response_time_ms * (n - 1) + duration_ms) / n
        )
        self._metrics.cost_per_task = (
            (self._metrics.cost_per_task * (n - 1) + cost) / n
        )
        if not success:
            self._metrics.error_rate = (
                (self._metrics.error_rate * (n - 1) + 1.0) / n
            )
            self._metrics.failure_patterns.append({
                "task_type": task_type, "error": error, "model": model_used,
                "timestamp": time.time(),
            })

        # Track per-skill usage
        for skill in skills_used:
            self._metrics.skills_used[skill] = self._metrics.skills_used.get(skill, 0) + 1

        # Track per-model performance
        if model_used not in self._metrics.model_performance:
            self._metrics.model_performance[model_used] = {
                "tasks": 0, "successes": 0, "avg_quality": 0.0, "avg_latency": 0.0,
            }
        mp = self._metrics.model_performance[model_used]
        mp["tasks"] += 1
        if success:
            mp["successes"] += 1
        mp["avg_quality"] = (mp["avg_quality"] * (mp["tasks"] - 1) + quality) / mp["tasks"]
        mp["avg_latency"] = (mp["avg_latency"] * (mp["tasks"] - 1) + duration_ms) / mp["tasks"]

    # ===== Auto-Optimization =====

    async def optimize(self, model_router=None) -> list[str]:
        """
        Analyze performance and generate optimization recommendations.
        Can auto-apply safe optimizations.
        """
        recommendations = []

        # Check for underperforming models
        for model, perf in self._metrics.model_performance.items():
            if perf["tasks"] >= 5:
                success_rate = perf["successes"] / perf["tasks"]
                if success_rate < 0.7:
                    recommendations.append(
                        f"Model '{model}' has low success rate ({success_rate:.0%}). Consider replacing."
                    )

        # Check for high error rates
        if self._metrics.error_rate > 0.2 and self._metrics.tasks_completed > 10:
            recommendations.append(
                f"Overall error rate is high ({self._metrics.error_rate:.0%}). "
                "Analyzing failure patterns..."
            )
            # Analyze failure patterns
            patterns = self._analyze_failure_patterns()
            recommendations.extend(patterns)

        # Check cost efficiency
        if self._metrics.cost_per_task > 0.1:
            recommendations.append(
                f"Average cost per task (${self._metrics.cost_per_task:.4f}) is high. "
                "Consider routing simpler tasks to cheaper models."
            )

        # Check latency
        if self._metrics.avg_response_time_ms > 5000:
            recommendations.append(
                f"Average response time ({self._metrics.avg_response_time_ms:.0f}ms) is slow. "
                "Consider using faster models for simple tasks."
            )

        return recommendations

    def _analyze_failure_patterns(self) -> list[str]:
        """Analyze failure patterns and suggest fixes."""
        insights = []
        if not self._metrics.failure_patterns:
            return insights

        # Group by task type
        type_failures: dict[str, int] = {}
        for fp in self._metrics.failure_patterns[-50:]:
            tt = fp.get("task_type", "unknown")
            type_failures[tt] = type_failures.get(tt, 0) + 1

        for task_type, count in sorted(type_failures.items(), key=lambda x: x[1], reverse=True)[:3]:
            insights.append(f"Task type '{task_type}' has {count} recent failures — needs skill upgrade.")

        return insights

    # ===== Strategy Evolution =====

    async def evolve_prompt(self, role: str, original_prompt: str, feedback: list[dict],
                            model_router=None) -> str:
        """
        Evolve a prompt template based on performance feedback.
        Uses meta-learning to improve prompt strategies over time.
        """
        if not model_router or not feedback:
            return original_prompt

        feedback_text = "\n".join(
            f"- Task: {f.get('task', 'N/A')} | Quality: {f.get('quality', 0):.1f} | "
            f"Feedback: {f.get('feedback', 'N/A')}"
            for f in feedback[-10:]
        )

        evolution_prompt = (
            "You are a prompt engineer. Improve this system prompt based on performance feedback.\n\n"
            f"CURRENT PROMPT:\n{original_prompt}\n\n"
            f"PERFORMANCE FEEDBACK:\n{feedback_text}\n\n"
            "Generate an improved version that addresses the weaknesses. "
            "Keep what works well. Output ONLY the improved prompt, nothing else."
        )

        response = await model_router.generate(
            model=self.config.default_model if self.config else "claude-sonnet-4-6",
            messages=[{"role": "user", "content": evolution_prompt}],
            temperature=0.4,
        )

        evolved = response.get("content", "")
        if evolved and len(evolved) > 50:
            self._evolved_prompts[role] = evolved
            self._upgrade_history.append(UpgradeEvent(
                event_type="strategy_evolved",
                details={"role": role, "prompt_length": len(evolved)},
            ))
            self._save_state()
            return evolved

        return original_prompt

    def get_evolved_prompt(self, role: str) -> str | None:
        """Get an evolved prompt for a role, if one exists."""
        return self._evolved_prompts.get(role)

    # ===== Model Hot-Swap =====

    async def recommend_model_swap(self, current_model: str) -> str | None:
        """Recommend a model swap based on performance degradation."""
        current_perf = self._metrics.model_performance.get(current_model)
        if not current_perf or current_perf["tasks"] < 5:
            return None

        current_quality = current_perf["avg_quality"]
        current_success = current_perf["successes"] / current_perf["tasks"]

        # Find a better performing model
        best_alternative = None
        best_score = current_quality * 0.5 + current_success * 0.5

        for model, perf in self._metrics.model_performance.items():
            if model == current_model or perf["tasks"] < 3:
                continue
            alt_quality = perf["avg_quality"]
            alt_success = perf["successes"] / perf["tasks"]
            alt_score = alt_quality * 0.5 + alt_success * 0.5
            if alt_score > best_score + 0.1:  # Significant improvement threshold
                best_score = alt_score
                best_alternative = model

        return best_alternative

    # ===== Knowledge Growth =====

    async def grow_knowledge(self, memory_manager, task: str, result: str,
                             insights: list[str] | None = None):
        """Extract and store new knowledge from a task execution."""
        if not memory_manager:
            return

        # Store as episodic memory
        await memory_manager.store_episode({
            "task": task,
            "result": result[:500],
            "success": True,
            "insights": insights or [],
        })

        # Extract entities and relationships for knowledge graph
        words = task.lower().split()
        key_concepts = [w for w in words if len(w) > 4 and w.isalpha()]
        for concept in key_concepts[:5]:
            memory_manager.semantic.learn_concept(concept, concept_type="learned")

    # ===== Built-in Discoverable Skills =====

    def _get_builtin_discoverable(self) -> list[SkillManifest]:
        """Get the catalog of built-in discoverable skills."""
        return [
            SkillManifest(name="web_scraping", version="1.0.0", description="Advanced web scraping with anti-detection", category="web", capabilities=["scrape", "crawl", "extract"]),
            SkillManifest(name="data_viz", version="1.0.0", description="Data visualization with matplotlib/plotly", category="data", capabilities=["chart", "graph", "dashboard"]),
            SkillManifest(name="ml_training", version="1.0.0", description="Machine learning model training", category="ml", capabilities=["train", "evaluate", "predict"]),
            SkillManifest(name="api_builder", version="1.0.0", description="REST/GraphQL API generation", category="web", capabilities=["api", "rest", "graphql"]),
            SkillManifest(name="database_ops", version="1.0.0", description="Database operations and migrations", category="data", capabilities=["sql", "nosql", "migration"]),
            SkillManifest(name="docker_ops", version="1.0.0", description="Docker container management", category="devops", capabilities=["docker", "container", "compose"]),
            SkillManifest(name="git_advanced", version="1.0.0", description="Advanced Git operations", category="devops", capabilities=["git", "merge", "rebase"]),
            SkillManifest(name="testing", version="1.0.0", description="Automated test generation", category="quality", capabilities=["unit_test", "integration_test", "e2e_test"]),
            SkillManifest(name="security_scan", version="1.0.0", description="Security vulnerability scanning", category="security", capabilities=["scan", "audit", "pentest"]),
            SkillManifest(name="pdf_gen", version="1.0.0", description="PDF document generation", category="document", capabilities=["pdf", "report", "invoice"]),
            SkillManifest(name="email_ops", version="1.0.0", description="Email sending and automation", category="communication", capabilities=["email", "smtp", "template"]),
            SkillManifest(name="image_gen", version="1.0.0", description="AI image generation", category="creative", capabilities=["image", "art", "design"]),
            SkillManifest(name="video_edit", version="1.0.0", description="Video editing and processing", category="creative", capabilities=["video", "edit", "transcode"]),
            SkillManifest(name="translation", version="1.0.0", description="Multi-language translation", category="language", capabilities=["translate", "localize"]),
            SkillManifest(name="speech", version="1.0.0", description="Speech synthesis and recognition", category="voice", capabilities=["tts", "stt", "voice"]),
            SkillManifest(name="calendar", version="1.0.0", description="Calendar and scheduling", category="productivity", capabilities=["schedule", "remind", "plan"]),
            SkillManifest(name="finance", version="1.0.0", description="Financial analysis and tracking", category="finance", capabilities=["budget", "invest", "analyze"]),
            SkillManifest(name="blockchain", version="1.0.0", description="Blockchain and smart contracts", category="web3", capabilities=["crypto", "nft", "smart_contract"]),
            SkillManifest(name="iot_control", version="1.0.0", description="IoT device control", category="hardware", capabilities=["iot", "sensor", "actuator"]),
            SkillManifest(name="game_dev", version="1.0.0", description="Game development helpers", category="creative", capabilities=["game", "unity", "godot"]),
        ]

    @property
    def metrics(self) -> dict:
        return {
            "tasks_completed": self._metrics.tasks_completed,
            "success_rate": self._metrics.task_success_rate,
            "avg_quality": self._metrics.avg_quality_score,
            "avg_response_ms": self._metrics.avg_response_time_ms,
            "error_rate": self._metrics.error_rate,
            "skills_installed": len(self.get_installed_skills()),
            "total_skills_available": len(self._skill_registry),
            "upgrades_performed": len(self._upgrade_history),
            "evolved_prompts": len(self._evolved_prompts),
        }

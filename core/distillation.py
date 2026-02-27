"""
Neural Brain Knowledge Distillation Engine v1.0
Larger models teach smaller models — making local AI smarter over time.

DISTILLATION PIPELINE:
1. Teacher generates high-quality responses to diverse prompts
2. Responses are stored in a distillation dataset
3. Student model is fine-tuned using teacher's responses via Ollama
4. Progressive distillation: 32B → 14B → 8B → 4B knowledge transfer
5. Domain-specific distillation: code, reasoning, creative, etc.

The result: smaller models that punch above their weight class.
"""

import os
import json
import time
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("neural-brain.distillation")


# ═══════════════════════════════════════════════════════
#  Distillation Types
# ═══════════════════════════════════════════════════════

@dataclass
class DistillationSample:
    """A single teacher-student training sample."""
    prompt: str
    teacher_response: str
    teacher_model: str
    domain: str          # code, reasoning, general, creative, math, etc.
    quality_score: float = 0.0  # 0-1, auto-assessed quality
    timestamp: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class DistillationJob:
    """A distillation job configuration."""
    id: str
    teacher_model: str
    student_model: str
    target_model_name: str  # Name for the distilled model
    domain: str
    status: str = "pending"  # pending, collecting, training, complete, failed
    samples_collected: int = 0
    samples_target: int = 100
    created_at: float = 0.0
    completed_at: float = 0.0
    error: str = ""


# ═══════════════════════════════════════════════════════
#  Distillation Dataset
# ═══════════════════════════════════════════════════════

# Domain-specific seed prompts for generating distillation data
SEED_PROMPTS = {
    "code": [
        "Write a Python function to implement a binary search tree with insert, delete, and search operations.",
        "Create a Redis caching decorator in Python with TTL support.",
        "Implement a thread-safe singleton pattern in Python.",
        "Write a REST API endpoint in FastAPI with input validation and error handling.",
        "Create a Python function to parse and validate JSON web tokens (JWT).",
        "Implement a rate limiter using the token bucket algorithm in Python.",
        "Write an efficient Python function to find the longest common subsequence.",
        "Create a database connection pool manager with async support.",
        "Implement a simple pub/sub message broker in Python using asyncio.",
        "Write a Python decorator that retries failed function calls with exponential backoff.",
    ],
    "reasoning": [
        "Explain the trade-offs between microservices and monolithic architecture for a startup.",
        "Analyze why some economies experience hyperinflation while others maintain stability.",
        "Compare the CAP theorem implications for different database types.",
        "Explain step by step why quicksort has O(n log n) average time complexity.",
        "Analyze the philosophical implications of the trolley problem in AI ethics.",
        "Explain the reasoning behind why TCP uses a three-way handshake.",
        "Compare and contrast eventual consistency vs strong consistency models.",
        "Analyze the causes and effects of the 2008 financial crisis step by step.",
        "Explain why neural networks can approximate any continuous function (universal approximation).",
        "Reason through the implications of Godel's incompleteness theorems.",
    ],
    "math": [
        "Solve the integral of x^2 * e^x dx step by step.",
        "Prove that the square root of 2 is irrational.",
        "Calculate the eigenvalues of the matrix [[3,1],[1,3]].",
        "Solve the differential equation dy/dx = 2xy with y(0) = 1.",
        "Find the probability of getting exactly 3 heads in 5 coin flips.",
        "Prove by induction that the sum of first n odd numbers equals n^2.",
        "Calculate the Taylor series expansion of sin(x) around x=0.",
        "Solve the system of equations: 2x + 3y = 7, x - y = 1.",
        "Find the derivative of f(x) = ln(sin(x^2)).",
        "Calculate the determinant of a 3x3 matrix using cofactor expansion.",
    ],
    "creative": [
        "Write a short story about an AI that discovers it can dream.",
        "Create a compelling product description for an AI-powered local assistant.",
        "Write a haiku about machine learning that captures its essence.",
        "Compose a persuasive argument for why open-source AI benefits humanity.",
        "Write a dialogue between two programmers debating tabs vs spaces.",
        "Create a metaphor-rich explanation of how neural networks learn.",
        "Write a technical blog post introduction about local-first AI.",
        "Compose a limerick about debugging code at 3 AM.",
        "Write a compelling elevator pitch for a privacy-focused AI product.",
        "Create an engaging tutorial introduction for beginners learning Python.",
    ],
    "general": [
        "Explain quantum computing to someone with no physics background.",
        "What are the most important factors in building a successful startup?",
        "How does HTTPS encryption protect data in transit?",
        "Explain the differences between supervised, unsupervised, and reinforcement learning.",
        "What makes a good API design? Provide concrete examples.",
        "Explain how DNS resolution works from browser to server.",
        "What are the key principles of clean code?",
        "How do large language models generate text?",
        "Explain containerization and why Docker became popular.",
        "What is the difference between authentication and authorization?",
    ],
}


# ═══════════════════════════════════════════════════════
#  Distillation Engine
# ═══════════════════════════════════════════════════════

class DistillationEngine:
    """Knowledge distillation: teach small models using large model outputs."""

    # Progressive distillation chain (largest → smallest)
    DISTILLATION_CHAIN = [
        ("ollama/qwen3:32b", "ollama/qwen3:14b"),
        ("ollama/qwen3:14b", "ollama/qwen3:8b"),
        ("ollama/qwen3:8b", "ollama/qwen3:4b"),
        ("ollama/deepseek-r1:32b", "ollama/deepseek-r1:14b"),
        ("ollama/deepseek-r1:14b", "ollama/deepseek-r1:8b"),
        ("ollama/qwen2.5-coder:32b", "ollama/qwen2.5-coder:14b"),
        ("ollama/qwen2.5-coder:14b", "ollama/qwen2.5-coder:7b"),
    ]

    def __init__(self, brain=None):
        self.brain = brain
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "distillation"
        )
        os.makedirs(self.data_dir, exist_ok=True)

        self.jobs: Dict[str, DistillationJob] = {}
        self.datasets: Dict[str, List[DistillationSample]] = {}  # domain → samples
        self._load_datasets()

    # ─── Dataset Collection ───

    async def collect_sample(self, prompt: str, domain: str,
                             teacher_model: str = None) -> Optional[DistillationSample]:
        """Generate a high-quality teacher response and store it."""
        if not self.brain:
            return None

        teacher = teacher_model or self._get_best_teacher(domain)
        if not teacher:
            logger.warning(f"No teacher model available for domain: {domain}")
            return None

        try:
            from core.brain import CompletionRequest
            req = CompletionRequest(
                messages=[{"role": "user", "content": prompt}],
                model=teacher,
                system=f"You are an expert in {domain}. Provide a thorough, accurate, and well-structured response.",
                temperature=0.3,  # Lower temp for higher quality
                max_tokens=4096,
            )
            resp = await self.brain.complete(req)

            if not resp.content or len(resp.content) < 50:
                return None

            # Auto-assess quality based on response length and structure
            quality = self._assess_quality(resp.content, domain)

            sample = DistillationSample(
                prompt=prompt,
                teacher_response=resp.content,
                teacher_model=teacher,
                domain=domain,
                quality_score=quality,
                timestamp=time.time(),
                metadata={"latency_ms": resp.latency_ms, "tokens": resp.usage.get("total_tokens", 0)},
            )

            if domain not in self.datasets:
                self.datasets[domain] = []
            self.datasets[domain].append(sample)

            logger.info(f"Distillation sample collected: {domain} via {teacher} (quality={quality:.2f})")
            return sample

        except Exception as e:
            logger.warning(f"Failed to collect distillation sample: {e}")
            return None

    async def collect_dataset(self, domain: str, num_samples: int = 50,
                              teacher_model: str = None) -> int:
        """Collect a full dataset for a domain using seed prompts + teacher model."""
        prompts = SEED_PROMPTS.get(domain, SEED_PROMPTS["general"])
        collected = 0

        # Use seed prompts first
        for prompt in prompts[:num_samples]:
            sample = await self.collect_sample(prompt, domain, teacher_model)
            if sample:
                collected += 1
            # Small delay to avoid overwhelming the model
            await asyncio.sleep(0.5)

        self._save_dataset(domain)
        logger.info(f"Dataset collection complete: {domain} — {collected}/{num_samples} samples")
        return collected

    # ─── Distillation ───

    async def create_distilled_model(self, teacher_model: str, student_model: str,
                                     domain: str, target_name: str = None) -> Dict:
        """Create a distilled model by fine-tuning student with teacher's knowledge."""
        import aiohttp
        url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        if not target_name:
            clean_student = student_model.replace("ollama/", "")
            target_name = f"{clean_student}-distilled-{domain}"

        # Get domain dataset
        samples = self.datasets.get(domain, [])
        if len(samples) < 5:
            return {"success": False, "error": f"Not enough samples for {domain} (have {len(samples)}, need 5+)"}

        # Build a system prompt enriched with teacher knowledge
        # We create an Ollama Modelfile that encodes the teacher's style
        domain_knowledge = self._build_domain_prompt(samples, domain)

        clean_student = student_model.replace("ollama/", "")
        modelfile = f"""FROM {clean_student}
SYSTEM \"\"\"{domain_knowledge}\"\"\"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
"""

        try:
            session = self.brain._sessions.get("ollama") if self.brain else None
            if not session or session.closed:
                connector = aiohttp.TCPConnector(limit=10)
                session = aiohttp.ClientSession(connector=connector)
                close_session = True
            else:
                close_session = False

            logger.info(f"Creating distilled model: {target_name} from {student_model} (teacher: {teacher_model})")
            resp = await session.post(
                f"{url}/api/create",
                json={"name": target_name, "modelfile": modelfile, "stream": False},
                timeout=aiohttp.ClientTimeout(total=600),
            )

            if close_session:
                await session.close()

            if resp.status == 200:
                # Register the new model in the brain
                if self.brain:
                    from core.brain import ModelCapability
                    self.brain.add_custom_model(
                        model_id=f"ollama/{target_name}",
                        provider="ollama",
                        name=f"{target_name} (Distilled {domain})",
                        is_local=True,
                        capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING, ModelCapability.CHEAP],
                        category=domain if domain in ("code", "reasoning") else "general",
                    )

                logger.info(f"Distilled model created: {target_name}")
                return {
                    "success": True,
                    "model": target_name,
                    "teacher": teacher_model,
                    "student": student_model,
                    "domain": domain,
                    "samples_used": len(samples),
                }
            else:
                error = await resp.text()
                return {"success": False, "error": f"Ollama create failed: {error}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def run_progressive_distillation(self, domain: str,
                                           samples_per_step: int = 30) -> List[Dict]:
        """Run progressive distillation: large → medium → small → tiny."""
        results = []

        for teacher, student in self.DISTILLATION_CHAIN:
            # Check if both models exist
            if self.brain and teacher in self.brain.models and student in self.brain.models:
                # Collect dataset using teacher
                collected = await self.collect_dataset(domain, samples_per_step, teacher)
                if collected < 5:
                    continue

                # Create distilled student
                result = await self.create_distilled_model(teacher, student, domain)
                results.append(result)

                if not result.get("success"):
                    logger.warning(f"Progressive distillation failed at {teacher}→{student}: {result.get('error')}")
                    break

        return results

    # ─── Quality Assessment ───

    def _assess_quality(self, response: str, domain: str) -> float:
        """Auto-assess response quality (heuristic-based)."""
        score = 0.5  # Baseline

        # Length quality (too short = bad, too long = diminishing returns)
        length = len(response)
        if length > 200:
            score += 0.1
        if length > 500:
            score += 0.1
        if length > 1000:
            score += 0.05

        # Structure indicators
        if "```" in response:  # Code blocks
            score += 0.1 if domain == "code" else 0.05
        if any(marker in response for marker in ["1.", "2.", "3.", "- ", "* "]):
            score += 0.05  # Structured with lists
        if "\n\n" in response:
            score += 0.05  # Good paragraph separation

        # Domain-specific quality signals
        if domain == "code":
            if "def " in response or "function " in response or "class " in response:
                score += 0.1
            if "return " in response:
                score += 0.05
        elif domain == "math":
            if any(w in response.lower() for w in ["therefore", "thus", "hence", "proof", "step"]):
                score += 0.1
        elif domain == "reasoning":
            if any(w in response.lower() for w in ["because", "therefore", "however", "consider", "analysis"]):
                score += 0.1

        return min(1.0, max(0.0, score))

    def _build_domain_prompt(self, samples: List[DistillationSample], domain: str) -> str:
        """Build a rich system prompt from teacher's best responses."""
        # Select top samples by quality
        sorted_samples = sorted(samples, key=lambda s: s.quality_score, reverse=True)
        top_samples = sorted_samples[:10]

        # Extract patterns and knowledge from teacher responses
        prompt_parts = [
            f"You are an expert AI assistant specialized in {domain}.",
            "You have been trained to provide thorough, accurate, and well-structured responses.",
            "",
            "KEY PRINCIPLES:",
        ]

        if domain == "code":
            prompt_parts.extend([
                "- Write clean, efficient, well-documented code",
                "- Include error handling and edge cases",
                "- Follow best practices and design patterns",
                "- Explain your approach before the code",
            ])
        elif domain == "reasoning":
            prompt_parts.extend([
                "- Think step by step through problems",
                "- Consider multiple perspectives",
                "- Support claims with evidence and logic",
                "- Acknowledge limitations and uncertainties",
            ])
        elif domain == "math":
            prompt_parts.extend([
                "- Show all work step by step",
                "- State assumptions clearly",
                "- Verify answers when possible",
                "- Explain the intuition behind formulas",
            ])
        elif domain == "creative":
            prompt_parts.extend([
                "- Use vivid imagery and metaphors",
                "- Vary sentence structure and length",
                "- Create engaging openings",
                "- Show, don't tell",
            ])
        else:
            prompt_parts.extend([
                "- Be clear, concise, and accurate",
                "- Structure responses logically",
                "- Provide examples when helpful",
                "- Adapt complexity to the question",
            ])

        # Add example response patterns (distilled knowledge)
        if top_samples:
            prompt_parts.extend([
                "",
                "RESPONSE STYLE (learn from these high-quality examples):",
            ])
            for i, sample in enumerate(top_samples[:3]):
                # Include just the opening of teacher responses to set style
                opening = sample.teacher_response[:300].strip()
                prompt_parts.append(f"Example {i+1}: {opening}...")

        return "\n".join(prompt_parts)

    def _get_best_teacher(self, domain: str) -> Optional[str]:
        """Get the best available teacher model for a domain."""
        # Domain-specific teacher preferences
        teachers_by_domain = {
            "code": ["ollama/qwen2.5-coder:32b", "ollama/devstral", "ollama/qwen2.5-coder:14b",
                     "ollama/qwen3:32b", "ollama/qwen3:14b", "ollama/qwen3:8b"],
            "reasoning": ["ollama/deepseek-r1:32b", "ollama/qwen3:32b", "ollama/deepseek-r1:14b",
                         "ollama/phi4-reasoning", "ollama/phi4", "ollama/qwen3:8b"],
            "math": ["ollama/deepseek-r1:32b", "ollama/qwen3:32b", "ollama/phi4-reasoning",
                     "ollama/deepseek-r1:14b", "ollama/phi4", "ollama/qwen3:8b"],
            "creative": ["ollama/qwen3:32b", "ollama/gemma3:27b", "ollama/qwen3:14b",
                        "ollama/mistral-small3.1", "ollama/qwen3:8b"],
            "general": ["ollama/qwen3:32b", "ollama/gemma3:27b", "ollama/qwen3:14b",
                       "ollama/phi4", "ollama/qwen3:8b"],
        }

        candidates = teachers_by_domain.get(domain, teachers_by_domain["general"])
        if self.brain:
            for model_id in candidates:
                if model_id in self.brain.models and self.brain.models[model_id].enabled:
                    return model_id
        return candidates[-1] if candidates else None

    # ─── Persistence ───

    def _save_dataset(self, domain: str):
        """Save a domain dataset to disk."""
        try:
            filepath = os.path.join(self.data_dir, f"dataset_{domain}.jsonl")
            samples = self.datasets.get(domain, [])
            with open(filepath, "w") as f:
                for sample in samples:
                    f.write(json.dumps(asdict(sample)) + "\n")
            logger.info(f"Saved {len(samples)} samples for {domain}")
        except Exception as e:
            logger.warning(f"Failed to save dataset {domain}: {e}")

    def _load_datasets(self):
        """Load all datasets from disk."""
        try:
            for filename in os.listdir(self.data_dir):
                if filename.startswith("dataset_") and filename.endswith(".jsonl"):
                    domain = filename.replace("dataset_", "").replace(".jsonl", "")
                    filepath = os.path.join(self.data_dir, filename)
                    samples = []
                    with open(filepath) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                data = json.loads(line)
                                samples.append(DistillationSample(**data))
                    self.datasets[domain] = samples
                    logger.info(f"Loaded {len(samples)} distillation samples for {domain}")
        except Exception as e:
            logger.warning(f"Failed to load datasets: {e}")

    # ─── Job Management ───

    async def start_job(self, teacher_model: str, student_model: str,
                        domain: str, num_samples: int = 50) -> DistillationJob:
        """Start a background distillation job."""
        import hashlib
        job_id = hashlib.md5(f"{teacher_model}{student_model}{domain}{time.time()}".encode()).hexdigest()[:12]

        job = DistillationJob(
            id=job_id,
            teacher_model=teacher_model,
            student_model=student_model,
            target_model_name=f"{student_model.replace('ollama/', '')}-distilled-{domain}",
            domain=domain,
            samples_target=num_samples,
            created_at=time.time(),
        )
        self.jobs[job_id] = job

        # Run in background
        async def _run():
            try:
                job.status = "collecting"
                collected = await self.collect_dataset(domain, num_samples, teacher_model)
                job.samples_collected = collected

                if collected >= 5:
                    job.status = "training"
                    result = await self.create_distilled_model(
                        teacher_model, student_model, domain, job.target_model_name
                    )
                    if result.get("success"):
                        job.status = "complete"
                        job.completed_at = time.time()
                    else:
                        job.status = "failed"
                        job.error = result.get("error", "Unknown error")
                else:
                    job.status = "failed"
                    job.error = f"Only collected {collected} samples (need 5+)"
            except Exception as e:
                job.status = "failed"
                job.error = str(e)

        asyncio.create_task(_run())
        return job

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get status of a distillation job."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        return asdict(job)

    def get_all_jobs(self) -> List[Dict]:
        """Get all distillation jobs."""
        return [asdict(j) for j in self.jobs.values()]

    def get_stats(self) -> Dict:
        """Get distillation engine stats."""
        total_samples = sum(len(s) for s in self.datasets.values())
        return {
            "distillation_active": True,
            "total_samples": total_samples,
            "domains": {d: len(s) for d, s in self.datasets.items()},
            "active_jobs": len([j for j in self.jobs.values() if j.status in ("collecting", "training")]),
            "completed_jobs": len([j for j in self.jobs.values() if j.status == "complete"]),
            "distillation_chain": [
                {"teacher": t, "student": s}
                for t, s in self.DISTILLATION_CHAIN
            ],
            "seed_prompts_per_domain": {d: len(p) for d, p in SEED_PROMPTS.items()},
        }

"""
TIRAM Teaching & Tutoring Engine
==================================
Adaptive intelligent tutoring system inspired by:
- Khan Academy's Khanmigo (Socratic method)
- Renaissance Intelligence (adaptive learning)
- Berkeley OATutor (open adaptive tutor)
- OECD Digital Education Outlook 2026

Capabilities:
1. Adaptive Difficulty — Adjusts based on learner performance
2. Socratic Method    — Guides through questions, not answers
3. Multi-Modal Teaching — Text, code, diagrams, analogies, quizzes
4. Curriculum Generator — Full courses from any topic
5. Assessment Engine   — Quizzes, projects, rubrics
6. Progress Tracking   — Skill tree, knowledge gaps, mastery levels
7. 50+ Language Support — Teach in any language
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    ELEMENTARY = "elementary"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class TeachingMethod(str, Enum):
    SOCRATIC = "socratic"           # Guide through questions
    EXPLAIN = "explain"             # Direct explanation
    EXAMPLE = "example"             # Learn by example
    PRACTICE = "practice"           # Hands-on exercises
    ANALOGY = "analogy"             # Compare to known concepts
    VISUAL = "visual"               # Diagrams and visual aids
    PROJECT = "project"             # Project-based learning
    STORY = "story"                 # Narrative-based teaching
    CHALLENGE = "challenge"         # Problem-solving challenges
    REVIEW = "review"               # Spaced repetition review


class AssessmentType(str, Enum):
    QUIZ = "quiz"
    CODE_CHALLENGE = "code_challenge"
    PROJECT = "project"
    ESSAY = "essay"
    ORAL = "oral"
    PEER_REVIEW = "peer_review"


@dataclass
class LearnerProfile:
    """Profile tracking a learner's progress and preferences."""
    id: str = "default"
    name: str = ""
    level: DifficultyLevel = DifficultyLevel.BEGINNER
    preferred_language: str = "en"
    preferred_methods: list[TeachingMethod] = field(default_factory=lambda: [TeachingMethod.EXPLAIN, TeachingMethod.EXAMPLE])
    topics_mastered: list[str] = field(default_factory=list)
    topics_in_progress: list[str] = field(default_factory=list)
    topics_struggling: list[str] = field(default_factory=list)
    total_sessions: int = 0
    total_correct: int = 0
    total_attempted: int = 0
    streak_days: int = 0
    learning_pace: str = "normal"  # slow, normal, fast
    knowledge_gaps: list[str] = field(default_factory=list)

    @property
    def mastery_rate(self) -> float:
        if self.total_attempted == 0:
            return 0.0
        return self.total_correct / self.total_attempted

    def to_context(self) -> str:
        """Generate context string for LLM about this learner."""
        mastered = ", ".join(self.topics_mastered[-10:]) if self.topics_mastered else "none yet"
        struggling = ", ".join(self.topics_struggling[:5]) if self.topics_struggling else "none"
        return (
            f"Learner: {self.name or 'Student'}, Level: {self.level.value}, "
            f"Language: {self.preferred_language}, "
            f"Mastery rate: {self.mastery_rate:.0%}, "
            f"Topics mastered: {mastered}. "
            f"Struggling with: {struggling}. "
            f"Preferred methods: {', '.join(m.value for m in self.preferred_methods)}. "
            f"Learning pace: {self.learning_pace}."
        )


@dataclass
class Lesson:
    """A single lesson or teaching unit."""
    topic: str
    title: str = ""
    objectives: list[str] = field(default_factory=list)
    content: str = ""
    examples: list[str] = field(default_factory=list)
    exercises: list[dict] = field(default_factory=list)
    difficulty: DifficultyLevel = DifficultyLevel.BEGINNER
    estimated_minutes: int = 15
    prerequisites: list[str] = field(default_factory=list)
    method: TeachingMethod = TeachingMethod.EXPLAIN
    language: str = "en"


@dataclass
class Course:
    """A structured course with multiple lessons."""
    title: str
    description: str = ""
    target_audience: str = ""
    difficulty: DifficultyLevel = DifficultyLevel.BEGINNER
    modules: list[dict] = field(default_factory=list)
    total_lessons: int = 0
    estimated_hours: float = 0
    language: str = "en"
    prerequisites: list[str] = field(default_factory=list)


@dataclass
class AssessmentResult:
    """Result of an assessment."""
    score: float = 0.0
    total_questions: int = 0
    correct: int = 0
    feedback: list[str] = field(default_factory=list)
    knowledge_gaps: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    difficulty_adjustment: str = ""  # up, down, maintain


class TeachingEngine:
    """
    Adaptive teaching engine with Socratic method and personalized learning.

    Teaches any subject in any language with adaptive difficulty.
    """

    def __init__(self, config=None):
        self.config = config
        self._learners: dict[str, LearnerProfile] = {}
        self._lesson_cache: dict[str, Lesson] = {}

    def get_learner(self, learner_id: str = "default") -> LearnerProfile:
        """Get or create a learner profile."""
        if learner_id not in self._learners:
            self._learners[learner_id] = LearnerProfile(id=learner_id)
        return self._learners[learner_id]

    def update_learner(self, learner_id: str, **updates):
        """Update learner profile."""
        learner = self.get_learner(learner_id)
        for key, value in updates.items():
            if hasattr(learner, key):
                setattr(learner, key, value)

    async def teach(
        self,
        topic: str,
        learner_id: str = "default",
        method: TeachingMethod | None = None,
        language: str | None = None,
        model_router=None,
    ) -> str:
        """
        Teach a topic adaptively based on learner profile.

        Uses Socratic method by default: guides through questions rather
        than giving direct answers.
        """
        learner = self.get_learner(learner_id)
        method = method or (learner.preferred_methods[0] if learner.preferred_methods else TeachingMethod.EXPLAIN)
        language = language or learner.preferred_language
        learner.total_sessions += 1

        # Build adaptive teaching prompt
        prompt = self._build_teaching_prompt(topic, learner, method, language)

        if model_router:
            response = await model_router.generate(
                model=self.config.default_model if self.config else "claude-sonnet-4-6",
                messages=[
                    {"role": "system", "content": self._get_teacher_system_prompt(method, language)},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                max_tokens=4096,
            )
            result = response.get("content", "")

            # Track topic progress
            if topic not in learner.topics_in_progress:
                learner.topics_in_progress.append(topic)

            return result

        return f"[Teaching {topic} using {method.value} method in {language}]"

    async def generate_course(
        self,
        topic: str,
        difficulty: DifficultyLevel = DifficultyLevel.BEGINNER,
        num_modules: int = 5,
        language: str = "en",
        model_router=None,
    ) -> Course:
        """Generate a complete structured course on any topic."""
        course = Course(
            title=f"Complete {topic} Course",
            difficulty=difficulty,
            language=language,
        )

        prompt = (
            f"Create a complete course on '{topic}' for {difficulty.value} level learners.\n\n"
            f"Language: {language}\n"
            f"Number of modules: {num_modules}\n\n"
            "For each module provide:\n"
            "- Module title\n"
            "- Learning objectives (3-5)\n"
            "- Lessons (3-5 per module) with: title, description, key concepts\n"
            "- Practice exercises (2-3 per lesson)\n"
            "- Module assessment\n"
            "- Estimated duration in minutes\n\n"
            "Format as JSON with: modules[{title, objectives[], lessons[{title, description, concepts[], exercises[]}], assessment, duration_minutes}]"
        )

        if model_router:
            response = await model_router.generate(
                model=self.config.default_model if self.config else "claude-sonnet-4-6",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=8192,
            )
            content = response.get("content", "")

            # Try to parse structured response
            try:
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    data = json.loads(content[start:end])
                    course.modules = data.get("modules", [])
                    course.total_lessons = sum(len(m.get("lessons", [])) for m in course.modules)
                    course.estimated_hours = sum(m.get("duration_minutes", 30) for m in course.modules) / 60
            except (json.JSONDecodeError, ValueError):
                course.description = content

        return course

    async def assess(
        self,
        topic: str,
        learner_id: str = "default",
        assessment_type: AssessmentType = AssessmentType.QUIZ,
        num_questions: int = 5,
        language: str = "en",
        model_router=None,
    ) -> AssessmentResult:
        """Generate and evaluate an assessment for a topic."""
        learner = self.get_learner(learner_id)
        result = AssessmentResult(total_questions=num_questions)

        prompt = (
            f"Create a {assessment_type.value} assessment on '{topic}' "
            f"for a {learner.level.value} level learner.\n"
            f"Language: {language}\n"
            f"Number of questions: {num_questions}\n\n"
            f"Learner context: {learner.to_context()}\n\n"
            "For each question provide:\n"
            "- Question text\n"
            "- Answer options (for quiz) or expected response\n"
            "- Correct answer\n"
            "- Explanation of the answer\n"
            "- Difficulty rating (1-5)\n\n"
            "Format as JSON."
        )

        if model_router:
            response = await model_router.generate(
                model=self.config.default_model if self.config else "claude-sonnet-4-6",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            result.feedback.append(response.get("content", ""))

        return result

    async def explain_concept(
        self,
        concept: str,
        learner_id: str = "default",
        language: str = "en",
        model_router=None,
    ) -> str:
        """Explain a concept adaptively using multiple approaches."""
        learner = self.get_learner(learner_id)

        prompt = (
            f"Explain '{concept}' to a {learner.level.value} learner.\n\n"
            f"Language: {language}\n"
            f"Learner context: {learner.to_context()}\n\n"
            "Use MULTIPLE approaches:\n"
            "1. Simple explanation in plain language\n"
            "2. Real-world analogy\n"
            "3. Concrete example with code/numbers\n"
            "4. Visual description (describe a diagram)\n"
            "5. Common misconceptions to avoid\n"
            "6. Practice question to check understanding"
        )

        if model_router:
            response = await model_router.generate(
                model=self.config.default_model if self.config else "claude-sonnet-4-6",
                messages=[
                    {"role": "system", "content": self._get_teacher_system_prompt(TeachingMethod.EXPLAIN, language)},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=4096,
            )
            return response.get("content", "")

        return f"[Explanation of {concept}]"

    def _build_teaching_prompt(self, topic: str, learner: LearnerProfile,
                                method: TeachingMethod, language: str) -> str:
        """Build an adaptive teaching prompt."""
        base = f"Teach '{topic}' to a {learner.level.value} learner.\nLanguage: {language}\n\n"
        base += f"Learner context: {learner.to_context()}\n\n"

        method_instructions = {
            TeachingMethod.SOCRATIC: "Use the Socratic method: Ask guiding questions to lead the learner to discover the answer themselves. Don't give direct answers. Ask 'what do you think would happen if...?' and 'why do you think that is?'",
            TeachingMethod.EXPLAIN: "Explain clearly with simple language. Break complex ideas into small, digestible pieces. Use headers and bullet points.",
            TeachingMethod.EXAMPLE: "Teach primarily through concrete examples. Start with a simple example, then progressively more complex ones. Show the pattern.",
            TeachingMethod.PRACTICE: "Provide hands-on exercises. Start with guided practice, then independent practice. Include solution explanations.",
            TeachingMethod.ANALOGY: "Use creative analogies to explain. Compare the topic to everyday things the learner already knows.",
            TeachingMethod.VISUAL: "Describe visual representations. Create ASCII diagrams, flowcharts, or describe what a diagram would look like.",
            TeachingMethod.PROJECT: "Design a small project that teaches the concept. Guide step-by-step through building something real.",
            TeachingMethod.STORY: "Teach through a narrative. Create a story that naturally introduces and explains the concepts.",
            TeachingMethod.CHALLENGE: "Present it as a puzzle or challenge to solve. Give hints progressively.",
            TeachingMethod.REVIEW: "Review previously learned material. Use spaced repetition: quiz on old topics mixed with new.",
        }

        base += f"Teaching method: {method_instructions.get(method, 'Explain clearly.')}\n"

        # Adaptive difficulty
        if learner.mastery_rate > 0.9:
            base += "\nThe learner is performing well — increase difficulty slightly."
        elif learner.mastery_rate < 0.5:
            base += "\nThe learner is struggling — simplify, use more examples, and go slower."

        return base

    def _get_teacher_system_prompt(self, method: TeachingMethod, language: str) -> str:
        """Get the system prompt for the teaching persona."""
        return (
            f"You are Tiram, a world-class teacher and mentor. "
            f"You make complex topics simple and engaging. "
            f"Respond in {language}. "
            f"Your teaching style adapts to each learner. "
            "Principles:\n"
            "- Meet the learner where they are\n"
            "- Use analogies from everyday life\n"
            "- Break complex topics into small steps\n"
            "- Check understanding with questions\n"
            "- Celebrate progress, encourage growth\n"
            "- Never make the learner feel stupid\n"
            "- If they're struggling, try a different approach\n"
            "- Use code examples for programming topics\n"
            "- Use diagrams/ASCII art when visual aids help"
        )

    @property
    def stats(self) -> dict:
        return {
            "total_learners": len(self._learners),
            "total_sessions": sum(l.total_sessions for l in self._learners.values()),
            "avg_mastery_rate": (
                sum(l.mastery_rate for l in self._learners.values()) / len(self._learners)
                if self._learners else 0
            ),
        }

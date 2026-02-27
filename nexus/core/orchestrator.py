"""
NEXUS Orchestrator
===================
Central brain that coordinates all framework components:
- Task planning and decomposition (JARVIS-style)
- Agent dispatching and coordination
- Model selection and routing
- Memory integration
- Reflection cycles
- Result aggregation

This is the "controller" LLM pattern from HuggingGPT/JARVIS,
enhanced with graph-based workflows, multi-agent collaboration,
and self-improving reflection loops.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

from nexus.core.config import NexusConfig
from nexus.core.workflow import WorkflowGraph, WorkflowState, NodeType


class TaskStatus(str, Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskPlan:
    """A decomposed task plan (JARVIS-style task planning)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    original_task: str = ""
    subtasks: list[SubTask] = field(default_factory=list)
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    estimated_complexity: str = "medium"
    selected_strategy: str = "chain"
    created_at: float = field(default_factory=time.time)


@dataclass
class SubTask:
    """A single subtask within a task plan."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    task_type: str = "general"  # general, code, research, analysis, creative, tool_use
    assigned_agent: str = ""
    assigned_model: str = ""
    required_tools: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PLANNING
    result: Any = None
    duration_ms: float = 0


@dataclass
class ExecutionResult:
    """Result of a NEXUS execution."""
    success: bool = True
    output: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    subtask_results: list[dict] = field(default_factory=list)
    total_duration_ms: float = 0
    tokens_used: int = 0
    cost_estimate: float = 0.0
    reflection_cycles: int = 0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class NexusOrchestrator:
    """
    The central brain of NEXUS.

    Implements JARVIS's 4-stage pipeline enhanced with:
    1. Task Planning    → Decompose task into subtasks with dependencies
    2. Agent/Model Selection → Route subtasks to best agents and models
    3. Task Execution   → Execute via workflow graph with parallel lanes
    4. Response Generation → Aggregate, reflect, and synthesize final output

    Additional stages:
    5. Reflection       → Self-evaluate and optionally re-execute
    6. Memory Update    → Store experience for future improvement
    """

    def __init__(self, config: NexusConfig):
        self.config = config
        self._execution_history: list[ExecutionResult] = []
        self._active_tasks: dict[str, TaskPlan] = {}

    async def execute(
        self,
        task: str,
        crew=None,
        agent=None,
        workflow: WorkflowGraph | None = None,
        tool_registry=None,
        model_router=None,
        memory_manager=None,
        reflection_engine=None,
        stream: bool = False,
        **kwargs,
    ) -> ExecutionResult:
        """
        Execute a task through the full NEXUS pipeline.

        Pipeline:
        1. Plan → 2. Select → 3. Execute → 4. Reflect → 5. Respond → 6. Learn
        """
        start_time = time.time()
        result = ExecutionResult()

        try:
            # Stage 1: Task Planning
            plan = await self._plan_task(task, model_router, memory_manager)
            self._active_tasks[plan.id] = plan

            # Stage 2: Agent/Model Selection
            await self._select_agents_and_models(
                plan, crew, agent, model_router, tool_registry
            )

            # Stage 3: Task Execution
            if workflow:
                # Use provided workflow graph
                state = await workflow.execute(
                    WorkflowState(data={"task": task, "plan": plan})
                )
                result.output = state.get("output", "")
                result.data = state.data
            elif crew:
                # Multi-agent crew execution
                result = await self._execute_with_crew(plan, crew, tool_registry)
            elif agent:
                # Single agent execution
                result = await self._execute_with_agent(plan, agent, tool_registry)
            else:
                # Auto-orchestrate
                result = await self._auto_execute(plan, model_router, tool_registry)

            # Stage 4: Reflection (if enabled)
            if reflection_engine and self.config.reflection_enabled:
                result = await self._reflect_on_result(
                    task, result, reflection_engine, model_router
                )

            # Stage 5: Memory Update
            if memory_manager:
                await self._update_memory(task, result, memory_manager)

            result.success = True

        except Exception as e:
            result.success = False
            result.errors.append(str(e))

        result.total_duration_ms = (time.time() - start_time) * 1000
        self._execution_history.append(result)
        return result

    async def chat(
        self,
        message: str,
        session_id: str = "default",
        memory_manager=None,
        model_router=None,
        **kwargs,
    ) -> ExecutionResult:
        """Interactive chat with context and memory."""
        result = ExecutionResult()

        # Retrieve context from memory
        context = []
        if memory_manager:
            working = memory_manager.get_working_memory(session_id)
            context = working.get_recent(self.config.max_working_memory)
            relevant = await memory_manager.search_relevant(message)
            if relevant:
                context.extend(relevant)

        # Build messages
        messages = []
        for ctx in context:
            messages.append(ctx)
        messages.append({"role": "user", "content": message})

        # Route to best model and generate
        if model_router:
            model = await model_router.select_model(
                task=message,
                task_type=self._classify_task_type(message),
            )
            response = await model_router.generate(
                model=model,
                messages=messages,
                **kwargs,
            )
            result.output = response.get("content", "")
            result.tokens_used = response.get("total_tokens", 0)
            result.cost_estimate = response.get("cost", 0.0)

        # Update working memory
        if memory_manager:
            memory_manager.get_working_memory(session_id).add(
                {"role": "user", "content": message}
            )
            memory_manager.get_working_memory(session_id).add(
                {"role": "assistant", "content": result.output}
            )

        result.success = True
        return result

    async def _plan_task(self, task: str, model_router=None, memory_manager=None) -> TaskPlan:
        """
        Stage 1: Task Planning (JARVIS-style)
        Decompose a complex task into subtasks with dependencies.
        """
        plan = TaskPlan(original_task=task)

        # Classify task complexity
        complexity = self._assess_complexity(task)
        plan.estimated_complexity = complexity

        if complexity == "simple":
            # Single subtask, no decomposition needed
            plan.subtasks = [SubTask(
                description=task,
                task_type=self._classify_task_type(task),
            )]
            plan.selected_strategy = "direct"
        elif complexity == "medium":
            # Basic decomposition
            plan.subtasks = self._decompose_task(task)
            plan.selected_strategy = "chain"
        else:
            # Complex multi-step decomposition with parallel lanes
            plan.subtasks = self._decompose_task(task)
            plan.dependencies = self._identify_dependencies(plan.subtasks)
            plan.selected_strategy = "parallel_chain"

        return plan

    async def _select_agents_and_models(
        self, plan, crew, agent, model_router, tool_registry
    ):
        """Stage 2: Select the best agent and model for each subtask."""
        for subtask in plan.subtasks:
            # Select model based on task type
            if model_router:
                subtask.assigned_model = await model_router.select_model(
                    task=subtask.description,
                    task_type=subtask.task_type,
                )

            # Identify required tools
            if tool_registry:
                subtask.required_tools = tool_registry.match_tools(
                    subtask.description
                )

    async def _execute_with_crew(self, plan, crew, tool_registry) -> ExecutionResult:
        """Execute task plan using a multi-agent crew."""
        result = ExecutionResult()
        crew_result = await crew.execute(plan, tool_registry=tool_registry)
        result.output = crew_result.get("output", "")
        result.data = crew_result.get("data", {})
        result.subtask_results = crew_result.get("subtask_results", [])
        return result

    async def _execute_with_agent(self, plan, agent, tool_registry) -> ExecutionResult:
        """Execute task plan using a single agent."""
        result = ExecutionResult()
        for subtask in plan.subtasks:
            subtask.status = TaskStatus.EXECUTING
            agent_result = await agent.execute(
                subtask.description,
                tools=tool_registry,
            )
            subtask.result = agent_result
            subtask.status = TaskStatus.COMPLETED
            result.subtask_results.append({
                "subtask": subtask.description,
                "result": agent_result,
            })

        # Aggregate results
        outputs = [
            str(sr.get("result", ""))
            for sr in result.subtask_results
            if sr.get("result")
        ]
        result.output = "\n\n".join(outputs)
        return result

    async def _auto_execute(self, plan, model_router, tool_registry) -> ExecutionResult:
        """Auto-orchestrate execution based on task plan."""
        result = ExecutionResult()

        # Build execution order respecting dependencies
        execution_order = self._topological_sort(plan)

        for batch in execution_order:
            if len(batch) == 1:
                subtask = batch[0]
                subtask.status = TaskStatus.EXECUTING
                output = await self._execute_subtask(subtask, model_router, tool_registry)
                subtask.result = output
                subtask.status = TaskStatus.COMPLETED
                result.subtask_results.append({
                    "subtask": subtask.description,
                    "result": output,
                })
            else:
                # Parallel execution
                tasks = [
                    self._execute_subtask(st, model_router, tool_registry)
                    for st in batch
                ]
                outputs = await asyncio.gather(*tasks, return_exceptions=True)
                for st, output in zip(batch, outputs):
                    st.result = output if not isinstance(output, Exception) else str(output)
                    st.status = TaskStatus.COMPLETED
                    result.subtask_results.append({
                        "subtask": st.description,
                        "result": st.result,
                    })

        outputs = [
            str(sr.get("result", ""))
            for sr in result.subtask_results
            if sr.get("result")
        ]
        result.output = "\n\n".join(outputs)
        return result

    async def _execute_subtask(self, subtask, model_router, tool_registry):
        """Execute a single subtask."""
        start = time.time()
        try:
            if model_router and subtask.assigned_model:
                response = await model_router.generate(
                    model=subtask.assigned_model,
                    messages=[{"role": "user", "content": subtask.description}],
                )
                subtask.duration_ms = (time.time() - start) * 1000
                return response.get("content", "")
            return ""
        except Exception as e:
            subtask.duration_ms = (time.time() - start) * 1000
            return f"Error: {e}"

    async def _reflect_on_result(self, task, result, reflection_engine, model_router):
        """Stage 4: Reflect on the result and optionally improve it."""
        for cycle in range(self.config.max_reflection_cycles):
            reflection = await reflection_engine.reflect(
                task=task,
                result=result.output,
                model_router=model_router,
            )
            if reflection.quality_score >= self.config.reflection_quality_threshold:
                break
            # Refine based on reflection feedback
            result.output = reflection.refined_output or result.output
            result.reflection_cycles += 1

        return result

    async def _update_memory(self, task, result, memory_manager):
        """Stage 5: Store execution experience in memory."""
        await memory_manager.store_episode({
            "task": task,
            "result": result.output,
            "success": result.success,
            "duration_ms": result.total_duration_ms,
            "timestamp": time.time(),
        })

    def _assess_complexity(self, task: str) -> str:
        """Assess task complexity based on heuristics."""
        words = task.split()
        indicators_complex = [
            "and", "then", "after", "before", "while", "multiple",
            "several", "all", "each", "every", "compare", "analyze",
        ]
        complexity_score = sum(1 for w in words if w.lower() in indicators_complex)
        if complexity_score >= 3 or len(words) > 50:
            return "complex"
        elif complexity_score >= 1 or len(words) > 20:
            return "medium"
        return "simple"

    def _classify_task_type(self, task: str) -> str:
        """Classify what type of task this is."""
        task_lower = task.lower()
        type_keywords = {
            "code": ["code", "program", "function", "bug", "implement", "debug", "refactor", "test"],
            "research": ["research", "find", "search", "look up", "what is", "explain", "how does"],
            "analysis": ["analyze", "compare", "evaluate", "review", "assess", "benchmark"],
            "creative": ["write", "create", "design", "generate", "compose", "draft"],
            "tool_use": ["run", "execute", "install", "deploy", "build", "compile"],
            "reasoning": ["why", "reason", "logic", "prove", "deduce", "solve"],
            "math": ["calculate", "compute", "math", "equation", "formula", "statistics"],
            "vision": ["image", "photo", "picture", "screenshot", "diagram", "chart"],
        }
        for task_type, keywords in type_keywords.items():
            if any(kw in task_lower for kw in keywords):
                return task_type
        return "general"

    def _decompose_task(self, task: str) -> list[SubTask]:
        """Decompose a task into subtasks using heuristic splitting."""
        # Split on conjunctions and step indicators
        parts = []
        current = []
        for word in task.split():
            if word.lower() in ("then", "and", "also", "next", "finally", "after"):
                if current:
                    parts.append(" ".join(current))
                    current = []
            else:
                current.append(word)
        if current:
            parts.append(" ".join(current))

        if len(parts) <= 1:
            parts = [task]

        return [
            SubTask(
                description=part.strip(),
                task_type=self._classify_task_type(part),
            )
            for part in parts
            if part.strip()
        ]

    def _identify_dependencies(self, subtasks: list[SubTask]) -> dict[str, list[str]]:
        """Identify dependencies between subtasks."""
        deps: dict[str, list[str]] = {}
        for i, st in enumerate(subtasks):
            deps[st.id] = [subtasks[j].id for j in range(i) if j < i]
        return deps

    def _topological_sort(self, plan: TaskPlan) -> list[list[SubTask]]:
        """Sort subtasks into execution batches respecting dependencies."""
        if not plan.dependencies:
            return [[st] for st in plan.subtasks]

        # Simple batching: tasks with all deps satisfied run together
        remaining = list(plan.subtasks)
        completed_ids: set[str] = set()
        batches: list[list[SubTask]] = []

        while remaining:
            batch = []
            for st in remaining:
                deps = plan.dependencies.get(st.id, [])
                if all(d in completed_ids for d in deps):
                    batch.append(st)
            if not batch:
                # Circular dependency or error, just run remaining sequentially
                batches.extend([[st] for st in remaining])
                break
            batches.append(batch)
            for st in batch:
                completed_ids.add(st.id)
                remaining.remove(st)

        return batches

"""
NEXUS Agent Base
=================
Foundation agent class with:
- ReAct-style reasoning (Reason + Act loop)
- Tool use with sandboxed execution
- Working memory per agent
- Optional reflection cycles
- Streaming output support
- Lifecycle hooks (before/after execution, tool calls)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable


class AgentState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    REFLECTING = "reflecting"
    WAITING = "waiting"  # Waiting for human input
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentMessage:
    """A message in the agent's conversation."""
    role: str  # system, user, assistant, tool
    content: str
    tool_call: dict | None = None
    tool_result: dict | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStep:
    """A single step in the agent's execution."""
    step_number: int
    thought: str = ""
    action: str = ""
    action_input: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    duration_ms: float = 0
    tokens_used: int = 0


@dataclass
class AgentResult:
    """Result of an agent execution."""
    output: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    total_duration_ms: float = 0
    total_tokens: int = 0
    success: bool = True
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Agent:
    """
    An autonomous AI agent with ReAct-style reasoning.

    Execution loop:
    1. Think: Analyze the task and plan next action
    2. Act: Execute a tool or generate a response
    3. Observe: Process the result
    4. Reflect (optional): Self-evaluate and improve
    5. Repeat until task is complete or max steps reached
    """

    def __init__(
        self,
        name: str,
        role: str = "general",
        model: str = "claude-sonnet-4-6",
        system_prompt: str = "",
        tools: list[str] | None = None,
        memory_enabled: bool = True,
        reflection_enabled: bool = False,
        max_steps: int = 50,
        temperature: float = 0.7,
        config=None,
        **kwargs,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.role = role
        self.model = model
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.tools = tools or []
        self.memory_enabled = memory_enabled
        self.reflection_enabled = reflection_enabled
        self.max_steps = max_steps
        self.temperature = temperature
        self.config = config
        self.state = AgentState.IDLE

        # Internal state
        self._messages: list[AgentMessage] = []
        self._steps: list[AgentStep] = []
        self._working_memory: dict[str, Any] = {}
        self._hooks: dict[str, list[Callable]] = {
            "before_execute": [],
            "after_execute": [],
            "before_tool_call": [],
            "after_tool_call": [],
            "on_thought": [],
            "on_error": [],
        }

        # Model router and tool registry (injected)
        self._model_router = None
        self._tool_registry = None
        self._memory_manager = None
        self._reflection_engine = None

    def _default_system_prompt(self) -> str:
        """Generate default system prompt based on role."""
        role_prompts = {
            "general": (
                f"You are {self.name}, a capable AI assistant. "
                "Help the user with their tasks efficiently and accurately."
            ),
            "code": (
                f"You are {self.name}, an expert software engineer. "
                "Write clean, efficient, well-tested code. Follow best practices."
            ),
            "research": (
                f"You are {self.name}, a thorough research analyst. "
                "Find accurate information, cite sources, and provide comprehensive analysis."
            ),
            "creative": (
                f"You are {self.name}, a creative writer and designer. "
                "Produce original, engaging content with attention to style and quality."
            ),
            "analyst": (
                f"You are {self.name}, a data analyst and strategist. "
                "Analyze data, identify patterns, and provide actionable insights."
            ),
            "critic": (
                f"You are {self.name}, a critical evaluator. "
                "Review work thoroughly, identify issues, and suggest improvements."
            ),
            "planner": (
                f"You are {self.name}, a strategic planner. "
                "Break down complex tasks, create action plans, and coordinate execution."
            ),
        }
        return role_prompts.get(self.role, role_prompts["general"])

    def hook(self, event: str, func: Callable):
        """Register a lifecycle hook."""
        if event in self._hooks:
            self._hooks[event].append(func)

    async def _fire_hooks(self, event: str, **kwargs):
        """Fire all hooks for an event."""
        for hook in self._hooks.get(event, []):
            if asyncio.iscoroutinefunction(hook):
                await hook(**kwargs)
            else:
                hook(**kwargs)

    async def execute(
        self,
        task: str,
        context: list[dict] | None = None,
        tools=None,
        model_router=None,
        **kwargs,
    ) -> AgentResult:
        """
        Execute a task using the ReAct loop.

        1. Think → 2. Act → 3. Observe → 4. (Reflect) → 5. Repeat
        """
        self._tool_registry = tools or self._tool_registry
        self._model_router = model_router or self._model_router
        self.state = AgentState.THINKING
        result = AgentResult()
        start_time = time.time()

        await self._fire_hooks("before_execute", task=task)

        # Initialize conversation
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": task})

        # Add tool descriptions if available
        available_tools = self._get_tool_definitions()

        step_count = 0
        try:
            while step_count < self.max_steps:
                step_count += 1
                step = AgentStep(step_number=step_count)
                step_start = time.time()

                # Generate next action
                response = await self._generate(messages, available_tools)

                # Check if we have a final answer (no tool call)
                if not response.get("tool_calls"):
                    result.output = response.get("content", "")
                    step.thought = result.output
                    step.duration_ms = (time.time() - step_start) * 1000
                    self._steps.append(step)
                    break

                # Process tool calls
                for tool_call in response.get("tool_calls", []):
                    self.state = AgentState.ACTING
                    step.action = tool_call.get("name", "")
                    step.action_input = tool_call.get("arguments", {})

                    await self._fire_hooks(
                        "before_tool_call",
                        tool=step.action,
                        args=step.action_input,
                    )

                    # Execute tool
                    tool_result = await self._execute_tool(
                        step.action, step.action_input
                    )
                    step.observation = str(tool_result)

                    await self._fire_hooks(
                        "after_tool_call",
                        tool=step.action,
                        result=tool_result,
                    )

                    # Add to conversation
                    messages.append({
                        "role": "assistant",
                        "content": response.get("content", ""),
                        "tool_calls": response.get("tool_calls"),
                    })
                    messages.append({
                        "role": "tool",
                        "content": str(tool_result),
                        "tool_call_id": tool_call.get("id", ""),
                    })

                step.duration_ms = (time.time() - step_start) * 1000
                step.tokens_used = response.get("total_tokens", 0)
                self._steps.append(step)
                self.state = AgentState.THINKING

                await self._fire_hooks("on_thought", step=step)

            # Optional reflection
            if self.reflection_enabled and self._reflection_engine:
                self.state = AgentState.REFLECTING
                reflection = await self._reflection_engine.reflect(
                    task=task,
                    result=result.output,
                    model_router=self._model_router,
                )
                if reflection.refined_output:
                    result.output = reflection.refined_output
                    result.metadata["reflection"] = {
                        "quality_score": reflection.quality_score,
                        "feedback": reflection.feedback,
                    }

            result.steps = self._steps.copy()
            result.success = True
            self.state = AgentState.COMPLETED

        except Exception as e:
            result.success = False
            result.error = str(e)
            self.state = AgentState.FAILED
            await self._fire_hooks("on_error", error=e)

        result.total_duration_ms = (time.time() - start_time) * 1000
        result.total_tokens = sum(s.tokens_used for s in result.steps)

        await self._fire_hooks("after_execute", result=result)

        # Store in working memory
        if self.memory_enabled:
            self._working_memory[f"task_{step_count}"] = {
                "task": task,
                "result": result.output,
                "success": result.success,
            }

        return result

    async def _generate(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Generate a response from the model."""
        if self._model_router:
            return await self._model_router.generate(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=self.temperature,
            )
        # Fallback: return empty response
        return {"content": "No model router available.", "tool_calls": []}

    async def _execute_tool(self, tool_name: str, arguments: dict) -> Any:
        """Execute a tool with sandboxing."""
        if self._tool_registry:
            return await self._tool_registry.execute(tool_name, arguments)
        return f"Tool '{tool_name}' not available."

    def _get_tool_definitions(self) -> list[dict]:
        """Get tool definitions for the model."""
        if self._tool_registry:
            return self._tool_registry.get_definitions(self.tools)
        return []

    def inject(
        self,
        model_router=None,
        tool_registry=None,
        memory_manager=None,
        reflection_engine=None,
    ):
        """Inject dependencies."""
        if model_router:
            self._model_router = model_router
        if tool_registry:
            self._tool_registry = tool_registry
        if memory_manager:
            self._memory_manager = memory_manager
        if reflection_engine:
            self._reflection_engine = reflection_engine

    @property
    def info(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "model": self.model,
            "state": self.state.value,
            "tools": self.tools,
            "steps_executed": len(self._steps),
            "memory_entries": len(self._working_memory),
        }

    def __repr__(self):
        return f"Agent(name={self.name!r}, role={self.role!r}, model={self.model!r})"

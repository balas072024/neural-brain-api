"""
NEXUS Multi-Agent Collaboration
=================================
Inspired by CrewAI + AutoGen, with additional patterns:

Strategies:
  - chain:     Sequential pipeline (A → B → C)
  - consensus: All agents answer, best is selected
  - debate:    Agents argue, judge decides
  - parallel:  Independent parallel execution, results merged
  - hierarchy: Manager delegates to specialists
  - swarm:     Dynamic hand-off between agents based on capability
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nexus.agents.base import Agent, AgentResult
from nexus.core.config import NexusConfig


class CollaborationStrategy(str, Enum):
    CHAIN = "chain"
    CONSENSUS = "consensus"
    DEBATE = "debate"
    PARALLEL = "parallel"
    HIERARCHY = "hierarchy"
    SWARM = "swarm"


@dataclass
class CrewResult:
    """Result from a multi-agent crew execution."""
    output: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    agent_results: dict[str, AgentResult] = field(default_factory=dict)
    subtask_results: list[dict] = field(default_factory=list)
    strategy_used: str = ""
    total_duration_ms: float = 0
    total_tokens: int = 0


class Crew:
    """
    A team of agents working together with a collaboration strategy.

    Patterns:
    - Chain: Each agent's output feeds into the next
    - Consensus: All agents work independently, judge picks best
    - Debate: Agents present, critique, and refine iteratively
    - Parallel: All agents work independently on different subtasks
    - Hierarchy: Manager agent delegates to specialists
    - Swarm: Dynamic routing based on agent capabilities
    """

    def __init__(
        self,
        name: str,
        agents: list[Agent],
        strategy: CollaborationStrategy = CollaborationStrategy.CHAIN,
        manager_model: str = "claude-sonnet-4-6",
        config: NexusConfig | None = None,
        max_debate_rounds: int = 3,
    ):
        self.name = name
        self.agents = agents
        self.strategy = strategy
        self.manager_model = manager_model
        self.config = config
        self.max_debate_rounds = max_debate_rounds

    async def execute(self, plan=None, task: str = "", tool_registry=None, **kwargs) -> dict:
        """Execute the crew's collaborative task."""
        task_text = plan.original_task if plan else task
        start_time = time.time()
        result = CrewResult(strategy_used=self.strategy.value)

        strategy_handlers = {
            CollaborationStrategy.CHAIN: self._execute_chain,
            CollaborationStrategy.CONSENSUS: self._execute_consensus,
            CollaborationStrategy.DEBATE: self._execute_debate,
            CollaborationStrategy.PARALLEL: self._execute_parallel,
            CollaborationStrategy.HIERARCHY: self._execute_hierarchy,
            CollaborationStrategy.SWARM: self._execute_swarm,
        }

        handler = strategy_handlers.get(self.strategy, self._execute_chain)
        result = await handler(task_text, plan, tool_registry, **kwargs)

        result.total_duration_ms = (time.time() - start_time) * 1000
        return {
            "output": result.output,
            "data": result.data,
            "subtask_results": result.subtask_results,
            "strategy": result.strategy_used,
            "duration_ms": result.total_duration_ms,
        }

    async def _execute_chain(self, task, plan, tool_registry, **kwargs) -> CrewResult:
        """Chain: Sequential pipeline where each agent builds on the previous."""
        result = CrewResult(strategy_used="chain")
        current_input = task
        context = []

        for agent in self.agents:
            prompt = (
                f"Previous context:\n{current_input}\n\n"
                f"Your role as {agent.role}: Complete your part of this task."
                if context else task
            )
            agent_result = await agent.execute(
                prompt, context=context, tools=tool_registry
            )
            result.agent_results[agent.name] = agent_result
            result.subtask_results.append({
                "agent": agent.name,
                "role": agent.role,
                "result": agent_result.output,
            })
            current_input = agent_result.output
            context.append({"role": "assistant", "content": agent_result.output})

        result.output = current_input
        return result

    async def _execute_consensus(self, task, plan, tool_registry, **kwargs) -> CrewResult:
        """Consensus: All agents work independently, best response selected."""
        result = CrewResult(strategy_used="consensus")

        # All agents work in parallel
        tasks = [
            agent.execute(task, tools=tool_registry)
            for agent in self.agents
        ]
        agent_results = await asyncio.gather(*tasks, return_exceptions=True)

        responses = []
        for agent, ar in zip(self.agents, agent_results):
            if isinstance(ar, AgentResult) and ar.success:
                result.agent_results[agent.name] = ar
                responses.append({
                    "agent": agent.name,
                    "role": agent.role,
                    "response": ar.output,
                })

        # Select best response (simple: pick longest non-error response)
        if responses:
            best = max(responses, key=lambda r: len(r["response"]))
            result.output = best["response"]
            result.data["selected_agent"] = best["agent"]

        result.subtask_results = responses
        return result

    async def _execute_debate(self, task, plan, tool_registry, **kwargs) -> CrewResult:
        """Debate: Agents present positions, critique each other, and refine."""
        result = CrewResult(strategy_used="debate")

        # Round 1: Initial positions
        positions = {}
        tasks = [
            agent.execute(f"Present your position on: {task}", tools=tool_registry)
            for agent in self.agents
        ]
        initial_results = await asyncio.gather(*tasks, return_exceptions=True)

        for agent, ar in zip(self.agents, initial_results):
            if isinstance(ar, AgentResult) and ar.success:
                positions[agent.name] = ar.output

        # Debate rounds
        for round_num in range(self.max_debate_rounds):
            all_positions = "\n\n".join(
                f"[{name}]: {pos}" for name, pos in positions.items()
            )
            critique_prompt = (
                f"Original task: {task}\n\n"
                f"Current positions from all agents:\n{all_positions}\n\n"
                f"Critique the other positions and refine your own. "
                f"This is debate round {round_num + 1}/{self.max_debate_rounds}."
            )

            critique_tasks = [
                agent.execute(critique_prompt, tools=tool_registry)
                for agent in self.agents
            ]
            critique_results = await asyncio.gather(*critique_tasks, return_exceptions=True)

            for agent, cr in zip(self.agents, critique_results):
                if isinstance(cr, AgentResult) and cr.success:
                    positions[agent.name] = cr.output

        # Final synthesis
        all_final = "\n\n".join(
            f"[{name}]: {pos}" for name, pos in positions.items()
        )
        result.output = f"Debate synthesis:\n\n{all_final}"
        result.data["debate_rounds"] = self.max_debate_rounds
        result.subtask_results = [
            {"agent": name, "final_position": pos}
            for name, pos in positions.items()
        ]
        return result

    async def _execute_parallel(self, task, plan, tool_registry, **kwargs) -> CrewResult:
        """Parallel: Agents work on different subtasks independently."""
        result = CrewResult(strategy_used="parallel")

        subtasks = plan.subtasks if plan else [type("ST", (), {"description": task, "id": "0"})]

        # Distribute subtasks across agents (round-robin)
        assignments = {}
        for i, subtask in enumerate(subtasks):
            agent = self.agents[i % len(self.agents)]
            if agent.name not in assignments:
                assignments[agent.name] = []
            assignments[agent.name].append(subtask)

        # Execute in parallel
        async def execute_agent_tasks(agent, agent_subtasks):
            results = []
            for st in agent_subtasks:
                ar = await agent.execute(st.description, tools=tool_registry)
                results.append({"subtask": st.description, "result": ar.output})
            return agent.name, results

        parallel_tasks = [
            execute_agent_tasks(
                next(a for a in self.agents if a.name == name), tasks_list
            )
            for name, tasks_list in assignments.items()
        ]
        parallel_results = await asyncio.gather(*parallel_tasks, return_exceptions=True)

        outputs = []
        for pr in parallel_results:
            if isinstance(pr, tuple):
                agent_name, task_results = pr
                result.subtask_results.extend(task_results)
                outputs.extend(r["result"] for r in task_results)

        result.output = "\n\n".join(outputs)
        return result

    async def _execute_hierarchy(self, task, plan, tool_registry, **kwargs) -> CrewResult:
        """Hierarchy: Manager agent delegates to specialists."""
        result = CrewResult(strategy_used="hierarchy")

        # First agent is the manager
        manager = self.agents[0]
        specialists = self.agents[1:]

        specialist_info = "\n".join(
            f"- {s.name} ({s.role}): {s.system_prompt[:100]}..."
            for s in specialists
        )

        manager_prompt = (
            f"You are the project manager. Your team:\n{specialist_info}\n\n"
            f"Task: {task}\n\n"
            f"Create a delegation plan. For each team member, specify their subtask."
        )

        manager_result = await manager.execute(manager_prompt, tools=tool_registry)
        result.agent_results[manager.name] = manager_result

        # Execute specialist tasks in parallel
        specialist_tasks = [
            specialist.execute(
                f"Manager's direction: {manager_result.output}\n\nYour specific task as {specialist.role}:",
                tools=tool_registry,
            )
            for specialist in specialists
        ]
        specialist_results = await asyncio.gather(*specialist_tasks, return_exceptions=True)

        outputs = [manager_result.output]
        for specialist, sr in zip(specialists, specialist_results):
            if isinstance(sr, AgentResult) and sr.success:
                result.agent_results[specialist.name] = sr
                outputs.append(f"\n[{specialist.name}]: {sr.output}")
                result.subtask_results.append({
                    "agent": specialist.name,
                    "result": sr.output,
                })

        result.output = "\n".join(outputs)
        return result

    async def _execute_swarm(self, task, plan, tool_registry, **kwargs) -> CrewResult:
        """Swarm: Dynamic hand-off between agents based on capability matching."""
        result = CrewResult(strategy_used="swarm")

        current_task = task
        used_agents = set()
        max_handoffs = len(self.agents) * 2

        for _ in range(max_handoffs):
            # Find best agent for current task
            best_agent = self._match_agent(current_task, used_agents)
            if not best_agent:
                break

            agent_result = await best_agent.execute(current_task, tools=tool_registry)
            result.agent_results[best_agent.name] = agent_result
            result.subtask_results.append({
                "agent": best_agent.name,
                "result": agent_result.output,
            })
            used_agents.add(best_agent.name)

            # Check if task is complete or needs handoff
            if agent_result.success and "HANDOFF:" not in agent_result.output:
                result.output = agent_result.output
                break

            # Extract handoff task
            if "HANDOFF:" in agent_result.output:
                parts = agent_result.output.split("HANDOFF:", 1)
                result.output = parts[0].strip()
                current_task = parts[1].strip()
            else:
                result.output = agent_result.output
                break

        return result

    def _match_agent(self, task: str, exclude: set[str]) -> Agent | None:
        """Find the best agent for a task based on role capabilities."""
        task_lower = task.lower()
        role_keywords = {
            "code": ["code", "program", "function", "bug", "implement"],
            "research": ["research", "find", "search", "what is", "explain"],
            "creative": ["write", "create", "design", "compose"],
            "critic": ["review", "evaluate", "check", "audit"],
            "analyst": ["analyze", "data", "statistics", "metrics"],
        }

        for agent in self.agents:
            if agent.name in exclude:
                continue
            keywords = role_keywords.get(agent.role, [])
            if any(kw in task_lower for kw in keywords):
                return agent

        # Fallback: return first available agent
        for agent in self.agents:
            if agent.name not in exclude:
                return agent
        return None

    @property
    def info(self) -> dict:
        return {
            "name": self.name,
            "strategy": self.strategy.value,
            "agents": [a.info for a in self.agents],
        }

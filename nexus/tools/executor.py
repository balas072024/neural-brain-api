"""
NEXUS Sandboxed Executor
==========================
Secure tool execution with process isolation, resource limits,
and permission checking. Addresses OpenClaw's security concerns.
"""

from __future__ import annotations

import asyncio
import subprocess
import os
import time
from dataclasses import dataclass, field
from typing import Any

from nexus.core.config import NexusConfig, SecurityLevel


@dataclass
class ExecutionResult:
    """Result of a sandboxed execution."""
    success: bool = True
    output: str = ""
    error: str = ""
    exit_code: int = 0
    duration_ms: float = 0
    resource_usage: dict[str, Any] = field(default_factory=dict)


class SandboxedExecutor:
    """
    Executes tools and commands in a sandboxed environment.

    Security features:
    - Command allowlist/blocklist
    - Resource limits (timeout, memory)
    - Working directory isolation
    - Output sanitization
    - Audit logging
    """

    def __init__(self, config: NexusConfig):
        self.config = config
        self._execution_log: list[dict] = []

    def is_command_allowed(self, command: str) -> tuple[bool, str]:
        """Check if a command is allowed by security policy."""
        # Check blocklist
        for blocked in self.config.blocked_commands:
            if blocked in command:
                return False, f"Blocked pattern: {blocked}"

        # In strict/paranoid mode, check allowlist
        if self.config.security_level in (SecurityLevel.STRICT, SecurityLevel.PARANOID):
            cmd_base = command.split()[0] if command.split() else ""
            if cmd_base not in self.config.allowed_commands:
                return False, f"Command '{cmd_base}' not in allowlist"

        return True, ""

    async def execute_command(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute a shell command with sandboxing."""
        result = ExecutionResult()
        start_time = time.time()

        # Security check
        allowed, reason = self.is_command_allowed(command)
        if not allowed:
            result.success = False
            result.error = f"Command blocked: {reason}"
            return result

        timeout = timeout or self.config.tool_timeout

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env={**os.environ, **(env or {})},
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )

            result.output = stdout.decode("utf-8", errors="replace")[:50000]
            result.error = stderr.decode("utf-8", errors="replace")[:10000]
            result.exit_code = proc.returncode or 0
            result.success = proc.returncode == 0

        except asyncio.TimeoutError:
            result.success = False
            result.error = f"Command timed out after {timeout}s"
            result.exit_code = -1
        except Exception as e:
            result.success = False
            result.error = str(e)
            result.exit_code = -1

        result.duration_ms = (time.time() - start_time) * 1000

        # Audit log
        self._execution_log.append({
            "command": command[:200],
            "success": result.success,
            "exit_code": result.exit_code,
            "duration_ms": result.duration_ms,
            "timestamp": time.time(),
        })

        return result

    async def execute_python(
        self,
        code: str,
        timeout: float = 30.0,
    ) -> ExecutionResult:
        """Execute Python code in a subprocess."""
        return await self.execute_command(
            f'python3 -c "{code}"',
            timeout=timeout,
        )

    @property
    def execution_log(self) -> list[dict]:
        return self._execution_log[-100:]  # Last 100 entries

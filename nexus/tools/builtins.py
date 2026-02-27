"""
NEXUS Built-in Tools
=====================
Core tools available to all agents out of the box.
Inspired by OpenClaw's built-in capabilities.
"""

from __future__ import annotations

import os
import json
import asyncio
import aiohttp
from pathlib import Path
from typing import Any

from nexus.tools.registry import ToolRegistry


def register_builtin_tools(registry: ToolRegistry):
    """Register all built-in tools with the registry."""

    # === File System Tools ===

    async def file_read(path: str) -> str:
        """Read the contents of a file."""
        try:
            p = Path(path).expanduser()
            if not p.exists():
                return f"File not found: {path}"
            if p.stat().st_size > 1_000_000:
                return f"File too large (>{1_000_000} bytes): {path}"
            return p.read_text(errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"

    async def file_write(path: str, content: str) -> str:
        """Write content to a file."""
        try:
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return f"Written {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    async def file_list(path: str) -> str:
        """List files in a directory."""
        try:
            p = Path(path).expanduser()
            if not p.is_dir():
                return f"Not a directory: {path}"
            entries = []
            for item in sorted(p.iterdir()):
                prefix = "📁" if item.is_dir() else "📄"
                size = item.stat().st_size if item.is_file() else 0
                entries.append(f"{prefix} {item.name} ({size} bytes)")
            return "\n".join(entries) if entries else "Empty directory"
        except Exception as e:
            return f"Error listing directory: {e}"

    async def file_search(pattern: str, path: str = ".") -> str:
        """Search for files matching a glob pattern."""
        try:
            p = Path(path).expanduser()
            matches = list(p.rglob(pattern))[:50]
            return "\n".join(str(m) for m in matches) if matches else "No matches found"
        except Exception as e:
            return f"Error searching: {e}"

    # === Web Tools ===

    async def web_search(query: str) -> str:
        """Search the web for information (requires API key)."""
        return f"Web search for: {query} (configure search API for results)"

    async def web_fetch(url: str) -> str:
        """Fetch content from a URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        return f"HTTP {resp.status}: {resp.reason}"
                    text = await resp.text()
                    return text[:10000]
        except Exception as e:
            return f"Error fetching URL: {e}"

    # === Shell Tools ===

    async def shell_exec(command: str) -> str:
        """Execute a shell command."""
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            output = stdout.decode("utf-8", errors="replace")
            errors = stderr.decode("utf-8", errors="replace")
            return f"{output}\n{errors}".strip()[:10000]
        except asyncio.TimeoutError:
            return "Command timed out after 60s"
        except Exception as e:
            return f"Error: {e}"

    # === Code Tools ===

    async def code_search(pattern: str, path: str = ".") -> str:
        """Search for a pattern in code files using grep-like matching."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "grep", "-rn", "--include=*.py", "--include=*.js", "--include=*.ts",
                "--include=*.go", "--include=*.rs", "--include=*.java",
                pattern, path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            results = stdout.decode("utf-8", errors="replace")
            return results[:10000] if results else "No matches found"
        except Exception as e:
            return f"Error searching: {e}"

    async def python_exec(code: str) -> str:
        """Execute Python code and return the output."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode("utf-8", errors="replace")
            errors = stderr.decode("utf-8", errors="replace")
            return f"{output}\n{errors}".strip()[:10000]
        except Exception as e:
            return f"Error: {e}"

    # === JSON/Data Tools ===

    async def json_parse(text: str) -> str:
        """Parse and pretty-print JSON."""
        try:
            data = json.loads(text)
            return json.dumps(data, indent=2)
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"

    # Register all tools
    tools = [
        ("file_read", file_read, "Read a file's contents", ["filesystem"]),
        ("file_write", file_write, "Write content to a file", ["filesystem"]),
        ("file_list", file_list, "List files in a directory", ["filesystem"]),
        ("file_search", file_search, "Search for files by glob pattern", ["filesystem"]),
        ("web_search", web_search, "Search the web for information", ["web"]),
        ("web_fetch", web_fetch, "Fetch content from a URL", ["web"]),
        ("shell", shell_exec, "Execute a shell command", ["system"]),
        ("code_search", code_search, "Search for patterns in code files", ["code"]),
        ("python_exec", python_exec, "Execute Python code", ["code"]),
        ("json_parse", json_parse, "Parse and pretty-print JSON", ["data"]),
    ]

    for name, func, desc, tags in tools:
        registry.register(name, func, description=desc, tags=tags)

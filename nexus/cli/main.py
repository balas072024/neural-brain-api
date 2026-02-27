"""
NEXUS CLI
==========
Command-line interface for the NEXUS framework.

Usage:
    python -m nexus.cli.main serve       # Start API server
    python -m nexus.cli.main chat        # Interactive chat
    python -m nexus.cli.main run "task"  # Execute a task
    python -m nexus.cli.main agents      # List agents
    python -m nexus.cli.main status      # Show framework status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="nexus",
        description="NEXUS AI Framework — Next-gen autonomous agent system",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8200, help="Port to bind to")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # chat
    chat_parser = subparsers.add_parser("chat", help="Interactive chat session")
    chat_parser.add_argument("--model", default=None, help="Model to use")
    chat_parser.add_argument("--agent", default=None, help="Named agent to use")
    chat_parser.add_argument("--session", default="cli", help="Session ID")

    # run
    run_parser = subparsers.add_parser("run", help="Execute a task")
    run_parser.add_argument("task", help="Task description")
    run_parser.add_argument("--agent", default=None, help="Named agent to use")
    run_parser.add_argument("--crew", default=None, help="Named crew to use")

    # agents
    subparsers.add_parser("agents", help="List all agents")

    # status
    subparsers.add_parser("status", help="Show framework status")

    # tools
    subparsers.add_parser("tools", help="List available tools")

    # knowledge
    kg_parser = subparsers.add_parser("knowledge", help="Query knowledge graph")
    kg_parser.add_argument("query", help="Query string")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "chat":
        asyncio.run(cmd_chat(args))
    elif args.command == "run":
        asyncio.run(cmd_run(args))
    elif args.command == "agents":
        cmd_agents(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "tools":
        cmd_tools(args)
    elif args.command == "knowledge":
        asyncio.run(cmd_knowledge(args))


def cmd_serve(args):
    """Start the NEXUS API server."""
    import uvicorn
    from nexus import Nexus
    from nexus.api.server import create_api
    from nexus.tools.builtins import register_builtin_tools

    nx = Nexus()
    register_builtin_tools(nx.tool_registry)
    app = create_api(nx)

    print(f"""
╔══════════════════════════════════════════════════╗
║          NEXUS AI Framework v1.0.0               ║
║     Next-Gen Autonomous Agent System             ║
╠══════════════════════════════════════════════════╣
║  API Server: http://{args.host}:{args.port}          ║
║  OpenAI-compat: /v1/chat/completions             ║
║  MCP Server: /mcp                                ║
║  Docs: http://{args.host}:{args.port}/docs           ║
╚══════════════════════════════════════════════════╝
    """)

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


async def cmd_chat(args):
    """Interactive chat session."""
    from nexus import Nexus
    from nexus.tools.builtins import register_builtin_tools

    nx = Nexus()
    register_builtin_tools(nx.tool_registry)

    print("NEXUS Interactive Chat (type 'quit' to exit)")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not user_input:
                continue

            result = await nx.chat(
                message=user_input,
                session_id=args.session,
            )
            print(f"\nNEXUS: {result.output}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


async def cmd_run(args):
    """Execute a single task."""
    from nexus import Nexus
    from nexus.tools.builtins import register_builtin_tools

    nx = Nexus()
    register_builtin_tools(nx.tool_registry)

    print(f"Executing: {args.task}")
    print("-" * 50)

    result = await nx.run(
        task=args.task,
        agent=args.agent,
        crew=args.crew,
    )

    print(f"\nResult: {result.output}")
    print(f"\nSuccess: {result.success}")
    print(f"Duration: {result.total_duration_ms:.0f}ms")
    print(f"Tokens: {result.tokens_used}")


def cmd_agents(args):
    """List all agents."""
    from nexus import Nexus
    nx = Nexus()
    agents = nx.agent_manager.list_agents()
    if not agents:
        print("No agents registered. Create agents via API or code.")
        return
    for agent in agents:
        print(f"  {agent['name']} ({agent['role']}) - {agent['state']} - model: {agent['model']}")


def cmd_status(args):
    """Show framework status."""
    from nexus import Nexus
    nx = Nexus()
    status = nx.status
    print(json.dumps(status, indent=2))


def cmd_tools(args):
    """List available tools."""
    from nexus import Nexus
    from nexus.tools.builtins import register_builtin_tools

    nx = Nexus()
    register_builtin_tools(nx.tool_registry)
    tools = nx.tool_registry.list_tools()
    print(f"Available tools ({len(tools)}):")
    for tool in tools:
        print(f"  {tool['name']}: {tool['description']} [{', '.join(tool['tags'])}]")


async def cmd_knowledge(args):
    """Query the knowledge graph."""
    from nexus import Nexus
    nx = Nexus()
    results = nx.memory_manager.semantic.recall(args.query)
    if not results:
        print("No knowledge found.")
        return
    for r in results:
        connections = ", ".join(f"{c['relation']} {c['entity']}" for c in r.get("connections", []))
        print(f"  {r['entity']} ({r['type']}): {connections}")


if __name__ == "__main__":
    main()

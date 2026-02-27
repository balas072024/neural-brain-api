"""
NEXUS Communication Channels
==============================
Multi-platform messaging support inspired by OpenClaw.
Supports: WhatsApp, Telegram, Slack, Discord, Teams, WebChat, and more.

Architecture:
- Channel Adapter Pattern: Each platform normalized to common message format
- Webhook receivers for incoming messages
- Outgoing message dispatch per channel
- A2A (Agent-to-Agent) protocol for inter-agent communication
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable


class ChannelType(str, Enum):
    WEBCHAT = "webchat"
    TELEGRAM = "telegram"
    SLACK = "slack"
    DISCORD = "discord"
    WHATSAPP = "whatsapp"
    TEAMS = "teams"
    SIGNAL = "signal"
    MATRIX = "matrix"
    EMAIL = "email"
    CLI = "cli"
    API = "api"
    A2A = "a2a"  # Agent-to-Agent protocol


@dataclass
class NormalizedMessage:
    """Platform-agnostic message format."""
    id: str = ""
    channel: ChannelType = ChannelType.API
    sender_id: str = ""
    sender_name: str = ""
    content: str = ""
    attachments: list[dict] = field(default_factory=list)
    reply_to: str | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class ChannelAdapter:
    """Base class for channel adapters."""

    def __init__(self, channel_type: ChannelType, credentials: dict[str, str] | None = None):
        self.channel_type = channel_type
        self.credentials = credentials or {}
        self._connected = False
        self._message_handler: Callable | None = None

    def on_message(self, handler: Callable[[NormalizedMessage], Awaitable[str]]):
        """Register a message handler."""
        self._message_handler = handler

    async def connect(self):
        """Connect to the messaging platform."""
        self._connected = True

    async def disconnect(self):
        """Disconnect from the messaging platform."""
        self._connected = False

    async def send(self, recipient_id: str, content: str, **kwargs):
        """Send a message to a recipient."""
        raise NotImplementedError

    def normalize(self, raw_message: dict) -> NormalizedMessage:
        """Convert platform-specific message to normalized format."""
        return NormalizedMessage(
            channel=self.channel_type,
            content=raw_message.get("text", raw_message.get("content", "")),
            sender_id=raw_message.get("from", raw_message.get("user", "")),
            raw=raw_message,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected


class TelegramAdapter(ChannelAdapter):
    """Telegram Bot API adapter."""

    def __init__(self, credentials: dict[str, str] | None = None):
        super().__init__(ChannelType.TELEGRAM, credentials)
        self.bot_token = credentials.get("bot_token", "") if credentials else ""

    async def send(self, recipient_id: str, content: str, **kwargs):
        import aiohttp
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        async with aiohttp.ClientSession() as session:
            await session.post(url, json={"chat_id": recipient_id, "text": content})


class SlackAdapter(ChannelAdapter):
    """Slack API adapter."""

    def __init__(self, credentials: dict[str, str] | None = None):
        super().__init__(ChannelType.SLACK, credentials)
        self.bot_token = credentials.get("bot_token", "") if credentials else ""

    async def send(self, recipient_id: str, content: str, **kwargs):
        import aiohttp
        url = "https://slack.com/api/chat.postMessage"
        headers = {"Authorization": f"Bearer {self.bot_token}"}
        async with aiohttp.ClientSession() as session:
            await session.post(url, json={"channel": recipient_id, "text": content}, headers=headers)


class DiscordAdapter(ChannelAdapter):
    """Discord Bot adapter."""

    def __init__(self, credentials: dict[str, str] | None = None):
        super().__init__(ChannelType.DISCORD, credentials)
        self.bot_token = credentials.get("bot_token", "") if credentials else ""

    async def send(self, recipient_id: str, content: str, **kwargs):
        import aiohttp
        url = f"https://discord.com/api/v10/channels/{recipient_id}/messages"
        headers = {"Authorization": f"Bot {self.bot_token}"}
        async with aiohttp.ClientSession() as session:
            await session.post(url, json={"content": content}, headers=headers)


class WebChatAdapter(ChannelAdapter):
    """WebSocket-based web chat adapter."""

    def __init__(self, credentials: dict[str, str] | None = None):
        super().__init__(ChannelType.WEBCHAT, credentials)
        self._connections: dict[str, Any] = {}

    async def send(self, recipient_id: str, content: str, **kwargs):
        conn = self._connections.get(recipient_id)
        if conn:
            await conn.send_str(content)


class A2AAdapter(ChannelAdapter):
    """
    Agent-to-Agent (A2A) Protocol adapter.
    Enables NEXUS agents to communicate with agents from other frameworks.
    Follows Google's A2A protocol specification.
    """

    def __init__(self, credentials: dict[str, str] | None = None):
        super().__init__(ChannelType.A2A, credentials)
        self._agent_registry: dict[str, dict] = {}

    async def discover_agent(self, agent_url: str) -> dict:
        """Discover another agent's capabilities via A2A protocol."""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{agent_url}/.well-known/agent.json") as resp:
                if resp.status == 200:
                    agent_info = await resp.json()
                    self._agent_registry[agent_url] = agent_info
                    return agent_info
        return {}

    async def send(self, recipient_id: str, content: str, **kwargs):
        """Send a message to another agent via A2A protocol."""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            await session.post(
                f"{recipient_id}/a2a/messages",
                json={"content": content, "type": "task", **kwargs},
            )


class ChannelManager:
    """
    Manages all connected communication channels.
    Routes messages between agents and platforms.
    """

    ADAPTERS = {
        "telegram": TelegramAdapter,
        "slack": SlackAdapter,
        "discord": DiscordAdapter,
        "webchat": WebChatAdapter,
        "a2a": A2AAdapter,
    }

    def __init__(self, config=None):
        self.config = config
        self._channels: dict[str, ChannelAdapter] = {}
        self._message_handlers: list[Callable] = []

    def connect(self, channel_type: str, **credentials):
        """Connect a messaging channel."""
        adapter_class = self.ADAPTERS.get(channel_type)
        if not adapter_class:
            raise ValueError(f"Unknown channel type: {channel_type}. Available: {list(self.ADAPTERS.keys())}")

        adapter = adapter_class(credentials=credentials)
        self._channels[channel_type] = adapter

        # Register message handler
        for handler in self._message_handlers:
            adapter.on_message(handler)

        return adapter

    def on_message(self, handler: Callable[[NormalizedMessage], Awaitable[str]]):
        """Register a global message handler for all channels."""
        self._message_handlers.append(handler)
        for adapter in self._channels.values():
            adapter.on_message(handler)

    async def broadcast(self, content: str, channels: list[str] | None = None):
        """Send a message to multiple channels."""
        targets = channels or list(self._channels.keys())
        for channel_type in targets:
            adapter = self._channels.get(channel_type)
            if adapter and adapter.is_connected:
                await adapter.send("broadcast", content)

    async def start_all(self):
        """Connect all configured channels."""
        tasks = [adapter.connect() for adapter in self._channels.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all(self):
        """Disconnect all channels."""
        tasks = [adapter.disconnect() for adapter in self._channels.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

    def list_channels(self) -> list[dict]:
        """List all connected channels."""
        return [
            {"type": name, "connected": adapter.is_connected}
            for name, adapter in self._channels.items()
        ]

    @property
    def connected_count(self) -> int:
        return sum(1 for a in self._channels.values() if a.is_connected)

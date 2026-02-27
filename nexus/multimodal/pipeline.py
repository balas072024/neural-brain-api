"""
NEXUS Multi-Modal Pipeline
============================
Process and route multi-modal inputs: text, image, audio, video.
Inspired by JARVIS's multi-modal model orchestration.

Features:
- Automatic modality detection
- Model routing based on modality
- Cross-modal translation (image→text, audio→text, etc.)
- Streaming pipeline for real-time processing
- Voice interface with STT/TTS integration
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from nexus.core.config import NexusConfig


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"
    DOCUMENT = "document"


@dataclass
class ModalInput:
    """A multi-modal input item."""
    modality: Modality
    content: str | bytes = ""  # Text content or binary data
    mime_type: str = ""
    file_path: str = ""
    url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModalOutput:
    """Output from multi-modal processing."""
    text: str = ""
    modality: Modality = Modality.TEXT
    artifacts: list[dict] = field(default_factory=list)  # Generated images, audio, etc.
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiModalPipeline:
    """
    Multi-modal processing pipeline.

    Detects input modality, routes to appropriate model,
    and produces outputs across modalities.
    """

    def __init__(self, config: NexusConfig):
        self.config = config
        self._processors: dict[Modality, Any] = {}

    def detect_modality(self, input_data: Any) -> Modality:
        """Auto-detect the modality of input data."""
        if isinstance(input_data, str):
            # Check for file paths
            if any(input_data.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]):
                return Modality.IMAGE
            if any(input_data.lower().endswith(ext) for ext in [".mp3", ".wav", ".ogg", ".flac", ".m4a"]):
                return Modality.AUDIO
            if any(input_data.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".webm"]):
                return Modality.VIDEO
            if any(input_data.lower().endswith(ext) for ext in [".pdf", ".docx", ".txt", ".md"]):
                return Modality.DOCUMENT
            if any(input_data.lower().endswith(ext) for ext in [".py", ".js", ".ts", ".go", ".rs", ".java"]):
                return Modality.CODE
            # Check for base64 image data
            if input_data.startswith("data:image/"):
                return Modality.IMAGE
            return Modality.TEXT
        elif isinstance(input_data, bytes):
            # Check magic bytes
            if input_data[:4] in (b"\x89PNG", b"\xff\xd8\xff"):
                return Modality.IMAGE
            return Modality.TEXT
        return Modality.TEXT

    async def process(
        self,
        inputs: list[ModalInput] | str,
        task: str = "",
        model_router=None,
    ) -> ModalOutput:
        """Process multi-modal inputs and generate output."""
        if isinstance(inputs, str):
            modality = self.detect_modality(inputs)
            inputs = [ModalInput(modality=modality, content=inputs)]

        output = ModalOutput()
        results = []

        for inp in inputs:
            if inp.modality == Modality.TEXT:
                results.append(inp.content)
            elif inp.modality == Modality.IMAGE:
                result = await self._process_image(inp, task, model_router)
                results.append(result)
            elif inp.modality == Modality.AUDIO:
                result = await self._process_audio(inp, task, model_router)
                results.append(result)
            elif inp.modality == Modality.CODE:
                results.append(f"[Code file: {inp.file_path or 'inline'}]\n{inp.content}")
            elif inp.modality == Modality.DOCUMENT:
                result = await self._process_document(inp)
                results.append(result)

        output.text = "\n\n".join(str(r) for r in results if r)
        return output

    async def _process_image(self, inp: ModalInput, task: str, model_router=None) -> str:
        """Process an image input — describe, analyze, or transform."""
        if model_router:
            # Use a vision-capable model
            messages = [{"role": "user", "content": task or "Describe this image in detail."}]

            if inp.file_path and Path(inp.file_path).exists():
                image_data = Path(inp.file_path).read_bytes()
                b64 = base64.b64encode(image_data).decode()
                mime = inp.mime_type or "image/png"
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task or "Describe this image."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }]

            model = await model_router.select_model(
                task_type="vision", requires_vision=True
            )
            response = await model_router.generate(model=model, messages=messages)
            return response.get("content", "Image processing complete.")

        return f"[Image: {inp.file_path or 'inline data'}]"

    async def _process_audio(self, inp: ModalInput, task: str, model_router=None) -> str:
        """Process audio input — transcribe or analyze."""
        if inp.file_path:
            return f"[Audio file: {inp.file_path}] — Transcription requires speech-to-text service."
        return "[Audio data received] — Transcription requires speech-to-text service."

    async def _process_document(self, inp: ModalInput) -> str:
        """Process document input — extract text."""
        if inp.file_path:
            path = Path(inp.file_path)
            if path.exists() and path.suffix in (".txt", ".md", ".py", ".js"):
                content = path.read_text(errors="replace")
                return f"[Document: {path.name}]\n{content[:10000]}"
        return f"[Document: {inp.file_path}]"

    async def text_to_speech(self, text: str, voice: str = "default") -> bytes | None:
        """Convert text to speech audio. Requires TTS service integration."""
        # Placeholder for TTS integration (e.g., ElevenLabs, OpenAI TTS)
        return None

    async def speech_to_text(self, audio_data: bytes) -> str:
        """Convert speech audio to text. Requires STT service integration."""
        # Placeholder for STT integration (e.g., Whisper, Deepgram)
        return ""

    def create_voice_pipeline(self, stt_provider: str = "whisper", tts_provider: str = "elevenlabs"):
        """Create a voice processing pipeline with STT and TTS."""
        return VoicePipeline(
            stt_provider=stt_provider,
            tts_provider=tts_provider,
            config=self.config,
        )


class VoicePipeline:
    """
    Real-time voice pipeline for conversational AI.

    Architecture: STT → LLM → TTS (streaming)
    Target latency: <200ms per turn.
    """

    def __init__(self, stt_provider: str = "whisper", tts_provider: str = "elevenlabs", config=None):
        self.stt_provider = stt_provider
        self.tts_provider = tts_provider
        self.config = config
        self._active = False

    async def start(self):
        """Start the voice pipeline."""
        self._active = True

    async def stop(self):
        """Stop the voice pipeline."""
        self._active = False

    async def process_turn(self, audio_input: bytes, model_router=None) -> dict:
        """
        Process a single voice turn:
        1. STT: Audio → Text
        2. LLM: Text → Response text
        3. TTS: Response text → Audio

        Returns both text and audio response.
        """
        start_time = time.time()

        # Step 1: Speech to Text
        transcript = await self._stt(audio_input)

        # Step 2: Generate response
        response_text = ""
        if model_router and transcript:
            result = await model_router.generate(
                model="claude-haiku-4-5",  # Fast model for voice
                messages=[{"role": "user", "content": transcript}],
                max_tokens=500,  # Keep responses concise for voice
            )
            response_text = result.get("content", "")

        # Step 3: Text to Speech
        response_audio = await self._tts(response_text)

        return {
            "transcript": transcript,
            "response_text": response_text,
            "response_audio": response_audio,
            "latency_ms": (time.time() - start_time) * 1000,
        }

    async def _stt(self, audio: bytes) -> str:
        """Speech-to-text conversion."""
        # Placeholder — integrate with Whisper, Deepgram, AssemblyAI, etc.
        return ""

    async def _tts(self, text: str) -> bytes | None:
        """Text-to-speech conversion."""
        # Placeholder — integrate with ElevenLabs, OpenAI TTS, etc.
        return None

    @property
    def is_active(self) -> bool:
        return self._active

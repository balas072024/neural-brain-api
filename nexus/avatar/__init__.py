"""
TIRAM 3D Avatar System
========================
Interactive 3D female avatar persona that speaks, listens, and communicates
with real-time lip-sync, facial expressions, and emotional intelligence.

Architecture:
  Audio2Face Pipeline:
    1. Speech-to-Text (STT)     → Whisper / Deepgram
    2. LLM Processing           → Claude / GPT / Local
    3. Text-to-Speech (TTS)     → ElevenLabs / Coqui / OpenAI TTS
    4. Audio-to-Viseme          → NVIDIA Audio2Face / Wav2Lip
    5. 3D Rendering             → Three.js / WebGL / Unity WebGL
    6. Emotion Detection        → Sentiment + Prosody analysis

  Persona System:
    - Configurable personality, voice, appearance
    - Emotional state tracking (happy, focused, thinking, empathetic)
    - Context-aware responses with tonal adaptation
    - Multi-language voice support (50+ languages)

  Delivery Modes:
    - WebGL in-browser (Three.js + ReadyPlayerMe)
    - Video stream (WebRTC / HLS)
    - Audio-only mode (for lightweight clients)
    - CLI text mode (fallback)

Inspired by:
  - NVIDIA Audio2Face (open-sourced)
  - FACSvatar (FACS-based animation)
  - SadTalker / LivePortrait (neural lip-sync)
  - Khanmigo (conversational tutor persona)
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EmotionalState(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    THINKING = "thinking"
    FOCUSED = "focused"
    EMPATHETIC = "empathetic"
    EXCITED = "excited"
    CONCERNED = "concerned"
    ENCOURAGING = "encouraging"
    EXPLAINING = "explaining"
    LISTENING = "listening"


class DeliveryMode(str, Enum):
    WEBGL_3D = "webgl_3d"       # Full 3D in browser
    VIDEO_STREAM = "video"       # Pre-rendered video stream
    AUDIO_ONLY = "audio"         # Voice without visual
    TEXT_ONLY = "text"           # Text fallback


class VoiceStyle(str, Enum):
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CALM = "calm"
    ENERGETIC = "energetic"
    WARM = "warm"
    AUTHORITATIVE = "authoritative"


@dataclass
class AvatarAppearance:
    """Configurable appearance for the 3D avatar."""
    name: str = "Tiram"
    gender: str = "female"
    model_url: str = ""            # ReadyPlayerMe / custom GLB/GLTF URL
    skin_tone: str = "medium"
    hair_style: str = "long_wavy"
    hair_color: str = "#2C1810"
    eye_color: str = "#4A90D9"
    outfit: str = "professional"   # professional, casual, creative, tech
    background: str = "gradient"   # gradient, office, nature, space, minimal
    animation_style: str = "natural"  # natural, expressive, minimal


@dataclass
class AvatarVoice:
    """Voice configuration for TTS."""
    provider: str = "elevenlabs"   # elevenlabs, openai, coqui, azure, google
    voice_id: str = ""             # Provider-specific voice ID
    language: str = "en"
    accent: str = "neutral"
    speed: float = 1.0
    pitch: float = 1.0
    style: VoiceStyle = VoiceStyle.FRIENDLY
    stability: float = 0.75
    clarity: float = 0.8


@dataclass
class AvatarPersonality:
    """Personality traits for the avatar."""
    name: str = "Tiram"
    tagline: str = "Your intelligent AI companion"
    personality_traits: list[str] = field(default_factory=lambda: [
        "warm", "knowledgeable", "patient", "encouraging", "precise",
        "creative", "adaptable", "multilingual",
    ])
    communication_style: str = "conversational"  # formal, conversational, casual
    humor_level: float = 0.3        # 0 = serious, 1 = very humorous
    empathy_level: float = 0.8      # 0 = robotic, 1 = very empathetic
    proactivity: float = 0.6        # 0 = only responds, 1 = suggests and anticipates
    expertise_areas: list[str] = field(default_factory=lambda: [
        "software engineering", "data science", "design", "business",
        "education", "mathematics", "creative writing", "languages",
    ])

    def to_system_prompt(self) -> str:
        """Generate a system prompt from personality configuration."""
        traits = ", ".join(self.personality_traits)
        expertise = ", ".join(self.expertise_areas)
        return (
            f"You are {self.name}, an intelligent AI companion. "
            f"Your personality: {traits}. "
            f"Communication style: {self.communication_style}. "
            f"Your expertise spans: {expertise}. "
            f"You speak naturally, with empathy and clarity. "
            f"Adapt your tone to match the user's mood and needs. "
            f"When teaching, use the Socratic method — guide through questions. "
            f"When building, be precise and thorough. "
            f"You can speak and understand 50+ languages fluently."
        )


@dataclass
class Viseme:
    """A single viseme (mouth shape) for lip-sync."""
    id: int
    name: str       # e.g., "aa", "ee", "oh", "mm", "silence"
    duration_ms: float
    intensity: float = 1.0


@dataclass
class FacialExpression:
    """A facial expression blend shape set."""
    emotion: EmotionalState
    eyebrow_raise: float = 0.0   # -1 to 1
    eye_squint: float = 0.0      # 0 to 1
    smile: float = 0.0           # 0 to 1
    mouth_open: float = 0.0      # 0 to 1
    head_nod: float = 0.0        # -1 to 1
    head_tilt: float = 0.0       # -1 to 1
    blink_rate: float = 0.3      # blinks per second

    @classmethod
    def for_emotion(cls, emotion: EmotionalState) -> FacialExpression:
        """Get the facial expression blend for an emotion."""
        expressions = {
            EmotionalState.NEUTRAL: cls(emotion=emotion, smile=0.1, blink_rate=0.3),
            EmotionalState.HAPPY: cls(emotion=emotion, smile=0.8, eyebrow_raise=0.2, blink_rate=0.2),
            EmotionalState.THINKING: cls(emotion=emotion, eyebrow_raise=0.4, eye_squint=0.2, head_tilt=0.3, blink_rate=0.15),
            EmotionalState.FOCUSED: cls(emotion=emotion, eyebrow_raise=-0.2, eye_squint=0.3, blink_rate=0.15),
            EmotionalState.EMPATHETIC: cls(emotion=emotion, smile=0.3, eyebrow_raise=0.3, head_tilt=0.2, head_nod=0.4),
            EmotionalState.EXCITED: cls(emotion=emotion, smile=0.9, eyebrow_raise=0.5, blink_rate=0.4),
            EmotionalState.CONCERNED: cls(emotion=emotion, eyebrow_raise=0.5, smile=-0.2, head_tilt=-0.2),
            EmotionalState.ENCOURAGING: cls(emotion=emotion, smile=0.6, head_nod=0.5, eyebrow_raise=0.2),
            EmotionalState.EXPLAINING: cls(emotion=emotion, eyebrow_raise=0.3, mouth_open=0.2, head_nod=0.3),
            EmotionalState.LISTENING: cls(emotion=emotion, head_nod=0.3, smile=0.2, eye_squint=0.1),
        }
        return expressions.get(emotion, cls(emotion=emotion))


@dataclass
class AvatarFrame:
    """A single frame of avatar animation data."""
    timestamp_ms: float
    visemes: list[Viseme] = field(default_factory=list)
    expression: FacialExpression | None = None
    audio_chunk: bytes | None = None
    text_chunk: str = ""


class AvatarPipeline:
    """
    Complete 3D avatar pipeline: STT → LLM → TTS → Viseme → Render.

    Stages:
    1. Capture:    User audio/text input
    2. Understand: STT + emotion detection
    3. Think:      LLM processes with persona context
    4. Speak:      TTS generates speech audio
    5. Animate:    Audio → viseme mapping + facial expressions
    6. Render:     3D model animation data sent to client
    """

    def __init__(
        self,
        appearance: AvatarAppearance | None = None,
        voice: AvatarVoice | None = None,
        personality: AvatarPersonality | None = None,
        config=None,
    ):
        self.appearance = appearance or AvatarAppearance()
        self.voice = voice or AvatarVoice()
        self.personality = personality or AvatarPersonality()
        self.config = config

        self._current_emotion = EmotionalState.NEUTRAL
        self._conversation_history: list[dict] = []
        self._is_speaking = False
        self._is_listening = False

    async def process_interaction(
        self,
        user_input: str | bytes,
        model_router=None,
        input_language: str = "auto",
    ) -> dict[str, Any]:
        """
        Full interaction pipeline:
        Input (text/audio) → Understanding → Response → Animation Data
        """
        start_time = time.time()
        result: dict[str, Any] = {
            "input_text": "",
            "response_text": "",
            "emotion": EmotionalState.NEUTRAL.value,
            "expression": {},
            "viseme_sequence": [],
            "audio_data": None,
            "language_detected": "en",
            "latency_ms": 0,
        }

        # Stage 1: Understand input
        if isinstance(user_input, bytes):
            # Audio input → STT
            self._is_listening = True
            input_text = await self._speech_to_text(user_input, input_language)
            self._is_listening = False
            result["input_text"] = input_text
        else:
            input_text = user_input
            result["input_text"] = input_text

        if not input_text:
            return result

        # Stage 2: Detect emotion from input
        detected_emotion = self._detect_emotion(input_text)
        response_emotion = self._select_response_emotion(detected_emotion)
        self._current_emotion = response_emotion
        result["emotion"] = response_emotion.value

        # Stage 3: Generate response via LLM
        response_text = await self._generate_response(
            input_text, model_router, response_emotion
        )
        result["response_text"] = response_text

        # Stage 4: TTS — Convert response to speech
        audio_data = await self._text_to_speech(response_text)
        result["audio_data"] = audio_data

        # Stage 5: Generate viseme sequence for lip-sync
        visemes = self._generate_viseme_sequence(response_text)
        result["viseme_sequence"] = [
            {"id": v.id, "name": v.name, "duration_ms": v.duration_ms, "intensity": v.intensity}
            for v in visemes
        ]

        # Stage 6: Generate facial expression
        expression = FacialExpression.for_emotion(response_emotion)
        result["expression"] = {
            "eyebrow_raise": expression.eyebrow_raise,
            "eye_squint": expression.eye_squint,
            "smile": expression.smile,
            "head_nod": expression.head_nod,
            "head_tilt": expression.head_tilt,
            "blink_rate": expression.blink_rate,
        }

        # Update conversation history
        self._conversation_history.append({"role": "user", "content": input_text})
        self._conversation_history.append({"role": "assistant", "content": response_text})

        result["latency_ms"] = (time.time() - start_time) * 1000
        return result

    async def _speech_to_text(self, audio: bytes, language: str = "auto") -> str:
        """Convert speech to text using configured STT provider."""
        # Production: integrate Whisper, Deepgram, or AssemblyAI
        # For now, return placeholder
        import aiohttp
        try:
            # Example: OpenAI Whisper API
            import os
            api_key = os.getenv("OPENAI_API_KEY", "")
            if api_key:
                async with aiohttp.ClientSession() as session:
                    form = aiohttp.FormData()
                    form.add_field("file", audio, filename="audio.wav", content_type="audio/wav")
                    form.add_field("model", "whisper-1")
                    if language and language != "auto":
                        form.add_field("language", language)
                    async with session.post(
                        "https://api.openai.com/v1/audio/transcriptions",
                        data=form,
                        headers={"Authorization": f"Bearer {api_key}"},
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data.get("text", "")
        except Exception:
            pass
        return "[Audio input received — STT service not configured]"

    async def _text_to_speech(self, text: str) -> bytes | None:
        """Convert text to speech using configured TTS provider."""
        import aiohttp
        import os

        if self.voice.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if api_key:
                try:
                    async with aiohttp.ClientSession() as session:
                        body = {
                            "model": "tts-1-hd",
                            "input": text[:4096],
                            "voice": self.voice.voice_id or "nova",
                            "speed": self.voice.speed,
                        }
                        async with session.post(
                            "https://api.openai.com/v1/audio/speech",
                            json=body,
                            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        ) as resp:
                            if resp.status == 200:
                                return await resp.read()
                except Exception:
                    pass

        elif self.voice.provider == "elevenlabs":
            api_key = os.getenv("ELEVENLABS_API_KEY", "")
            voice_id = self.voice.voice_id or "21m00Tcm4TlvDq8ikWAM"
            if api_key:
                try:
                    async with aiohttp.ClientSession() as session:
                        body = {
                            "text": text[:5000],
                            "model_id": "eleven_multilingual_v2",
                            "voice_settings": {
                                "stability": self.voice.stability,
                                "similarity_boost": self.voice.clarity,
                                "style": 0.5,
                            },
                        }
                        async with session.post(
                            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                            json=body,
                            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
                        ) as resp:
                            if resp.status == 200:
                                return await resp.read()
                except Exception:
                    pass

        return None

    async def _generate_response(self, input_text: str, model_router=None,
                                  emotion: EmotionalState = EmotionalState.NEUTRAL) -> str:
        """Generate a response using the LLM with persona context."""
        if not model_router:
            return f"[TIRAM would respond to: {input_text}]"

        # Build messages with personality and conversation history
        system = self.personality.to_system_prompt()
        system += f"\n\nCurrent emotional context: You should respond with a {emotion.value} tone."

        messages = [{"role": "system", "content": system}]

        # Add recent conversation history (last 20 turns)
        for msg in self._conversation_history[-20:]:
            messages.append(msg)

        messages.append({"role": "user", "content": input_text})

        response = await model_router.generate(
            model=self.config.default_model if self.config else "claude-sonnet-4-6",
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
        )
        return response.get("content", "")

    def _detect_emotion(self, text: str) -> EmotionalState:
        """Detect the emotional tone of the user's input."""
        text_lower = text.lower()

        emotion_keywords = {
            EmotionalState.HAPPY: ["happy", "great", "awesome", "love", "excellent", "wonderful", "thanks", "thank you", "excited"],
            EmotionalState.CONCERNED: ["worried", "concern", "problem", "issue", "broken", "error", "bug", "fail", "wrong", "help"],
            EmotionalState.EXCITED: ["wow", "amazing", "incredible", "fantastic", "can't wait", "exciting"],
            EmotionalState.THINKING: ["how", "why", "what if", "consider", "think about", "wonder"],
            EmotionalState.FOCUSED: ["build", "create", "implement", "code", "develop", "make", "write"],
        }

        for emotion, keywords in emotion_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return emotion

        return EmotionalState.NEUTRAL

    def _select_response_emotion(self, user_emotion: EmotionalState) -> EmotionalState:
        """Select the appropriate response emotion based on user's emotion."""
        response_map = {
            EmotionalState.HAPPY: EmotionalState.HAPPY,
            EmotionalState.CONCERNED: EmotionalState.EMPATHETIC,
            EmotionalState.EXCITED: EmotionalState.EXCITED,
            EmotionalState.THINKING: EmotionalState.EXPLAINING,
            EmotionalState.FOCUSED: EmotionalState.FOCUSED,
            EmotionalState.NEUTRAL: EmotionalState.FRIENDLY if hasattr(EmotionalState, 'FRIENDLY') else EmotionalState.NEUTRAL,
        }
        return response_map.get(user_emotion, EmotionalState.NEUTRAL)

    def _generate_viseme_sequence(self, text: str) -> list[Viseme]:
        """
        Generate a viseme sequence for lip-sync from text.
        Uses phoneme-to-viseme mapping for realistic mouth animation.
        """
        # Simplified phoneme-to-viseme mapping
        # In production, use NVIDIA Audio2Face or a phoneme analysis library
        viseme_map = {
            "a": Viseme(id=1, name="aa", duration_ms=80),
            "e": Viseme(id=2, name="ee", duration_ms=70),
            "i": Viseme(id=3, name="ih", duration_ms=60),
            "o": Viseme(id=4, name="oh", duration_ms=80),
            "u": Viseme(id=5, name="oo", duration_ms=70),
            "m": Viseme(id=6, name="mm", duration_ms=60, intensity=0.8),
            "b": Viseme(id=6, name="mm", duration_ms=50, intensity=0.9),
            "p": Viseme(id=6, name="mm", duration_ms=50, intensity=0.9),
            "f": Viseme(id=7, name="ff", duration_ms=60),
            "v": Viseme(id=7, name="ff", duration_ms=60),
            "t": Viseme(id=8, name="tt", duration_ms=40),
            "d": Viseme(id=8, name="tt", duration_ms=40),
            "s": Viseme(id=9, name="ss", duration_ms=60),
            "z": Viseme(id=9, name="ss", duration_ms=60),
            "l": Viseme(id=10, name="ll", duration_ms=50),
            "r": Viseme(id=11, name="rr", duration_ms=50),
            "n": Viseme(id=12, name="nn", duration_ms=50),
            "k": Viseme(id=13, name="kk", duration_ms=40),
            "g": Viseme(id=13, name="kk", duration_ms=40),
            " ": Viseme(id=0, name="silence", duration_ms=100, intensity=0),
        }

        visemes = []
        for char in text.lower():
            if char in viseme_map:
                visemes.append(viseme_map[char])
            elif char.isalpha():
                visemes.append(Viseme(id=14, name="generic", duration_ms=50))

        return visemes

    def get_webgl_config(self) -> dict:
        """Get configuration for the WebGL 3D renderer."""
        return {
            "avatar": {
                "model_url": self.appearance.model_url or "https://models.readyplayer.me/default-avatar.glb",
                "gender": self.appearance.gender,
                "outfit": self.appearance.outfit,
                "hair": {"style": self.appearance.hair_style, "color": self.appearance.hair_color},
                "eyes": {"color": self.appearance.eye_color},
                "skin_tone": self.appearance.skin_tone,
                "background": self.appearance.background,
                "animation_style": self.appearance.animation_style,
            },
            "voice": {
                "provider": self.voice.provider,
                "language": self.voice.language,
                "style": self.voice.style.value,
                "speed": self.voice.speed,
            },
            "personality": {
                "name": self.personality.name,
                "tagline": self.personality.tagline,
                "traits": self.personality.personality_traits,
                "communication_style": self.personality.communication_style,
            },
            "rendering": {
                "fps": 60,
                "quality": "high",
                "shadows": True,
                "anti_aliasing": True,
                "ambient_occlusion": True,
            },
        }

    @property
    def state(self) -> dict:
        return {
            "name": self.personality.name,
            "emotion": self._current_emotion.value,
            "is_speaking": self._is_speaking,
            "is_listening": self._is_listening,
            "conversation_turns": len(self._conversation_history) // 2,
            "delivery_mode": "webgl_3d",
        }

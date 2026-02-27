"""
TIRAM Multilingual Intelligence System
=========================================
Full multilingual support across 50+ languages with:
- Auto language detection
- Real-time translation
- Cultural context adaptation
- Multilingual content generation
- Code comments in any language
- Localization assistance (i18n/l10n)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LanguageProfile:
    """Profile for a supported language."""
    code: str        # ISO 639-1 code
    name: str        # English name
    native_name: str # Name in the language itself
    direction: str = "ltr"  # ltr or rtl
    script: str = "Latin"
    region: str = ""


# Comprehensive language support — 50+ languages
SUPPORTED_LANGUAGES: dict[str, LanguageProfile] = {
    "en": LanguageProfile("en", "English", "English", script="Latin"),
    "es": LanguageProfile("es", "Spanish", "Español", script="Latin"),
    "fr": LanguageProfile("fr", "French", "Français", script="Latin"),
    "de": LanguageProfile("de", "German", "Deutsch", script="Latin"),
    "it": LanguageProfile("it", "Italian", "Italiano", script="Latin"),
    "pt": LanguageProfile("pt", "Portuguese", "Português", script="Latin"),
    "nl": LanguageProfile("nl", "Dutch", "Nederlands", script="Latin"),
    "ru": LanguageProfile("ru", "Russian", "Русский", script="Cyrillic"),
    "uk": LanguageProfile("uk", "Ukrainian", "Українська", script="Cyrillic"),
    "pl": LanguageProfile("pl", "Polish", "Polski", script="Latin"),
    "cs": LanguageProfile("cs", "Czech", "Čeština", script="Latin"),
    "ro": LanguageProfile("ro", "Romanian", "Română", script="Latin"),
    "hu": LanguageProfile("hu", "Hungarian", "Magyar", script="Latin"),
    "el": LanguageProfile("el", "Greek", "Ελληνικά", script="Greek"),
    "tr": LanguageProfile("tr", "Turkish", "Türkçe", script="Latin"),
    "zh": LanguageProfile("zh", "Chinese", "中文", script="CJK"),
    "ja": LanguageProfile("ja", "Japanese", "日本語", script="CJK"),
    "ko": LanguageProfile("ko", "Korean", "한국어", script="Hangul"),
    "ar": LanguageProfile("ar", "Arabic", "العربية", direction="rtl", script="Arabic"),
    "he": LanguageProfile("he", "Hebrew", "עברית", direction="rtl", script="Hebrew"),
    "fa": LanguageProfile("fa", "Persian", "فارسی", direction="rtl", script="Arabic"),
    "ur": LanguageProfile("ur", "Urdu", "اردو", direction="rtl", script="Arabic"),
    "hi": LanguageProfile("hi", "Hindi", "हिन्दी", script="Devanagari"),
    "bn": LanguageProfile("bn", "Bengali", "বাংলা", script="Bengali"),
    "ta": LanguageProfile("ta", "Tamil", "தமிழ்", script="Tamil"),
    "te": LanguageProfile("te", "Telugu", "తెలుగు", script="Telugu"),
    "mr": LanguageProfile("mr", "Marathi", "मराठी", script="Devanagari"),
    "gu": LanguageProfile("gu", "Gujarati", "ગુજરાતી", script="Gujarati"),
    "kn": LanguageProfile("kn", "Kannada", "ಕನ್ನಡ", script="Kannada"),
    "ml": LanguageProfile("ml", "Malayalam", "മലയാളം", script="Malayalam"),
    "pa": LanguageProfile("pa", "Punjabi", "ਪੰਜਾਬੀ", script="Gurmukhi"),
    "th": LanguageProfile("th", "Thai", "ไทย", script="Thai"),
    "vi": LanguageProfile("vi", "Vietnamese", "Tiếng Việt", script="Latin"),
    "id": LanguageProfile("id", "Indonesian", "Bahasa Indonesia", script="Latin"),
    "ms": LanguageProfile("ms", "Malay", "Bahasa Melayu", script="Latin"),
    "tl": LanguageProfile("tl", "Filipino", "Filipino", script="Latin"),
    "sw": LanguageProfile("sw", "Swahili", "Kiswahili", script="Latin"),
    "am": LanguageProfile("am", "Amharic", "አማርኛ", script="Ethiopic"),
    "yo": LanguageProfile("yo", "Yoruba", "Yorùbá", script="Latin"),
    "ig": LanguageProfile("ig", "Igbo", "Igbo", script="Latin"),
    "ha": LanguageProfile("ha", "Hausa", "Hausa", script="Latin"),
    "zu": LanguageProfile("zu", "Zulu", "isiZulu", script="Latin"),
    "sv": LanguageProfile("sv", "Swedish", "Svenska", script="Latin"),
    "no": LanguageProfile("no", "Norwegian", "Norsk", script="Latin"),
    "da": LanguageProfile("da", "Danish", "Dansk", script="Latin"),
    "fi": LanguageProfile("fi", "Finnish", "Suomi", script="Latin"),
    "sk": LanguageProfile("sk", "Slovak", "Slovenčina", script="Latin"),
    "bg": LanguageProfile("bg", "Bulgarian", "Български", script="Cyrillic"),
    "sr": LanguageProfile("sr", "Serbian", "Српски", script="Cyrillic"),
    "hr": LanguageProfile("hr", "Croatian", "Hrvatski", script="Latin"),
    "ca": LanguageProfile("ca", "Catalan", "Català", script="Latin"),
    "eu": LanguageProfile("eu", "Basque", "Euskara", script="Latin"),
    "gl": LanguageProfile("gl", "Galician", "Galego", script="Latin"),
    "af": LanguageProfile("af", "Afrikaans", "Afrikaans", script="Latin"),
    "ne": LanguageProfile("ne", "Nepali", "नेपाली", script="Devanagari"),
    "si": LanguageProfile("si", "Sinhala", "සිංහල", script="Sinhala"),
    "my": LanguageProfile("my", "Burmese", "မြန်မာ", script="Myanmar"),
    "km": LanguageProfile("km", "Khmer", "ភាសាខ្មែរ", script="Khmer"),
    "lo": LanguageProfile("lo", "Lao", "ລາວ", script="Lao"),
    "ka": LanguageProfile("ka", "Georgian", "ქართული", script="Georgian"),
    "hy": LanguageProfile("hy", "Armenian", "Հայերեն", script="Armenian"),
    "az": LanguageProfile("az", "Azerbaijani", "Azərbaycan", script="Latin"),
    "uz": LanguageProfile("uz", "Uzbek", "Oʻzbek", script="Latin"),
    "kk": LanguageProfile("kk", "Kazakh", "Қазақ", script="Cyrillic"),
    "mn": LanguageProfile("mn", "Mongolian", "Монгол", script="Cyrillic"),
}


class MultilingualEngine:
    """
    Multilingual intelligence engine for TIRAM.

    Capabilities:
    - Auto-detect language from text
    - Translate between any supported languages
    - Generate content in any language
    - Adapt cultural context
    - Localization (i18n) assistance
    """

    def __init__(self, config=None):
        self.config = config
        self.languages = SUPPORTED_LANGUAGES
        self._default_language = "en"

    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text using character analysis.
        Returns ISO 639-1 language code.
        """
        if not text.strip():
            return self._default_language

        # Character-range based detection
        char_ranges = {
            "zh": (0x4E00, 0x9FFF),   # CJK Unified Ideographs
            "ja": (0x3040, 0x309F),   # Hiragana
            "ko": (0xAC00, 0xD7AF),   # Hangul
            "ar": (0x0600, 0x06FF),   # Arabic
            "he": (0x0590, 0x05FF),   # Hebrew
            "hi": (0x0900, 0x097F),   # Devanagari
            "th": (0x0E00, 0x0E7F),   # Thai
            "ru": (0x0400, 0x04FF),   # Cyrillic
            "el": (0x0370, 0x03FF),   # Greek
            "ka": (0x10A0, 0x10FF),   # Georgian
            "hy": (0x0530, 0x058F),   # Armenian
            "bn": (0x0980, 0x09FF),   # Bengali
            "ta": (0x0B80, 0x0BFF),   # Tamil
            "te": (0x0C00, 0x0C7F),   # Telugu
            "kn": (0x0C80, 0x0CFF),   # Kannada
            "ml": (0x0D00, 0x0D7F),   # Malayalam
            "gu": (0x0A80, 0x0AFF),   # Gujarati
            "pa": (0x0A00, 0x0A7F),   # Gurmukhi
            "my": (0x1000, 0x109F),   # Myanmar
            "km": (0x1780, 0x17FF),   # Khmer
            "lo": (0x0E80, 0x0EFF),   # Lao
            "am": (0x1200, 0x137F),   # Ethiopic
        }

        char_counts: dict[str, int] = {}
        for char in text:
            code_point = ord(char)
            for lang, (start, end) in char_ranges.items():
                if start <= code_point <= end:
                    char_counts[lang] = char_counts.get(lang, 0) + 1

        if char_counts:
            return max(char_counts, key=char_counts.get)

        # For Latin-script languages, use common word heuristics
        text_lower = text.lower()
        latin_indicators = {
            "es": ["el", "la", "los", "las", "de", "en", "que", "por", "con", "una", "como", "más", "pero"],
            "fr": ["le", "la", "les", "des", "une", "est", "dans", "pour", "avec", "que", "pas", "sur"],
            "de": ["der", "die", "das", "und", "ist", "von", "mit", "für", "auf", "ein", "den", "dem"],
            "it": ["il", "la", "di", "che", "è", "per", "con", "una", "sono", "della", "dei"],
            "pt": ["o", "a", "os", "as", "de", "em", "que", "por", "com", "uma", "para", "não"],
            "nl": ["de", "het", "een", "van", "en", "in", "is", "dat", "op", "voor", "met"],
            "tr": ["bir", "ve", "bu", "için", "ile", "olan", "gibi", "daha"],
            "pl": ["i", "w", "na", "do", "nie", "się", "jest", "co", "to", "jak"],
            "sv": ["och", "att", "det", "är", "för", "med", "den", "på", "av"],
            "id": ["yang", "dan", "di", "ini", "untuk", "dengan", "dari", "pada"],
            "vi": ["và", "của", "có", "trong", "được", "cho", "là", "này"],
        }

        words = text_lower.split()
        lang_scores: dict[str, int] = {}
        for lang, indicators in latin_indicators.items():
            score = sum(1 for word in words if word in indicators)
            if score > 0:
                lang_scores[lang] = score

        if lang_scores:
            return max(lang_scores, key=lang_scores.get)

        return "en"  # Default to English

    async def translate(self, text: str, target_language: str,
                       source_language: str = "auto", model_router=None) -> str:
        """Translate text between languages."""
        if source_language == "auto":
            source_language = self.detect_language(text)

        if source_language == target_language:
            return text

        target_info = self.languages.get(target_language)
        target_name = target_info.native_name if target_info else target_language

        if not model_router:
            return f"[Translation to {target_language}: {text}]"

        response = await model_router.generate(
            model=self.config.default_model if self.config else "claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": (
                    f"You are a professional translator. Translate accurately from "
                    f"{source_language} to {target_name}. "
                    "Preserve meaning, tone, and cultural nuance. "
                    "Output ONLY the translation, nothing else."
                )},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
        )
        return response.get("content", text)

    async def generate_multilingual(self, prompt: str, languages: list[str],
                                     model_router=None) -> dict[str, str]:
        """Generate content in multiple languages simultaneously."""
        results: dict[str, str] = {}

        # Generate in primary language first (usually English)
        primary = languages[0] if languages else "en"
        if model_router:
            response = await model_router.generate(
                model=self.config.default_model if self.config else "claude-sonnet-4-6",
                messages=[{"role": "user", "content": prompt}],
            )
            primary_text = response.get("content", "")
            results[primary] = primary_text

            # Translate to other languages
            for lang in languages[1:]:
                translated = await self.translate(primary_text, lang, primary, model_router)
                results[lang] = translated
        else:
            for lang in languages:
                results[lang] = f"[{lang}] {prompt}"

        return results

    def get_language_info(self, code: str) -> dict | None:
        """Get information about a language."""
        lang = self.languages.get(code)
        if lang:
            return {
                "code": lang.code,
                "name": lang.name,
                "native_name": lang.native_name,
                "direction": lang.direction,
                "script": lang.script,
            }
        return None

    def list_languages(self) -> list[dict]:
        """List all supported languages."""
        return [
            {"code": l.code, "name": l.name, "native_name": l.native_name, "script": l.script}
            for l in self.languages.values()
        ]

    @property
    def language_count(self) -> int:
        return len(self.languages)

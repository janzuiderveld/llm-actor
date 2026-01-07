from __future__ import annotations

from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.tts_service import TTSService
from pipecat.utils.string import SENTENCE_ENDING_PUNCTUATION, match_endofsentence
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

from app.config import DEFAULT_TTS_PROVIDER, RuntimeConfig
from services.macos_say_tts import MacosSayTTSService


class QuoteAwareTextAggregator(SimpleTextAggregator):
    """Text aggregator that treats closing quotes/parentheses after punctuation as sentence endings."""

    _CLOSING_TRAILERS = "\"')]}»›”’」』】》）］"

    async def aggregate(self, text: str):  # type: ignore[override]
        self._text += text

        eos_end_marker = match_endofsentence(self._text)
        if eos_end_marker:
            result = self._text[:eos_end_marker]
            self._text = self._text[eos_end_marker:]
            return result

        trimmed = self._text.rstrip()
        if not trimmed:
            return None

        stripped = trimmed.rstrip(self._CLOSING_TRAILERS)
        if stripped and stripped[-1] in SENTENCE_ENDING_PUNCTUATION:
            result = trimmed
            consumed = len(trimmed)
            remainder = self._text[consumed:]
            self._text = remainder.lstrip()
            return result

        return None


def build_deepgram_tts(config: RuntimeConfig, api_key: str) -> DeepgramTTSService:
    return DeepgramTTSService(
        api_key=api_key,
        voice=config.tts.voice,
        encoding=config.tts.encoding,
        sample_rate=config.tts.sample_rate,
        text_aggregator=QuoteAwareTextAggregator(),
    )


def build_tts_service(config: RuntimeConfig, deepgram_api_key: str) -> TTSService:
    provider = (config.tts.provider or DEFAULT_TTS_PROVIDER).strip().lower()
    if provider in ("deepgram", "deepgram_aura", "deepgram-aura"):
        return build_deepgram_tts(config, deepgram_api_key)
    if provider in ("macos_say", "macos-say", "say"):
        return MacosSayTTSService(
            voice=config.tts.say_voice,
            rate_wpm=config.tts.say_rate_wpm,
            audio_device=config.tts.say_audio_device,
            interactive=config.tts.say_interactive,
        )
    raise ValueError(f"Unknown TTS provider: {config.tts.provider!r}")

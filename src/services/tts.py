from __future__ import annotations

from pipecat.services.deepgram.tts import DeepgramTTSService

from app.config import RuntimeConfig


def build_deepgram_tts(config: RuntimeConfig, api_key: str) -> DeepgramTTSService:
    return DeepgramTTSService(
        api_key=api_key,
        voice=config.tts.voice,
        encoding=config.tts.encoding,
        sample_rate=config.tts.sample_rate,
    )

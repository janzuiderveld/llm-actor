from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import RuntimeConfig
from services.llm import build_google_llm
from services.tts import build_deepgram_tts


@pytest.mark.asyncio
async def test_deepgram_tts_streaming():
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        pytest.skip("DEEPGRAM_API_KEY not set")
    config = RuntimeConfig()
    service = build_deepgram_tts(config, api_key)
    chunks = []
    async for frame in service.run_tts("Testing one two three"):
        chunks.append(frame)
        from pipecat.frames.frames import TTSAudioRawFrame

        if isinstance(frame, TTSAudioRawFrame):
            assert frame.audio
            break
    assert chunks


@pytest.mark.asyncio
async def test_gemini_completion():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not set")
    config = RuntimeConfig()
    service = build_google_llm(config, api_key)
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

    context = OpenAILLMContext(messages=[{"role": "user", "content": "Hello"}])
    pair = service.create_context_aggregator(context)
    assistant = pair.assistant()
    await assistant.handle_aggregation("Hi there")
    assert assistant.messages

from pathlib import Path

import pytest

from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection

from app.filters import ActionExtractorFilter, ReasoningTraceFilter


@pytest.mark.asyncio
async def test_reasoning_trace_filter_strips_think_blocks_streamed():
    filter = ReasoningTraceFilter()
    utterances = [
        "<",
        "<think",
        "<think>secret",
        "<think>secret</think>",
        "<think>secret</think>Hello",
    ]

    collected: list[str] = []
    for text in utterances:
        frame = LLMTextFrame(text=text)
        await filter.process_frame(frame, FrameDirection.DOWNSTREAM)
        collected.append(frame.text)

    assert collected == ["", "", "", "", "Hello"]


@pytest.mark.asyncio
async def test_reasoning_trace_filter_prevents_action_extraction(tmp_path: Path):
    actions_path = tmp_path / "actions.txt"
    reasoning_filter = ReasoningTraceFilter()
    action_filter = ActionExtractorFilter(actions_path)

    frame = LLMTextFrame(text="<think><UNLOCK></think>Hi <LOCK>")
    await reasoning_filter.process_frame(frame, FrameDirection.DOWNSTREAM)
    await action_filter.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert frame.text == "Hi "
    assert actions_path.read_text().splitlines() == ["LOCK"]


@pytest.mark.asyncio
async def test_reasoning_trace_filter_preserves_leading_space_in_delta_stream():
    filter = ReasoningTraceFilter()
    chunks = ["Hello,", " world", " again"]
    collected: list[str] = []

    for chunk in chunks:
        frame = LLMTextFrame(text=chunk)
        await filter.process_frame(frame, FrameDirection.DOWNSTREAM)
        collected.append(frame.text)

    assert collected == ["Hello,", " world", " again"]


@pytest.mark.asyncio
async def test_reasoning_trace_filter_keeps_action_punctuation_cleanup(tmp_path: Path):
    actions_path = tmp_path / "actions.txt"
    reasoning_filter = ReasoningTraceFilter()
    action_filter = ActionExtractorFilter(actions_path)

    frame = LLMTextFrame(text="Wait!,<think>ignore</think><Make_Coffee>,Okay")
    await reasoning_filter.process_frame(frame, FrameDirection.DOWNSTREAM)
    await action_filter.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert frame.text == "Wait! Okay"
    assert actions_path.read_text().splitlines() == ["Make_Coffee"]

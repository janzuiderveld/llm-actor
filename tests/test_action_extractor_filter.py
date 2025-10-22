from pathlib import Path

import pytest

from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection

from app.filters import ActionExtractorFilter


@pytest.mark.asyncio
async def test_action_extractor_filters_streamed_directive(tmp_path: Path):
    actions_path = tmp_path / "actions.txt"
    filter = ActionExtractorFilter(actions_path)

    frame = LLMTextFrame(text="<")
    await filter.process_frame(frame, FrameDirection.DOWNSTREAM)
    assert frame.text == ""
    assert not actions_path.exists()

    frame = LLMTextFrame(text="< UNLOCK")
    await filter.process_frame(frame, FrameDirection.DOWNSTREAM)
    assert frame.text == ""
    assert not actions_path.exists()

    frame = LLMTextFrame(text="< UNLOCK>")
    await filter.process_frame(frame, FrameDirection.DOWNSTREAM)
    assert frame.text == ""
    assert actions_path.read_text().strip() == "UNLOCK"


@pytest.mark.asyncio
async def test_action_extractor_handles_delta_chunks(tmp_path: Path):
    actions_path = tmp_path / "actions.txt"
    filter = ActionExtractorFilter(actions_path)

    chunks = ["<", " UNLOCK", " 837 385>"]
    for chunk in chunks:
        frame = LLMTextFrame(text=chunk)
        await filter.process_frame(frame, FrameDirection.DOWNSTREAM)
        assert frame.text == ""

    assert actions_path.read_text().splitlines() == ["UNLOCK 837 385"]


@pytest.mark.asyncio
async def test_action_extractor_passthrough_regular_text(tmp_path: Path):
    actions_path = tmp_path / "actions.txt"
    filter = ActionExtractorFilter(actions_path)

    utterances = [
        "Hello there",
        "Hello there<",
        "Hello there<still talking",
        "Hello there<still talking>",
        " and again",
    ]

    expected_outputs = ["Hello there", "", "", "", " and again"]
    collected = []

    for text in utterances:
        frame = LLMTextFrame(text=text)
        await filter.process_frame(frame, FrameDirection.DOWNSTREAM)
        collected.append(frame.text)

    assert collected == expected_outputs
    contents = actions_path.read_text().splitlines()
    assert contents == ["still talking"]

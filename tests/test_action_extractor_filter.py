from pathlib import Path

import pytest

from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection

from app.filters import ActionExtractionState, ActionExtractorFilter


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


@pytest.mark.asyncio
async def test_action_extractor_preserves_leading_space_in_delta_stream(tmp_path: Path):
    actions_path = tmp_path / "actions.txt"
    filter = ActionExtractorFilter(actions_path)

    chunks = ["Hello,", " world", " again"]
    collected = []

    for chunk in chunks:
        frame = LLMTextFrame(text=chunk)
        await filter.process_frame(frame, FrameDirection.DOWNSTREAM)
        collected.append(frame.text)

    assert collected == ["Hello,", " world", " again"]


@pytest.mark.asyncio
async def test_action_extractor_tracks_raw_text_and_actions(tmp_path: Path):
    actions_path = tmp_path / "actions.txt"
    state = ActionExtractionState()
    filter = ActionExtractorFilter(actions_path, action_state=state)

    frame = LLMTextFrame(text="Good things take time. <Make_Coffee>")
    await filter.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert frame.text == "Good things take time. "
    assert state.raw_text == "Good things take time. <Make_Coffee>"
    assert state.actions == ["Make_Coffee"]


@pytest.mark.asyncio
async def test_action_extractor_drops_action_adjacent_punctuation(tmp_path: Path):
    actions_path = tmp_path / "actions.txt"
    filter = ActionExtractorFilter(actions_path)

    frame = LLMTextFrame(
        text="Good things drip slow, even mercy!,<Make_Coffee>,You paying, cleaner?"
    )
    await filter.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert frame.text == "Good things drip slow, even mercy! You paying, cleaner?"
    assert actions_path.read_text().splitlines() == ["Make_Coffee"]

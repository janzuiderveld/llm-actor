import asyncio

import pytest

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
)
from pipecat.processors.aggregators.llm_response import LLMAssistantAggregatorParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.pipeline.task import TaskManager, TaskManagerParams

from app.pipeline import AssistantAggregator
from pipecat.services.google.llm import GoogleLLMContext


class RecordingProcessor(FrameProcessor):
    def __init__(self):
        super().__init__(enable_direct_mode=True)
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self.frames.append(frame)


async def _run_case(
    sentences: list[str], partials: list[str] | None = None
) -> tuple[list[str], str]:
    context = GoogleLLMContext(messages=[])
    aggregated_messages: list[str] = []
    if partials is None:
        partial_cb = None
    else:
        partial_cb = partials.append
    aggregator = AssistantAggregator(
        context,
        params=LLMAssistantAggregatorParams(),
        on_message=aggregated_messages.append,
        on_partial=partial_cb,
    )
    recorder = RecordingProcessor()
    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    aggregator._task_manager = task_manager
    recorder._task_manager = task_manager
    aggregator.link(recorder)

    await aggregator.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
    await aggregator.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)

    for sentence in sentences:
        await aggregator.process_frame(LLMTextFrame(text=sentence), FrameDirection.DOWNSTREAM)

    await aggregator.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    await aggregator.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    await asyncio.sleep(0)

    streamed = [frame.text for frame in recorder.frames if isinstance(frame, LLMTextFrame)]
    assert len(aggregated_messages) == 1
    history = aggregated_messages[0]
    return streamed, history


@pytest.mark.asyncio
async def test_assistant_aggregator_streams_all_text():
    cases = [
        [
            "I apologize , but  I'm still not clear on what you're asking for when  you say \"output  and action.\"",
            " Could you explain it in a different way, or give me an example of what you're looking for?\"",
        ],
        ["1? 2 ? 3? 4? 5? 6? 7? 8? 9? 10?"],
    ]

    for sentences in cases:
        partials: list[str] = []
        streamed, history = await _run_case(sentences, partials)
        assert streamed == sentences
        expected_full = "".join(sentences).strip()
        assert _normalize(history) == _normalize(expected_full)
        if sentences:
            assert partials == sentences


def _normalize(text: str) -> str:
    return " ".join(text.split())

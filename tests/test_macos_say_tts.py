import asyncio
import os
import shutil
import sys
import time

import pytest

from pipecat.frames.frames import (
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.task import TaskManager, TaskManagerParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.processors.aggregators.llm_response import LLMAssistantAggregatorParams
from pipecat.services.google.llm import GoogleLLMContext
from pipecat.clocks.system_clock import SystemClock

from app.filters import ActionExtractionState
from app.pipeline import SpokenAssistantAggregator
from services.macos_say_tts import MacosSayTTSService, _parse_say_interactive_render


class RecordingProcessor(FrameProcessor):
    def __init__(self):
        super().__init__(enable_direct_mode=True)
        self.frames = []

    async def process_frame(self, frame, direction: FrameDirection):  # type: ignore[override]
        await super().process_frame(frame, direction)
        self.frames.append(frame)


def test_parse_say_interactive_render_extracts_highlight():
    render = (
        "\x1b[M"
        "\x1b[7mHello\x1b(B\x1b[m, world!"
        "\x1b[?12l\x1b[?25h"
    )
    parsed = _parse_say_interactive_render(render)
    assert parsed.text == "Hello, world!"
    assert parsed.highlight == (0, 5)


@pytest.mark.asyncio
async def test_spoken_assistant_aggregator_appends_cutoff_marker_on_interruption():
    context = GoogleLLMContext(messages=[])
    received: list[str] = []
    aggregator = SpokenAssistantAggregator(
        context,
        params=LLMAssistantAggregatorParams(expect_stripped_words=False),
        on_message=received.append,
        cutoff_marker="[.. cut off by user utterance]",
    )
    recorder = RecordingProcessor()

    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    aggregator._task_manager = task_manager
    recorder._task_manager = task_manager
    aggregator.link(recorder)

    await aggregator.process_frame(StartFrame(allow_interruptions=True), FrameDirection.DOWNSTREAM)
    await aggregator.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await aggregator.process_frame(TTSTextFrame("Hello, "), FrameDirection.DOWNSTREAM)
    await aggregator.process_frame(TTSTextFrame("world!"), FrameDirection.DOWNSTREAM)
    await aggregator.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    assert received == ["Hello, world! [.. cut off by user utterance]"]


@pytest.mark.asyncio
async def test_spoken_assistant_aggregator_updates_context_on_partials():
    context = GoogleLLMContext(messages=[])
    partials: list[str] = []
    aggregator = SpokenAssistantAggregator(
        context,
        params=LLMAssistantAggregatorParams(expect_stripped_words=False),
        on_partial=partials.append,
    )
    recorder = RecordingProcessor()

    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    aggregator._task_manager = task_manager
    recorder._task_manager = task_manager
    aggregator.link(recorder)

    await aggregator.process_frame(StartFrame(allow_interruptions=True), FrameDirection.DOWNSTREAM)
    await aggregator.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await aggregator.process_frame(TTSTextFrame("Hello, "), FrameDirection.DOWNSTREAM)
    await aggregator.process_frame(TTSTextFrame("world!"), FrameDirection.DOWNSTREAM)
    await aggregator.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    assert partials == ["Hello, ", "Hello, world!"]
    assert len(context.messages) == 1
    message = context.messages[0]
    assert message.role == "model"
    assert message.parts[0].text == "Hello, world!"


@pytest.mark.asyncio
async def test_spoken_assistant_aggregator_prefers_action_state_for_history():
    context = GoogleLLMContext(messages=[])
    received: list[str] = []
    action_state = ActionExtractionState(
        raw_text="Good things take time. <Make_Coffee>",
        actions=["Make_Coffee"],
    )
    aggregator = SpokenAssistantAggregator(
        context,
        params=LLMAssistantAggregatorParams(expect_stripped_words=False),
        on_message=received.append,
        action_state=action_state,
    )
    recorder = RecordingProcessor()

    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    aggregator._task_manager = task_manager
    recorder._task_manager = task_manager
    aggregator.link(recorder)

    await aggregator.process_frame(StartFrame(allow_interruptions=True), FrameDirection.DOWNSTREAM)
    await aggregator.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await aggregator.process_frame(TTSTextFrame("Good things take time."), FrameDirection.DOWNSTREAM)
    await aggregator.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    assert received == ["Good things take time. <Make_Coffee>"]
    assert len(context.messages) == 1
    message = context.messages[0]
    assert message.role == "model"
    assert message.parts[0].text == "Good things take time. <Make_Coffee>"


@pytest.mark.asyncio
async def test_spoken_assistant_aggregator_preserves_raw_text_when_tts_cleaned():
    context = GoogleLLMContext(messages=[])
    received: list[str] = []
    raw_text = "Good things drip slow, even mercy!,<Make_Coffee>,You paying, cleaner?"
    action_state = ActionExtractionState(raw_text=raw_text, actions=["Make_Coffee"])
    aggregator = SpokenAssistantAggregator(
        context,
        params=LLMAssistantAggregatorParams(expect_stripped_words=False),
        on_message=received.append,
        action_state=action_state,
    )
    recorder = RecordingProcessor()

    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    aggregator._task_manager = task_manager
    recorder._task_manager = task_manager
    aggregator.link(recorder)

    await aggregator.process_frame(StartFrame(allow_interruptions=True), FrameDirection.DOWNSTREAM)
    await aggregator.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await aggregator.process_frame(
        TTSTextFrame("Good things drip slow, even mercy! You paying, cleaner?"),
        FrameDirection.DOWNSTREAM,
    )
    await aggregator.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    assert received == [raw_text]
    assert len(context.messages) == 1
    message = context.messages[0]
    assert message.role == "model"
    assert message.parts[0].text == raw_text


@pytest.mark.asyncio
async def test_macos_say_tts_latency_and_interrupt():
    if sys.platform != "darwin":
        pytest.skip("macOS-only test")
    if shutil.which("say") is None:
        pytest.skip("macOS `say` not available")
    if os.getenv("RUN_SAY_TESTS") != "1":
        pytest.skip("Set RUN_SAY_TESTS=1 to run macOS say integration test (plays audio)")

    voice = os.getenv("SAY_TEST_VOICE", "Alex")

    service = MacosSayTTSService(voice=voice, interactive=True)
    recorder = RecordingProcessor()
    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    clock = SystemClock()
    await service.setup(FrameProcessorSetup(clock=clock, task_manager=task_manager))
    await recorder.setup(FrameProcessorSetup(clock=clock, task_manager=task_manager))
    service.link(recorder)

    await service.process_frame(StartFrame(allow_interruptions=True), FrameDirection.DOWNSTREAM)

    text = "Hello, world! This is a long utterance used to validate interruption handling. " * 4

    speak_task = asyncio.create_task(service.queue_frame(TTSSpeakFrame(text), FrameDirection.DOWNSTREAM))

    start = time.monotonic()
    while True:
        if any(isinstance(frame, TTSTextFrame) for frame in recorder.frames):
            break
        if time.monotonic() - start > 1.5:
            break
        await asyncio.sleep(0.01)

    interrupt_at = time.monotonic()
    await service.queue_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    while True:
        if any(isinstance(frame, TTSStoppedFrame) for frame in recorder.frames):
            break
        if time.monotonic() - interrupt_at > 2.0:
            break
        await asyncio.sleep(0.01)

    speak_task.cancel()

    stopped_frames = [frame for frame in recorder.frames if isinstance(frame, TTSStoppedFrame)]
    assert stopped_frames, "Expected a TTSStoppedFrame after interruption"

    stopped = stopped_frames[-1]
    assert stopped.metadata.get("tts_interrupted") is True

    latency_ms = stopped.metadata.get("tts_first_highlight_ms")
    assert latency_ms is not None
    assert latency_ms < 1000.0

    stop_latency_s = time.monotonic() - interrupt_at
    assert stop_latency_s < 1.0

    spoken = "".join(frame.text for frame in recorder.frames if isinstance(frame, TTSTextFrame)).strip()
    assert spoken
    assert text.startswith(spoken)

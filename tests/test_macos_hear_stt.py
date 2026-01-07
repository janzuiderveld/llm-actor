import asyncio
import sys
from typing import Type

import pytest

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    InterruptionTaskFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.task import TaskManager, TaskManagerParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup

from services.macos_hear_stt import HearSTTOptions, MacosHearSTTService


class RecordingProcessor(FrameProcessor):
    def __init__(self):
        super().__init__(enable_direct_mode=True)
        self.frames: list[Frame] = []

    async def process_frame(self, frame, direction: FrameDirection):  # type: ignore[override]
        await super().process_frame(frame, direction)
        self.frames.append(frame)


class InterruptionResponder(FrameProcessor):
    """Minimal pipeline-task stand-in for `push_interruption_task_frame_and_wait()`."""

    def __init__(self):
        super().__init__(enable_direct_mode=True)

    async def process_frame(self, frame, direction: FrameDirection):  # type: ignore[override]
        await super().process_frame(frame, direction)
        if direction == FrameDirection.UPSTREAM and isinstance(frame, InterruptionTaskFrame):
            await self.push_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
            return
        await self.push_frame(frame, direction)


async def _wait_for_frame(recorder: RecordingProcessor, frame_type: Type[Frame], timeout: float = 1.0) -> None:
    start = asyncio.get_running_loop().time()
    while True:
        if any(isinstance(frame, frame_type) for frame in recorder.frames):
            return
        if asyncio.get_running_loop().time() - start > timeout:
            raise AssertionError(f"Timed out waiting for {frame_type.__name__}")
        await asyncio.sleep(0.005)


def _python_emitter(lines: list[str], *, delay_s: float = 0.01, tail_sleep_s: float = 1.0) -> list[str]:
    quoted = "[" + ", ".join(repr(line) for line in lines) + "]"
    code = (
        "import time\n"
        f"lines={quoted}\n"
        f"delay={delay_s!r}\n"
        "for line in lines:\n"
        "    print(line, flush=True)\n"
        "    time.sleep(delay)\n"
        f"time.sleep({tail_sleep_s!r})\n"
    )
    return [sys.executable, "-u", "-c", code]


@pytest.mark.asyncio
async def test_hear_stt_emits_only_final_transcription_after_silence():
    options = HearSTTOptions(final_silence_sec=0.05, restart_on_final=False)
    command = _python_emitter(
        [
            "Hello",
            "Hello can",
            "Hello can you hear me how are you doing today.",
        ],
        delay_s=0.01,
        tail_sleep_s=1.0,
    )

    service = MacosHearSTTService(options=options, command=command, check_platform=False)
    responder = InterruptionResponder()
    recorder = RecordingProcessor()

    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    clock = SystemClock()

    await responder.setup(FrameProcessorSetup(clock=clock, task_manager=task_manager))
    await service.setup(FrameProcessorSetup(clock=clock, task_manager=task_manager))
    await recorder.setup(FrameProcessorSetup(clock=clock, task_manager=task_manager))

    responder.link(service)
    service.link(recorder)

    await responder.process_frame(StartFrame(allow_interruptions=True), FrameDirection.DOWNSTREAM)

    await _wait_for_frame(recorder, TranscriptionFrame, timeout=1.0)

    transcripts = [frame for frame in recorder.frames if isinstance(frame, TranscriptionFrame)]
    assert len(transcripts) == 1
    assert transcripts[0].text == "Hello can you hear me how are you doing today."

    assert any(isinstance(frame, UserStartedSpeakingFrame) for frame in recorder.frames)
    assert any(isinstance(frame, UserStoppedSpeakingFrame) for frame in recorder.frames)

    await service.cleanup()
    await recorder.cleanup()
    await responder.cleanup()


@pytest.mark.asyncio
async def test_hear_stt_interrupts_on_first_hypothesis():
    options = HearSTTOptions(final_silence_sec=0.2, restart_on_final=False)
    command = _python_emitter(["Hello"], delay_s=0.01, tail_sleep_s=1.0)

    service = MacosHearSTTService(options=options, command=command, check_platform=False)
    responder = InterruptionResponder()
    recorder = RecordingProcessor()

    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    clock = SystemClock()

    await responder.setup(FrameProcessorSetup(clock=clock, task_manager=task_manager))
    await service.setup(FrameProcessorSetup(clock=clock, task_manager=task_manager))
    await recorder.setup(FrameProcessorSetup(clock=clock, task_manager=task_manager))

    responder.link(service)
    service.link(recorder)

    await responder.process_frame(StartFrame(allow_interruptions=True), FrameDirection.DOWNSTREAM)

    await _wait_for_frame(recorder, InterruptionFrame, timeout=1.0)
    await _wait_for_frame(recorder, UserStartedSpeakingFrame, timeout=1.0)

    first_interrupt = next(i for i, frame in enumerate(recorder.frames) if isinstance(frame, InterruptionFrame))
    first_started = next(i for i, frame in enumerate(recorder.frames) if isinstance(frame, UserStartedSpeakingFrame))
    assert first_interrupt < first_started

    await service.cleanup()
    await recorder.cleanup()
    await responder.cleanup()


@pytest.mark.asyncio
async def test_hear_stt_restarts_after_final_and_keeps_listening():
    options = HearSTTOptions(final_silence_sec=0.03, restart_on_final=True)
    command = _python_emitter(["Hi there."], delay_s=0.005, tail_sleep_s=10.0)

    service = MacosHearSTTService(options=options, command=command, check_platform=False)
    responder = InterruptionResponder()
    recorder = RecordingProcessor()

    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    clock = SystemClock()

    await responder.setup(FrameProcessorSetup(clock=clock, task_manager=task_manager))
    await service.setup(FrameProcessorSetup(clock=clock, task_manager=task_manager))
    await recorder.setup(FrameProcessorSetup(clock=clock, task_manager=task_manager))

    responder.link(service)
    service.link(recorder)

    await responder.process_frame(StartFrame(allow_interruptions=True), FrameDirection.DOWNSTREAM)

    start = asyncio.get_running_loop().time()
    while True:
        transcripts = [frame for frame in recorder.frames if isinstance(frame, TranscriptionFrame)]
        if len(transcripts) >= 2:
            break
        if asyncio.get_running_loop().time() - start > 1.0:
            raise AssertionError("Timed out waiting for second transcription after restart")
        await asyncio.sleep(0.01)

    await service.cleanup()
    await recorder.cleanup()
    await responder.cleanup()


@pytest.mark.asyncio
async def test_hear_stt_keep_mic_open_disables_restart():
    options = HearSTTOptions(final_silence_sec=0.03, restart_on_final=True, keep_mic_open=True)
    command = _python_emitter(["Hi there."], delay_s=0.005, tail_sleep_s=1.0)

    service = MacosHearSTTService(options=options, command=command, check_platform=False)
    responder = InterruptionResponder()
    recorder = RecordingProcessor()

    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    clock = SystemClock()

    await responder.setup(FrameProcessorSetup(clock=clock, task_manager=task_manager))
    await service.setup(FrameProcessorSetup(clock=clock, task_manager=task_manager))
    await recorder.setup(FrameProcessorSetup(clock=clock, task_manager=task_manager))

    responder.link(service)
    service.link(recorder)

    await responder.process_frame(StartFrame(allow_interruptions=True), FrameDirection.DOWNSTREAM)
    await _wait_for_frame(recorder, TranscriptionFrame, timeout=1.0)

    first_count = len([frame for frame in recorder.frames if isinstance(frame, TranscriptionFrame)])
    assert first_count == 1

    await asyncio.sleep(0.25)
    second_count = len([frame for frame in recorder.frames if isinstance(frame, TranscriptionFrame)])
    assert second_count == 1

    await service.cleanup()
    await recorder.cleanup()
    await responder.cleanup()

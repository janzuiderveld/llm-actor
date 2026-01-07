import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipecat.frames.frames import LLMRunFrame

from app.pipeline import VoicePipelineController


@pytest.mark.asyncio
async def test_inbox_burst_coalesces_llm_runs(tmp_path: Path) -> None:
    dummy_config = SimpleNamespace(config=None)
    dummy_session = SimpleNamespace(session_name="test")
    dummy_history = SimpleNamespace(add=lambda *args, **kwargs: None)
    dummy_logger = SimpleNamespace(emit=lambda *args, **kwargs: None)
    dummy_metrics = SimpleNamespace(marks={}, mark=lambda *args, **kwargs: None)

    controller = VoicePipelineController(
        dummy_config,
        dummy_session,
        dummy_history,
        dummy_logger,
        dummy_metrics,
        actions_path=tmp_path / "actions.txt",
        inbox_path=tmp_path / "inbox.txt",
        params_path=tmp_path / "params.ndjson",
    )

    controller._inbox_debounce_s = 0.01

    class DummyContext:
        def __init__(self) -> None:
            self.messages = []

        def add_messages(self, messages) -> None:
            self.messages.extend(messages)

    class DummyAggregator:
        def __init__(self) -> None:
            self.context = DummyContext()

    class DummyTask:
        def __init__(self) -> None:
            self.frames = []

        async def queue_frame(self, frame) -> None:
            self.frames.append(frame)

    controller._user_aggregator = DummyAggregator()
    controller._components = SimpleNamespace(task=DummyTask())

    seen_user_messages = []

    async def fake_on_user_message(text: str) -> None:
        seen_user_messages.append(text)

    controller._on_user_message = fake_on_user_message  # type: ignore[assignment]

    await asyncio.gather(
        controller._handle_inbox_push("A"),
        controller._handle_inbox_push("B"),
    )

    await asyncio.sleep(0.05)

    assert seen_user_messages == ["A", "B"]
    assert controller._user_aggregator.context.messages == [
        {"role": "user", "content": "A"},
        {"role": "user", "content": "B"},
    ]
    run_frames = [frame for frame in controller._components.task.frames if isinstance(frame, LLMRunFrame)]
    assert len(run_frames) == 1


@pytest.mark.asyncio
async def test_inbox_run_waits_for_llm_idle(tmp_path: Path) -> None:
    dummy_config = SimpleNamespace(config=None)
    dummy_session = SimpleNamespace(session_name="test")
    dummy_history = SimpleNamespace(add=lambda *args, **kwargs: None)
    dummy_logger = SimpleNamespace(emit=lambda *args, **kwargs: None)
    dummy_metrics = SimpleNamespace(marks={}, mark=lambda *args, **kwargs: None)

    controller = VoicePipelineController(
        dummy_config,
        dummy_session,
        dummy_history,
        dummy_logger,
        dummy_metrics,
        actions_path=tmp_path / "actions.txt",
        inbox_path=tmp_path / "inbox.txt",
        params_path=tmp_path / "params.ndjson",
    )

    class DummyContext:
        def __init__(self) -> None:
            self.messages = []

        def add_messages(self, messages) -> None:
            self.messages.extend(messages)

    class DummyAggregator:
        def __init__(self) -> None:
            self.context = DummyContext()

    class DummyTask:
        def __init__(self) -> None:
            self.frames = []

        async def queue_frame(self, frame) -> None:
            self.frames.append(frame)

    controller._user_aggregator = DummyAggregator()
    controller._components = SimpleNamespace(task=DummyTask())
    controller._inbox_debounce_s = 0.0
    controller._llm_busy = True

    async def fake_on_user_message(text: str) -> None:
        return None

    controller._on_user_message = fake_on_user_message  # type: ignore[assignment]

    await controller._handle_inbox_push("ping")
    await asyncio.sleep(0)

    controller._notify_llm_response_end()
    await asyncio.sleep(0)

    run_frames = [frame for frame in controller._components.task.frames if isinstance(frame, LLMRunFrame)]
    assert len(run_frames) == 1

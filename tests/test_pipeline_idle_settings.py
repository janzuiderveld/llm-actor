import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

import app.pipeline as pipeline_module
import services.llm as llm_module
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    InterimTranscriptionFrame,
    STTMuteFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from app.config import ConfigManager, RuntimeConfig
from app.history import ConversationHistory
from app.logging_io import EventLogger
from app.metrics import MetricsTracker
from app.pipeline import VoicePipelineController
from app.session import create_session
from projects.utils import apply_runtime_config_overrides


@dataclass
class DummyComponents:
    task: object
    pipeline: object = None
    runner: object = None
    inbox_watcher: object = None
    params_watcher: object = None


class DummyTask:
    def __init__(self) -> None:
        self.frames: list[object] = []

    async def queue_frame(self, frame: object) -> None:
        self.frames.append(frame)


def _build_controller(tmp_path: Path) -> VoicePipelineController:
    config_path = tmp_path / "config.json"
    manager = ConfigManager(path=config_path)
    session_paths = create_session(tmp_path, manager.config, session_name="test-session")
    history = ConversationHistory(
        session_paths.transcript,
        clean_transcript_path=session_paths.llm_transcript,
        max_messages=manager.config.pipeline.max_history_messages,
    )
    event_logger = EventLogger(session_paths.event_log)
    metrics = MetricsTracker(event_logger)
    return VoicePipelineController(
        manager,
        session_paths,
        history,
        event_logger,
        metrics,
        actions_path=tmp_path / "actions.txt",
        inbox_path=tmp_path / "inbox.txt",
        params_path=tmp_path / "params.ndjson",
    )


def test_pipeline_config_defaults() -> None:
    config = RuntimeConfig()
    assert config.pipeline.idle_timeout_secs == 300
    assert config.pipeline.cancel_on_idle_timeout is True
    assert config.pipeline.pause_stt_on_idle is False
    assert config.pipeline.history_on_idle == "keep"
    assert config.pipeline.max_history_messages == 50
    assert config.llm.request_timeout_s == 30.0


def test_idle_timeout_frames_include_stt_tts_events(tmp_path: Path) -> None:
    controller = _build_controller(tmp_path)
    frames = controller._idle_timeout_frames()
    assert UserStartedSpeakingFrame in frames
    assert UserStoppedSpeakingFrame in frames
    assert TranscriptionFrame in frames
    assert InterimTranscriptionFrame in frames
    assert TTSStartedFrame in frames
    assert TTSAudioRawFrame in frames
    assert TTSStoppedFrame in frames


def test_apply_runtime_config_overrides_updates_pipeline(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    ConfigManager(path=config_path)

    apply_runtime_config_overrides(
        {
            "pipeline": {
                "idle_timeout_secs": 12,
                "cancel_on_idle_timeout": False,
                "pause_stt_on_idle": True,
                "history_on_idle": "reset",
                "max_history_messages": 12,
            }
        },
        config_path=config_path,
    )

    updated = ConfigManager(path=config_path).config.pipeline
    assert updated.idle_timeout_secs == 12
    assert updated.cancel_on_idle_timeout is False
    assert updated.pause_stt_on_idle is True
    assert updated.history_on_idle == "reset"
    assert updated.max_history_messages == 12


@pytest.mark.asyncio
async def test_set_stt_muted_queues_frame(tmp_path: Path) -> None:
    controller = _build_controller(tmp_path)
    task = DummyTask()
    controller._components = DummyComponents(task=task)
    controller._stt_service = object()

    await controller._set_stt_muted(True)

    assert len(task.frames) == 1
    assert isinstance(task.frames[0], STTMuteFrame)
    assert task.frames[0].mute is True


def test_request_stt_resume_schedules_when_muted(tmp_path: Path) -> None:
    controller = _build_controller(tmp_path)
    scheduled: list[tuple[object, tuple[object, ...]]] = []

    def fake_schedule(func, *args):
        scheduled.append((func, args))

    controller._schedule_coroutine = fake_schedule  # type: ignore[assignment]
    controller._pause_stt_on_idle = True
    controller._stt_muted = True

    controller._request_stt_resume()

    assert scheduled == [(controller._set_stt_muted, (False,))]


def test_history_max_messages_limits_buffer(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    manager = ConfigManager(path=config_path)
    manager.apply_updates(pipeline={"max_history_messages": 2})
    session_paths = create_session(tmp_path, manager.config, session_name="test-session")
    history = ConversationHistory(
        session_paths.transcript,
        clean_transcript_path=session_paths.llm_transcript,
        max_messages=manager.config.pipeline.max_history_messages,
    )

    history.add("user", "one")
    history.add("assistant", "two")
    history.add("user", "three")

    exported = history.export()
    assert [entry["content"] for entry in exported] == ["two", "three"]


@pytest.mark.asyncio
async def test_idle_timeout_resets_history_when_configured(tmp_path: Path) -> None:
    controller = _build_controller(tmp_path)
    controller._history.set_system_message("Stay helpful.")
    controller._history.add("user", "Hello")
    controller._history_on_idle = "reset"
    context = OpenAILLMContext(
        messages=[
            {"role": "system", "content": "Stay helpful."},
            {"role": "user", "content": "Hello"},
        ]
    )

    class DummyAggregator:
        def __init__(self, context_obj):
            self.context = context_obj

    dummy = DummyAggregator(context)
    controller._user_aggregator = dummy
    controller._assistant_aggregator = dummy

    await controller._handle_idle_timeout()

    assert controller._history.export() == [{"role": "system", "content": "Stay helpful."}]
    assert context.get_messages() == [{"role": "system", "content": "Stay helpful."}]


@pytest.mark.asyncio
async def test_inbox_push_resets_llm_busy(tmp_path: Path) -> None:
    controller = _build_controller(tmp_path)
    controller._llm_busy = True
    controller._llm_busy_since = time.time() - 5
    scheduled: list[object] = []

    async def fake_interrupt() -> None:
        return None

    async def fake_schedule(delay: object = None) -> None:
        scheduled.append(delay)

    controller._interrupt_assistant_if_needed = fake_interrupt  # type: ignore[assignment]
    controller._schedule_inbox_run = fake_schedule  # type: ignore[assignment]

    await controller._handle_inbox_push("Hello")

    assert controller._llm_busy is False
    assert controller._llm_busy_since is None
    assert len(controller._pending_inbox) == 1
    assert scheduled == [None]


@pytest.mark.asyncio
async def test_inbox_push_queues_before_run(tmp_path: Path) -> None:
    controller = _build_controller(tmp_path)
    calls: list[tuple[str, str]] = []

    async def fake_append(text: str) -> None:
        calls.append(("append", text))

    async def fake_queue() -> None:
        calls.append(("run", ""))

    controller._append_inbox_message = fake_append  # type: ignore[assignment]
    controller._queue_llm_run = fake_queue  # type: ignore[assignment]

    await controller._handle_inbox_push("First")
    await controller._handle_inbox_push("Second")

    assert len(controller._pending_inbox) == 2
    assert calls == []

    await controller._trigger_inbox_run()

    assert controller._pending_inbox_current is not None
    assert controller._pending_inbox_current.text == "First"
    assert calls == [("append", "First"), ("run", "")]


@pytest.mark.asyncio
async def test_queue_llm_run_refreshes_clean_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _build_controller(tmp_path)
    controller._config_manager.apply_updates(llm={"system_prompt": "Time {clean_time}."})

    class DummyContext:
        def __init__(self) -> None:
            self.system_message = None

    class DummyAggregator:
        def __init__(self) -> None:
            self.context = DummyContext()

    class DummyLLM:
        pass

    controller._llm_service = DummyLLM()
    controller._assistant_aggregator = DummyAggregator()
    controller._user_aggregator = DummyAggregator()
    controller._components = DummyComponents(task=DummyTask())

    monkeypatch.setattr(
        pipeline_module.prompt_templates,
        "render_clean_time",
        lambda template: template.replace("{clean_time}", "2024-01-01 12:00:00"),
    )

    await controller._queue_llm_run()

    assert controller._assistant_aggregator.context.system_message == "Time 2024-01-01 12:00:00."
    assert controller._user_aggregator.context.system_message == "Time 2024-01-01 12:00:00."
    assert getattr(controller._llm_service, "_system_instruction") == "Time 2024-01-01 12:00:00."


def test_build_google_llm_uses_request_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class DummyGoogleLLM:
        class InputParams:
            def __init__(self, *args: object, **kwargs: object) -> None:
                return None

        def __init__(self, **kwargs: object) -> None:
            captured["http_options"] = kwargs.get("http_options")

    monkeypatch.setattr(llm_module, "GoogleLLMService", DummyGoogleLLM)
    monkeypatch.setattr(llm_module, "_build_google_http_options", lambda value: "opts")
    config = RuntimeConfig()
    config.llm.request_timeout_s = 12.0

    llm_module.build_google_llm(config, api_key="key")

    assert captured["http_options"] == "opts"


def test_google_http_options_timeout_converts_to_ms() -> None:
    options = llm_module._build_google_http_options(12.5)
    assert options is not None
    assert options.timeout == 12500


def test_build_pipeline_does_not_gate_stt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    controller = _build_controller(tmp_path)
    dummy_input = object()
    dummy_output = object()

    class DummyTransport:
        def __init__(self, params: object) -> None:
            self.params = params

        def input(self) -> object:
            return dummy_input

        def output(self) -> object:
            return dummy_output

    class DummyPipeline:
        def __init__(self, processors: list[object]) -> None:
            self.processors = processors

    class DummyContext:
        def __init__(self) -> None:
            self.context = object()

    class DummyContextPair:
        def user(self) -> DummyContext:
            return DummyContext()

        def assistant(self) -> DummyContext:
            return DummyContext()

    class DummyAggregator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

    class DummyLLM:
        def event_handler(self, _name: str):
            def decorator(func):
                return func

            return decorator

    monkeypatch.setattr(pipeline_module, "get_api_keys", lambda: {"deepgram": "key", "google": "", "openai": ""})
    monkeypatch.setattr(
        pipeline_module,
        "build_stt_service",
        lambda config, deepgram_api_key: (object(), "deepgram_flux"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_llm_service",
        lambda config, google_api_key, openai_api_key: (DummyLLM(), "dummy"),
    )
    monkeypatch.setattr(pipeline_module, "build_tts_service", lambda config, deepgram_key: object())
    monkeypatch.setattr(pipeline_module, "ensure_devices_selected", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline_module, "LocalAudioTransport", DummyTransport)
    monkeypatch.setattr(pipeline_module, "create_llm_context", lambda llm, history: DummyContextPair())
    monkeypatch.setattr(pipeline_module, "UserAggregator", DummyAggregator)
    monkeypatch.setattr(pipeline_module, "OpenAIUserAggregator", DummyAggregator)
    monkeypatch.setattr(pipeline_module, "SpokenAssistantAggregator", DummyAggregator)
    monkeypatch.setattr(pipeline_module, "OpenAISpokenAssistantAggregator", DummyAggregator)
    monkeypatch.setattr(pipeline_module, "Pipeline", DummyPipeline)

    pipeline_obj = controller._build_pipeline(RuntimeConfig())

    assert controller._speech_gate is None
    assert dummy_input in pipeline_obj.processors
    assert dummy_output in pipeline_obj.processors
    assert not any(
        isinstance(processor, pipeline_module.AssistantSpeechGate)
        for processor in pipeline_obj.processors
    )


def test_llm_end_requeues_on_empty_response(tmp_path: Path) -> None:
    controller = _build_controller(tmp_path)
    controller._pending_inbox_current = pipeline_module.PendingInboxItem(text="Brew")
    controller._pending_inbox_started_at = time.time() - 2
    controller._pending_inbox_max_attempts = 2
    controller._llm_text_seen_since_start = False
    scheduled: list[tuple[object, tuple[object, ...]]] = []

    def fake_schedule(func, *args):
        scheduled.append((func, args))

    controller._schedule_coroutine = fake_schedule  # type: ignore[assignment]

    controller._notify_llm_response_end()

    assert controller._pending_inbox_current is None
    assert len(controller._pending_inbox) == 1
    assert controller._pending_inbox[0].attempts == 1
    assert scheduled


@pytest.mark.asyncio
async def test_inbox_timeout_requeues_pending(tmp_path: Path) -> None:
    controller = _build_controller(tmp_path)
    controller._pending_inbox_timeout_s = 0.01
    controller._pending_inbox_max_attempts = 2
    controller._schedule_coroutine = lambda *args, **kwargs: None  # type: ignore[assignment]
    controller._pending_inbox_current = pipeline_module.PendingInboxItem(text="Brew")
    controller._pending_inbox_started_at = time.time() - 1
    controller._llm_text_seen_since_start = False

    controller._start_inbox_timeout()
    await asyncio.sleep(0.05)

    assert controller._pending_inbox_current is None
    assert len(controller._pending_inbox) == 1
    assert controller._pending_inbox[0].attempts == 1


@pytest.mark.asyncio
async def test_llm_response_monitor_calls_callbacks() -> None:
    calls: list[str] = []

    async def on_start() -> None:
        calls.append("start")

    async def on_text() -> None:
        calls.append("text")

    async def on_end() -> None:
        calls.append("end")

    monitor = pipeline_module.LLMResponseMonitor(
        on_start=on_start,
        on_text=on_text,
        on_end=on_end,
    )

    await monitor.process_frame(LLMFullResponseStartFrame(), pipeline_module.FrameDirection.DOWNSTREAM)
    await monitor.process_frame(LLMTextFrame(text="Hello"), pipeline_module.FrameDirection.DOWNSTREAM)
    await monitor.process_frame(LLMFullResponseEndFrame(), pipeline_module.FrameDirection.DOWNSTREAM)

    assert calls == ["start", "text", "end"]

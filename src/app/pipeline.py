from __future__ import annotations

import asyncio
import inspect
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from pipecat.frames.frames import (
    AudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    LLMRunFrame,
    STTMuteFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TextFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from openai import NOT_GIVEN
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import LLMAssistantAggregatorParams, LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.google.llm import (
    Content,
    GoogleAssistantContextAggregator,
    GoogleUserContextAggregator,
    Part,
)
from pipecat.services.openai.llm import OpenAIAssistantContextAggregator, OpenAIUserContextAggregator
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from app.audio_events import should_log_tts_audio_event
from app.config import DEFAULT_TTS_PROVIDER, ConfigManager, RuntimeConfig, get_api_keys
from app.devices import ensure_audio_device_preferences, ensure_devices_selected
from app.filters import ActionExtractionState, ActionExtractorFilter, ReasoningTraceFilter, STTStandaloneIFilter
from app.history import ConversationHistory
from app.inbox_watch import InboxWatcher
from app.logging_io import EventLogger
from app.metrics import MetricsTracker
from app.params_apply import ParamsWatcher
from app import prompt_templates
from app.session import SessionPaths, new_session
from services.llm import build_llm_service, create_llm_context, normalize_model_for_service, parse_llm_model_spec
from services.stt import build_stt_service
from services.tts import build_tts_service

UserCallback = Callable[[str], Awaitable[None]]
LLM_TEXT_IS_TEXTFRAME = issubclass(LLMTextFrame, TextFrame)


def _normalize_history_on_idle(value: object) -> str:
    if not isinstance(value, str):
        return "keep"
    normalized = value.strip().lower()
    if normalized in {"reset", "clear", "delete"}:
        return "reset"
    if normalized == "keep":
        return "keep"
    return "keep"


def _coerce_max_history_messages(value: object, default: int = 50) -> int:
    try:
        max_messages = int(value)
    except (TypeError, ValueError):
        return default
    return max_messages if max_messages > 0 else default


class UserAggregator(GoogleUserContextAggregator):
    def __init__(
        self,
        *args,
        on_message: Optional[UserCallback] = None,
        transform: Optional[Callable[[str], str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._on_message = on_message
        self._transform = transform

    async def handle_aggregation(self, aggregation: str):  # type: ignore[override]
        if self._transform:
            aggregation = self._transform(aggregation)
        await super().handle_aggregation(aggregation)
        if self._on_message and aggregation:
            await self._on_message(aggregation)


class OpenAIUserAggregator(OpenAIUserContextAggregator):
    def __init__(
        self,
        *args,
        on_message: Optional[UserCallback] = None,
        transform: Optional[Callable[[str], str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._on_message = on_message
        self._transform = transform

    async def handle_aggregation(self, aggregation: str):  # type: ignore[override]
        if self._transform:
            aggregation = self._transform(aggregation)
        await super().handle_aggregation(aggregation)
        if self._on_message and aggregation:
            await self._on_message(aggregation)


class AssistantAggregator(GoogleAssistantContextAggregator):
    def __init__(
        self,
        *args,
        on_message: Optional[UserCallback] = None,
        on_partial: Optional[UserCallback] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._on_message = on_message
        self._on_partial = on_partial

    async def handle_aggregation(self, aggregation: str):  # type: ignore[override]
        await super().handle_aggregation(aggregation)
        clean_text = aggregation.strip()
        if self._on_message and clean_text:
            result = self._on_message(clean_text)
            if inspect.isawaitable(result):
                await result

    async def _handle_text(self, frame: TextFrame):  # type: ignore[override]
        await super()._handle_text(frame)
        if not getattr(self, "_started", 0):
            return
        text = frame.text
        if not text or not text.strip():
            return
        if self._on_partial:
            result = self._on_partial(text)
            if inspect.isawaitable(result):
                await result
        if isinstance(frame, LLMTextFrame):
            await self.push_frame(LLMTextFrame(text=text))
            if not LLM_TEXT_IS_TEXTFRAME:
                await self.push_frame(TextFrame(text=text))

    async def _handle_llm_start(self, frame: LLMFullResponseStartFrame):  # type: ignore[override]
        await super()._handle_llm_start(frame)
        await self.push_frame(frame)

    async def _handle_llm_end(self, frame: LLMFullResponseEndFrame):  # type: ignore[override]
        await super()._handle_llm_end(frame)
        await self.push_frame(frame)


class SpokenAssistantAggregator(GoogleAssistantContextAggregator):
    def __init__(
        self,
        *args,
        on_message: Optional[UserCallback] = None,
        on_partial: Optional[UserCallback] = None,
        cutoff_marker: str = "[.. cut off by user utterance]",
        action_state: Optional[ActionExtractionState] = None,
        **kwargs,
    ):
        super().__init__(*args, enable_direct_mode=True, **kwargs)
        self._on_message = on_message
        self._on_partial = on_partial
        self._cutoff_marker = cutoff_marker
        self._interrupted = False
        self._draft_message: Content | None = None
        self._action_state = action_state

    async def process_frame(self, frame, direction: FrameDirection):  # type: ignore[override]
        if isinstance(frame, InterruptionFrame):
            self._interrupted = True
            await super().process_frame(frame, direction)
            self._draft_message = None
            self._interrupted = False
            return
        await super().process_frame(frame, direction)

    async def _handle_llm_start(self, frame: LLMFullResponseStartFrame):  # type: ignore[override]
        await super()._handle_llm_start(frame)
        if getattr(self, "_started", 0) == 1:
            self._draft_message = None

    async def _handle_llm_end(self, frame: LLMFullResponseEndFrame):  # type: ignore[override]
        await super()._handle_llm_end(frame)
        if not getattr(self, "_started", 0):
            self._draft_message = None

    async def _handle_text(self, frame: TextFrame):  # type: ignore[override]
        await super()._handle_text(frame)
        if not getattr(self, "_started", 0):
            return

        if not frame.text or not frame.text.strip():
            return

        aggregation = getattr(self, "_aggregation", "")
        clean_text = aggregation.strip()
        if clean_text:
            self._upsert_draft_message(clean_text)

        if self._on_partial:
            result = self._on_partial(aggregation)
            if inspect.isawaitable(result):
                await result

    async def handle_aggregation(self, aggregation: str):  # type: ignore[override]
        clean_text = aggregation.strip()
        logged_text = self._resolve_logged_text(clean_text)
        if not logged_text:
            return
        if self._interrupted and self._cutoff_marker:
            logged_text = f"{logged_text.rstrip()} {self._cutoff_marker}"

        self._upsert_draft_message(logged_text)

        if self._on_message:
            result = self._on_message(logged_text)
            if inspect.isawaitable(result):
                await result

    def _upsert_draft_message(self, text: str) -> None:
        message = self._draft_message
        if message is None:
            message = Content(role="model", parts=[Part(text=text)])
            self._context.add_message(message)
            self._draft_message = message
            return

        if getattr(message, "parts", None) and message.parts:
            first_part = message.parts[0]
            if getattr(first_part, "text", None) is not None:
                first_part.text = text
                return
        message.parts = [Part(text=text)]

    def _resolve_logged_text(self, clean_text: str) -> str:
        if not self._action_state:
            return clean_text

        raw_text, actions = self._action_state.snapshot()
        raw_text = raw_text.strip()
        if actions and raw_text:
            return raw_text
        return clean_text


class OpenAISpokenAssistantAggregator(OpenAIAssistantContextAggregator):
    def __init__(
        self,
        *args,
        on_message: Optional[UserCallback] = None,
        on_partial: Optional[UserCallback] = None,
        cutoff_marker: str = "[.. cut off by user utterance]",
        action_state: Optional[ActionExtractionState] = None,
        **kwargs,
    ):
        super().__init__(*args, enable_direct_mode=True, **kwargs)
        self._on_message = on_message
        self._on_partial = on_partial
        self._cutoff_marker = cutoff_marker
        self._interrupted = False
        self._draft_message: dict | None = None
        self._action_state = action_state

    async def process_frame(self, frame, direction: FrameDirection):  # type: ignore[override]
        if isinstance(frame, InterruptionFrame):
            self._interrupted = True
            await super().process_frame(frame, direction)
            self._draft_message = None
            self._interrupted = False
            return
        await super().process_frame(frame, direction)

    async def _handle_llm_start(self, frame: LLMFullResponseStartFrame):  # type: ignore[override]
        await super()._handle_llm_start(frame)
        if getattr(self, "_started", 0) == 1:
            self._draft_message = None

    async def _handle_llm_end(self, frame: LLMFullResponseEndFrame):  # type: ignore[override]
        await super()._handle_llm_end(frame)
        if not getattr(self, "_started", 0):
            self._draft_message = None

    async def _handle_text(self, frame: TextFrame):  # type: ignore[override]
        await super()._handle_text(frame)
        if not getattr(self, "_started", 0):
            return

        if not frame.text or not frame.text.strip():
            return

        aggregation = getattr(self, "_aggregation", "")
        clean_text = aggregation.strip()
        if clean_text:
            self._upsert_draft_message(clean_text)

        if self._on_partial:
            result = self._on_partial(aggregation)
            if inspect.isawaitable(result):
                await result

    async def handle_aggregation(self, aggregation: str):  # type: ignore[override]
        clean_text = aggregation.strip()
        logged_text = self._resolve_logged_text(clean_text)
        if not logged_text:
            return
        if self._interrupted and self._cutoff_marker:
            logged_text = f"{logged_text.rstrip()} {self._cutoff_marker}"

        self._upsert_draft_message(logged_text)

        if self._on_message:
            result = self._on_message(logged_text)
            if inspect.isawaitable(result):
                await result

    def _upsert_draft_message(self, text: str) -> None:
        message = self._draft_message
        if message is None:
            message = {"role": "assistant", "content": text}
            self._context.add_message(message)
            self._draft_message = message
            return
        message["content"] = text

    def _resolve_logged_text(self, clean_text: str) -> str:
        if not self._action_state:
            return clean_text

        raw_text, actions = self._action_state.snapshot()
        raw_text = raw_text.strip()
        if actions and raw_text:
            return raw_text
        return clean_text


class AssistantSpeechGate(FrameProcessor):
    def __init__(self, release_delay: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self._release_delay = release_delay
        self._muted = False
        self._unmute_task: Optional[asyncio.Task[None]] = None

    def start_speaking(self) -> None:
        if self._unmute_task:
            self._unmute_task.cancel()
            self._unmute_task = None
        self._muted = True

    def stop_speaking(self) -> None:
        if self._unmute_task:
            self._unmute_task.cancel()
        loop = self.get_event_loop()
        self._unmute_task = loop.create_task(self._delayed_unmute())

    async def _delayed_unmute(self) -> None:
        try:
            await asyncio.sleep(self._release_delay)
        except asyncio.CancelledError:
            return
        self._muted = False
        self._unmute_task = None

    async def cleanup(self) -> None:
        if self._unmute_task:
            self._unmute_task.cancel()
            self._unmute_task = None
        await super().cleanup()

    async def process_frame(self, frame, direction: FrameDirection):  # type: ignore[override]
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and self._muted and isinstance(frame, AudioRawFrame):
            return
        await self.push_frame(frame, direction)


class LLMResponseMonitor(FrameProcessor):
    def __init__(
        self,
        *,
        on_start: Optional[Callable[[], Awaitable[None] | None]] = None,
        on_end: Optional[Callable[[], Awaitable[None] | None]] = None,
        on_text: Optional[Callable[[], Awaitable[None] | None]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._on_start = on_start
        self._on_end = on_end
        self._on_text = on_text

    async def process_frame(self, frame, direction: FrameDirection):  # type: ignore[override]
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMFullResponseStartFrame) and self._on_start:
                result = self._on_start()
                if inspect.isawaitable(result):
                    await result
            elif isinstance(frame, LLMFullResponseEndFrame) and self._on_end:
                result = self._on_end()
                if inspect.isawaitable(result):
                    await result
            elif isinstance(frame, LLMTextFrame) and self._on_text:
                result = self._on_text()
                if inspect.isawaitable(result):
                    await result
            elif isinstance(frame, TextFrame) and not isinstance(frame, LLMTextFrame) and self._on_text:
                result = self._on_text()
                if inspect.isawaitable(result):
                    await result
        await self.push_frame(frame, direction)


@dataclass
class PipelineComponents:
    pipeline: Pipeline
    task: PipelineTask
    runner: PipelineRunner
    inbox_watcher: InboxWatcher
    params_watcher: ParamsWatcher


@dataclass
class PendingInboxItem:
    text: str
    attempts: int = 0


class VoicePipelineController:
    def __init__(
        self,
        config_manager: ConfigManager,
        session_paths: SessionPaths,
        history: ConversationHistory,
        event_logger: EventLogger,
        metrics: MetricsTracker,
        *,
        actions_path: Path,
        inbox_path: Path,
        params_path: Path,
    ):
        self._config_manager = config_manager
        self._session_paths = session_paths
        self._history = history
        self._event_logger = event_logger
        self._metrics = metrics
        self._actions_path = actions_path
        self._inbox_path = inbox_path
        self._params_path = params_path
        self._loop = asyncio.get_event_loop()
        self._inbox_buffer: list[str] = []
        self._inbox_buffer_lock = threading.Lock()
        self._inbox_push_lock = asyncio.Lock()
        self._pending_inbox: deque[PendingInboxItem] = deque()
        self._pending_inbox_current: Optional[PendingInboxItem] = None
        self._pending_inbox_started_at: Optional[float] = None
        self._pending_inbox_retry = False
        self._pending_inbox_max_attempts = 1
        self._pending_inbox_timeout_s: Optional[float] = None
        self._pending_inbox_timeout_task: Optional[asyncio.Task[None]] = None
        self._llm_busy = False
        self._llm_busy_since: Optional[float] = None
        self._llm_text_seen_since_start = False
        self._run_after_llm_idle = False
        self._inbox_debounce_s = 0.15
        self._inbox_run_task: Optional[asyncio.Task[None]] = None
        self._components: Optional[PipelineComponents] = None
        self._transport: Optional[LocalAudioTransport] = None
        self._stt_service = None
        self._llm_service: Optional[Any] = None
        self._tts_service = None
        self._llm_request_timeout_s: Optional[float] = None
        self._speech_gate: Optional[AssistantSpeechGate] = None
        self._user_aggregator: Optional[Any] = None
        self._assistant_aggregator: Optional[Any] = None
        self._stt_muted = False
        self._pause_stt_on_idle = False
        self._history_on_idle = "keep"
        self._last_tts_audio_event = 0.0
        self._tts_audio_event_interval_s = 1.0

    async def _on_user_message(self, text: str) -> None:
        if self._components:
            self._components.params_watcher.drain_pending()
        self._history.add("user", text)
        if "turn_start" not in self._metrics.marks:
            self._metrics.mark("turn_start")

    async def _on_assistant_message(self, text: str) -> None:
        self._history.add("assistant", text, replace_last=True)

    async def _on_assistant_partial(self, text: str) -> None:
        self._history.add_partial("assistant", text)

    def _build_pipeline(self, config: RuntimeConfig) -> Pipeline:
        keys = get_api_keys()
        stt_service, stt_provider = build_stt_service(config, deepgram_api_key=keys["deepgram"])
        tts_provider = (config.tts.provider or DEFAULT_TTS_PROVIDER).strip().lower()
        needs_deepgram = stt_provider == "deepgram_flux" or tts_provider in ("deepgram", "deepgram_aura", "deepgram-aura")
        if needs_deepgram and not keys["deepgram"]:
            raise RuntimeError("DEEPGRAM_API_KEY must be set when using Deepgram STT/TTS.")

        ensure_devices_selected(
            self._config_manager,
            require_input=stt_provider == "deepgram_flux",
            require_output=True,
        )

        transport_params = LocalAudioTransportParams(
            audio_in_enabled=stt_provider == "deepgram_flux",
            audio_in_sample_rate=16000,
            audio_out_enabled=True,
            audio_out_sample_rate=config.audio.output_sample_rate or config.tts.sample_rate,
            input_device_index=config.audio.input_device_index,
            output_device_index=config.audio.output_device_index,
        )
        self._transport = LocalAudioTransport(transport_params)

        self._stt_service = stt_service
        self._llm_service, llm_provider = build_llm_service(
            config,
            google_api_key=keys["google"],
            openai_api_key=keys["openai"],
        )
        self._tts_service = build_tts_service(config, keys["deepgram"] or "")
        self._speech_gate = None
        self._llm_request_timeout_s = config.llm.request_timeout_s
        if self._llm_request_timeout_s is None:
            self._pending_inbox_timeout_s = 45.0
        else:
            self._pending_inbox_timeout_s = max(5.0, self._llm_request_timeout_s + 5.0)
        self._register_llm_handlers()

        context_pair = create_llm_context(self._llm_service, self._history.export())
        base_user = context_pair.user()
        base_assistant = context_pair.assistant()

        if llm_provider in {"ollama", "openai"}:
            self._user_aggregator = OpenAIUserAggregator(
                base_user.context,
                params=getattr(base_user, "_params", LLMUserAggregatorParams()),
                on_message=self._on_user_message,
                transform=self._consume_inbox_buffer,
            )
        else:
            self._user_aggregator = UserAggregator(
                base_user.context,
                params=getattr(base_user, "_params", LLMUserAggregatorParams()),
                on_message=self._on_user_message,
                transform=self._consume_inbox_buffer,
            )
        provider = (config.tts.provider or DEFAULT_TTS_PROVIDER).strip().lower()
        assistant_params = LLMAssistantAggregatorParams(
            expect_stripped_words=provider not in ("macos_say", "macos-say", "say")
        )
        action_state = ActionExtractionState()
        if llm_provider in {"ollama", "openai"}:
            self._assistant_aggregator = OpenAISpokenAssistantAggregator(
                base_assistant.context,
                params=assistant_params,
                on_message=self._on_assistant_message,
                on_partial=self._on_assistant_partial,
                cutoff_marker=config.tts.cutoff_marker,
                action_state=action_state,
            )
        else:
            self._assistant_aggregator = SpokenAssistantAggregator(
                base_assistant.context,
                params=assistant_params,
                on_message=self._on_assistant_message,
                on_partial=self._on_assistant_partial,
                cutoff_marker=config.tts.cutoff_marker,
                action_state=action_state,
            )

        processors = []
        if stt_provider == "deepgram_flux":
            processors.append(self._transport.input())
        processors.extend([
            self._stt_service,
            STTStandaloneIFilter(event_logger=self._event_logger),
            self._user_aggregator,
            self._llm_service,
            LLMResponseMonitor(
                on_start=self._notify_llm_response_start,
                on_end=self._notify_llm_response_end,
                on_text=self._note_llm_text,
            ),
            ReasoningTraceFilter(),
            ActionExtractorFilter(self._actions_path, self._event_logger, action_state=action_state),
            self._tts_service,
            self._assistant_aggregator,
            self._transport.output(),
        ])
        return Pipeline(processors)

    def _idle_timeout_frames(self) -> tuple[type, ...]:
        return (
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            TranscriptionFrame,
            InterimTranscriptionFrame,
            TTSStartedFrame,
            TTSAudioRawFrame,
            TTSStoppedFrame,
        )

    def _install_event_hooks(self, task: PipelineTask) -> None:
        metrics = self._metrics
        event_logger = self._event_logger

        @task.event_handler("on_frame_reached_downstream")
        async def _on_down(task_obj, frame):  # type: ignore[unused-ignore]
            if isinstance(frame, UserStartedSpeakingFrame):
                metrics.mark("turn_start")
            elif isinstance(frame, UserStoppedSpeakingFrame):
                metrics.mark("audio_in_last_packet")
            elif isinstance(frame, LLMTextFrame) and "llm_first_token" not in metrics.marks:
                metrics.mark("llm_first_token")
            elif (
                isinstance(frame, TextFrame)
                and not isinstance(frame, LLMTextFrame)
                and "llm_first_token" not in metrics.marks
            ):
                metrics.mark("llm_first_token")
            elif isinstance(frame, TTSStartedFrame):
                if self._speech_gate:
                    self._speech_gate.start_speaking()
            elif isinstance(frame, TTSAudioRawFrame):
                if "tts_first_audio" not in metrics.marks:
                    metrics.mark("tts_first_audio")
                audio = getattr(frame, "audio", None)
                audio_bytes = len(audio) if audio else 0
                now = time.time()
                if should_log_tts_audio_event(
                    self._last_tts_audio_event,
                    now,
                    audio_bytes,
                    self._tts_audio_event_interval_s,
                ):
                    self._last_tts_audio_event = now
                    event_logger.emit(
                        "tts_audio_buffer",
                        {"audio_bytes": audio_bytes, "timestamp": now},
                    )
            elif isinstance(frame, TTSStoppedFrame):
                if self._speech_gate:
                    self._speech_gate.stop_speaking()
                metrics.mark("turn_complete")
                metrics.compute_turn_metrics()
                metrics.reset()
                if self._components:
                    self._components.params_watcher.drain_pending()

        @task.event_handler("on_pipeline_started")
        async def _on_started(task_obj, frame):  # type: ignore[unused-ignore]
            event_logger.emit("pipeline_started", {"timestamp": time.time()})

        @task.event_handler("on_pipeline_finished")
        async def _on_finished(task_obj, frame):  # type: ignore[unused-ignore]
            event_logger.emit("pipeline_finished", {"timestamp": time.time()})

        @task.event_handler("on_idle_timeout")
        async def _on_idle_timeout(task_obj):  # type: ignore[unused-ignore]
            await self._handle_idle_timeout()

    def _register_llm_handlers(self) -> None:
        if not self._llm_service or not self._event_logger:
            return

        @self._llm_service.event_handler("on_completion_timeout")
        async def _on_completion_timeout(service):  # type: ignore[unused-ignore]
            payload = {"timestamp": time.time()}
            if self._llm_busy_since is not None:
                payload["busy_seconds"] = time.time() - self._llm_busy_since
            if self._llm_request_timeout_s is not None:
                payload["request_timeout_s"] = self._llm_request_timeout_s
            payload["llm_text_seen"] = self._llm_text_seen_since_start
            self._event_logger.emit("llm_completion_timeout", payload)
            self._complete_pending_inbox(
                "llm_completion_timeout",
                retry_on_empty=True,
            )
            self._reset_llm_busy("llm_completion_timeout")
            if self._pending_inbox or self._pending_inbox_current:
                self._schedule_coroutine(self._schedule_inbox_run, 0.0)

    def _consume_inbox_buffer(self, text: str) -> str:
        with self._inbox_buffer_lock:
            if not self._inbox_buffer:
                return text
            extras = "\n".join(self._inbox_buffer)
            self._inbox_buffer.clear()
        if text:
            return f"{text}\n{extras}"
        return extras

    def _schedule_coroutine(self, coro_func: Callable[..., Awaitable[None]], *args: object) -> None:
        coro = coro_func(*args)
        try:
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        except RuntimeError:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                coro.close()
                return
            loop.create_task(coro)
        else:
            future.add_done_callback(lambda fut: fut.exception())

    def _note_llm_text(self) -> None:
        self._llm_text_seen_since_start = True

    def _request_stt_resume(self) -> None:
        if not self._pause_stt_on_idle or not self._stt_muted:
            return
        self._schedule_coroutine(self._set_stt_muted, False)

    def _inbox_callback(self, mode: str, payload: str) -> None:
        self._request_stt_resume()
        if mode == "append":
            with self._inbox_buffer_lock:
                self._inbox_buffer.append(payload)
            return
        if mode == "push":
            text = self._consume_inbox_buffer(payload)
            future = asyncio.run_coroutine_threadsafe(self._handle_inbox_push(text), self._loop)
            future.add_done_callback(lambda fut: fut.exception())

    async def _handle_inbox_push(self, text: str) -> None:
        async with self._inbox_push_lock:
            await self._interrupt_assistant_if_needed()
            self._reset_llm_busy("inbox_push")
            if not text:
                return
            self._pending_inbox.append(PendingInboxItem(text=text))
            if self._event_logger:
                self._event_logger.emit(
                    "inbox_queued",
                    {"pending": len(self._pending_inbox), "timestamp": time.time()},
                )
            await self._schedule_inbox_run()

    async def _inject_user_text(self, text: str) -> None:
        if not self._components:
            return
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        frame = TranscriptionFrame(text=text, user_id="inbox", timestamp=timestamp)
        await self._components.task.queue_frame(frame)

    async def _inject_user_turn(self, text: str) -> None:
        if not self._components:
            return
        await self._components.task.queue_frame(UserStartedSpeakingFrame(emulated=True))
        await self._inject_user_text(text)
        await self._components.task.queue_frame(UserStoppedSpeakingFrame(emulated=True))

    async def _append_inbox_message(self, text: str) -> None:
        if not self._components:
            return
        await self._on_user_message(text)
        context = None
        if self._user_aggregator:
            context = getattr(self._user_aggregator, "context", None)
        if context is None and self._assistant_aggregator:
            context = getattr(self._assistant_aggregator, "context", None)
        if context is None:
            return

        message = {"role": "user", "content": text}
        add_messages = getattr(context, "add_messages", None)
        if callable(add_messages):
            add_messages([message])
            return
        add_message = getattr(context, "add_message", None)
        if callable(add_message):
            add_message(message)

    async def _schedule_inbox_run(self, delay: Optional[float] = None) -> None:
        if delay is None:
            delay = self._inbox_debounce_s
        if self._inbox_run_task and not self._inbox_run_task.done():
            return
        self._run_after_llm_idle = False
        if self._event_logger:
            self._event_logger.emit(
                "inbox_run_scheduled",
                {"delay_s": delay, "pending": len(self._pending_inbox), "timestamp": time.time()},
            )

        async def _delayed() -> None:
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return
            await self._trigger_inbox_run()

        self._inbox_run_task = asyncio.create_task(_delayed())

    async def _trigger_inbox_run(self) -> None:
        self._inbox_run_task = None
        if self._llm_busy or self._pending_inbox_current is not None:
            self._run_after_llm_idle = True
            if self._event_logger:
                self._event_logger.emit(
                    "inbox_run_deferred",
                    {
                        "llm_busy": self._llm_busy,
                        "pending": len(self._pending_inbox),
                        "timestamp": time.time(),
                    },
                )
            return
        if not self._pending_inbox:
            return
        item = self._pending_inbox.popleft()
        self._pending_inbox_current = item
        self._pending_inbox_started_at = time.time()
        self._start_inbox_timeout()
        if self._event_logger:
            self._event_logger.emit(
                "inbox_run_start",
                {
                    "pending": len(self._pending_inbox),
                    "timestamp": self._pending_inbox_started_at,
                },
            )
        await self._append_inbox_message(item.text)
        await self._queue_llm_run()

    async def _queue_llm_run(self) -> None:
        if not self._components:
            return
        self._refresh_dynamic_system_prompt()
        if self._event_logger:
            self._event_logger.emit("llm_run_queued", {"timestamp": time.time()})
        await self._components.task.queue_frame(LLMRunFrame())

    def _start_inbox_timeout(self) -> None:
        if self._pending_inbox_timeout_s is None:
            return
        item = self._pending_inbox_current
        if item is None:
            return
        if self._pending_inbox_timeout_task and not self._pending_inbox_timeout_task.done():
            self._pending_inbox_timeout_task.cancel()

        async def _watch(item: PendingInboxItem) -> None:
            try:
                await asyncio.sleep(self._pending_inbox_timeout_s)
            except asyncio.CancelledError:
                return
            if self._pending_inbox_current is not item:
                return
            if self._event_logger:
                self._event_logger.emit(
                    "inbox_run_timeout",
                    {
                        "pending": len(self._pending_inbox),
                        "attempt": item.attempts,
                        "timeout_s": self._pending_inbox_timeout_s,
                        "timestamp": time.time(),
                    },
                )
            self._complete_pending_inbox("inbox_run_timeout", retry_on_empty=True)
            self._reset_llm_busy("inbox_run_timeout")
            if self._pending_inbox:
                self._schedule_coroutine(self._schedule_inbox_run, 0.0)

        self._pending_inbox_timeout_task = asyncio.create_task(_watch(item))

    def _cancel_inbox_timeout(self) -> None:
        if self._pending_inbox_timeout_task and not self._pending_inbox_timeout_task.done():
            self._pending_inbox_timeout_task.cancel()
        self._pending_inbox_timeout_task = None

    async def _interrupt_assistant_if_needed(self) -> None:
        if not self._assistant_aggregator:
            return
        started = getattr(self._assistant_aggregator, "_FrameProcessor__started", False)
        if not started:
            return
        try:
            await self._assistant_aggregator.push_interruption_task_frame_and_wait()
            if self._event_logger:
                self._event_logger.emit("inbox_interrupt", {"source": "inbox"})
            self._reset_llm_busy("inbox_interrupt")
        except Exception as exc:  # pragma: no cover - defensive
            if self._event_logger:
                self._event_logger.emit("inbox_interrupt_error", {"error": repr(exc)})

    def _notify_llm_response_start(self) -> None:
        self._llm_busy = True
        self._llm_busy_since = time.time()
        self._llm_text_seen_since_start = False

    def _notify_llm_response_end(self) -> None:
        self._llm_busy = False
        self._llm_busy_since = None
        self._complete_pending_inbox("llm_response_end", retry_on_empty=True)
        if self._run_after_llm_idle or self._pending_inbox:
            self._run_after_llm_idle = False
            self._schedule_coroutine(self._schedule_inbox_run, 0.0)

    def _complete_pending_inbox(self, reason: str, *, retry_on_empty: bool = False) -> None:
        if self._pending_inbox_current is None:
            return
        item = self._pending_inbox_current
        duration_s = None
        if self._pending_inbox_started_at is not None:
            duration_s = time.time() - self._pending_inbox_started_at
        if retry_on_empty and not self._llm_text_seen_since_start:
            self._pending_inbox_retry = True
            if self._event_logger:
                self._event_logger.emit(
                    "llm_empty_response",
                    {"timestamp": time.time()},
                )
        retry_item = self._pending_inbox_retry
        self._pending_inbox_current = None
        self._pending_inbox_retry = False
        self._pending_inbox_started_at = None
        self._cancel_inbox_timeout()
        if self._event_logger:
            payload = {
                "pending": len(self._pending_inbox),
                "timestamp": time.time(),
                "attempt": item.attempts,
                "reason": reason,
            }
            if duration_s is not None:
                payload["duration_s"] = duration_s
            self._event_logger.emit("inbox_run_complete", payload)
        if retry_item:
            if item.attempts < self._pending_inbox_max_attempts:
                item.attempts += 1
                self._pending_inbox.appendleft(item)
                if self._event_logger:
                    self._event_logger.emit(
                        "inbox_run_requeued",
                        {
                            "attempt": item.attempts,
                            "pending": len(self._pending_inbox),
                            "timestamp": time.time(),
                        },
                    )
            elif self._event_logger:
                self._event_logger.emit(
                    "inbox_run_dropped",
                    {
                        "attempt": item.attempts,
                        "pending": len(self._pending_inbox),
                        "timestamp": time.time(),
                    },
                )

    def _reset_llm_busy(self, reason: str) -> None:
        if not self._llm_busy:
            return
        busy_seconds = None
        if self._llm_busy_since is not None:
            busy_seconds = time.time() - self._llm_busy_since
        self._llm_busy = False
        self._llm_busy_since = None
        self._run_after_llm_idle = False
        if self._event_logger:
            payload = {"reason": reason, "timestamp": time.time()}
            if busy_seconds is not None:
                payload["busy_seconds"] = busy_seconds
            self._event_logger.emit("llm_busy_reset", payload)

    async def _handle_idle_timeout(self) -> None:
        if self._history_on_idle == "reset":
            self._history.reset(system_prompt=self._history.system_message)
            self._reset_llm_contexts()
        if not self._pause_stt_on_idle:
            return
        await self._set_stt_muted(True)

    def _reset_llm_contexts(self) -> None:
        contexts: list[object] = []
        for aggregator in (self._user_aggregator, self._assistant_aggregator):
            context = getattr(aggregator, "context", None)
            if context is not None:
                contexts.append(context)

        if not contexts:
            return

        unique_contexts: list[object] = []
        seen_ids: set[int] = set()
        for context in contexts:
            context_id = id(context)
            if context_id in seen_ids:
                continue
            seen_ids.add(context_id)
            unique_contexts.append(context)

        for context in unique_contexts:
            set_messages = getattr(context, "set_messages", None)
            if callable(set_messages):
                set_messages([])
                continue
            messages = getattr(context, "messages", None)
            if isinstance(messages, list):
                messages.clear()

        system_prompt = self._history.system_message
        if isinstance(system_prompt, str) or system_prompt is None:
            for context in unique_contexts:
                self._update_context_system_prompt(context, system_prompt)

    async def _set_stt_muted(self, mute: bool) -> None:
        if self._stt_muted == mute:
            return
        self._stt_muted = mute
        if not self._components:
            return
        await self._components.task.queue_frame(STTMuteFrame(mute=mute))

    @staticmethod
    def _update_context_system_prompt(context: object, new_prompt: Optional[str]) -> None:
        if context is None:
            return
        if hasattr(context, "system_message"):
            setattr(context, "system_message", new_prompt)
            return
        get_messages = getattr(context, "get_messages", None)
        set_messages = getattr(context, "set_messages", None)
        if not callable(get_messages) or not callable(set_messages):
            return
        messages = list(get_messages())
        filtered: list[object] = []
        for message in messages:
            role = None
            if isinstance(message, dict):
                role = message.get("role")
            else:
                role = getattr(message, "role", None)
            if role == "system":
                continue
            filtered.append(message)
        if isinstance(new_prompt, str):
            filtered.insert(0, {"role": "system", "content": new_prompt})
        set_messages(filtered)

    def _apply_system_prompt(self, new_prompt: Optional[str], *, update_history: bool = True) -> None:
        if self._llm_service:
            setattr(self._llm_service, "_system_instruction", new_prompt)
        if self._assistant_aggregator:
            self._update_context_system_prompt(getattr(self._assistant_aggregator, "context", None), new_prompt)
        if self._user_aggregator:
            self._update_context_system_prompt(getattr(self._user_aggregator, "context", None), new_prompt)
        if update_history and (isinstance(new_prompt, str) or new_prompt is None):
            self._history.set_system_message(new_prompt)

    def _refresh_dynamic_system_prompt(self) -> None:
        config = getattr(self._config_manager, "config", None)
        if not config:
            return
        template = getattr(getattr(config, "llm", None), "system_prompt", None)
        rendered = prompt_templates.render_clean_time(template)
        if not isinstance(template, str) or rendered is None or rendered == template:
            return
        self._apply_system_prompt(rendered, update_history=False)

    def _apply_param_updates(self, updates: dict) -> None:
        if not updates:
            return
        if "llm" in updates and self._llm_service:
            llm_updates = updates["llm"]
            model = llm_updates.get("model")
            if model:
                normalized_model = normalize_model_for_service(model, self._llm_service)
                if normalized_model:
                    self._llm_service.set_model_name(normalized_model)
                elif self._event_logger:
                    self._event_logger.emit("llm_model_ignored", {"model": model})
            if "temperature" in llm_updates or "max_tokens" in llm_updates:
                params = getattr(self._llm_service, "_settings", {})
                if "temperature" in llm_updates:
                    params["temperature"] = llm_updates["temperature"]
                if "max_tokens" in llm_updates:
                    max_tokens = llm_updates["max_tokens"]
                    provider, _ = parse_llm_model_spec(self._config_manager.config.llm.model)
                    if provider == "openai":
                        params["max_completion_tokens"] = max_tokens
                        if "max_tokens" in params:
                            params["max_tokens"] = NOT_GIVEN
                    else:
                        params["max_tokens"] = max_tokens
            if "system_prompt" in llm_updates:
                new_prompt = llm_updates.get("system_prompt")
                self._apply_system_prompt(new_prompt)
        if "stt" in updates and self._stt_service:
            stt_updates = updates["stt"]
            params = getattr(self._stt_service, "_params", None)
            if params:
                if "eager_eot_threshold" in stt_updates:
                    params.eager_eot_threshold = stt_updates["eager_eot_threshold"]
                if "eot_threshold" in stt_updates:
                    params.eot_threshold = stt_updates["eot_threshold"]
                if "eot_timeout_ms" in stt_updates:
                    params.eot_timeout_ms = stt_updates["eot_timeout_ms"]
        if "tts" in updates and self._tts_service:
            tts_updates = updates["tts"]
            voice = tts_updates.get("voice")
            if voice:
                self._tts_service.set_voice(voice)
            if "encoding" in tts_updates:
                self._tts_service._settings["encoding"] = tts_updates["encoding"]  # type: ignore[attr-defined]
            if "sample_rate" in tts_updates:
                self._tts_service.sample_rate = tts_updates["sample_rate"]
        self._event_logger.emit("params_update", updates)

    async def start(self) -> None:
        config = self._config_manager.config
        pipeline_config = config.pipeline
        self._pause_stt_on_idle = bool(pipeline_config.pause_stt_on_idle)
        self._history_on_idle = _normalize_history_on_idle(pipeline_config.history_on_idle)
        pipeline = self._build_pipeline(config)

        params = PipelineParams(
            allow_interruptions=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=config.tts.sample_rate,
            enable_metrics=True,
        )
        task = PipelineTask(
            pipeline,
            params=params,
            conversation_id=self._session_paths.session_name,
            cancel_on_idle_timeout=pipeline_config.cancel_on_idle_timeout,
            idle_timeout_secs=pipeline_config.idle_timeout_secs,
            idle_timeout_frames=self._idle_timeout_frames(),
        )
        runner = PipelineRunner()
        self._install_event_hooks(task)

        inbox_watcher = InboxWatcher(self._inbox_path, self._inbox_callback, event_logger=self._event_logger)
        params_watcher = ParamsWatcher(
            self._params_path,
            self._config_manager,
            self._history,
            apply_callback=self._apply_param_updates,
            event_logger=self._event_logger,
        )
        self._components = PipelineComponents(
            pipeline=pipeline,
            task=task,
            runner=runner,
            inbox_watcher=inbox_watcher,
            params_watcher=params_watcher,
        )

        self._components.inbox_watcher.start()
        self._components.params_watcher.start()
        await asyncio.sleep(0)
        self._components.params_watcher.drain_pending()
        if self._pause_stt_on_idle:
            await task.queue_frame(STTMuteFrame(mute=True))
            self._stt_muted = True

        await runner.run(task)

    async def stop(self) -> None:
        if not self._components:
            return
        await self._components.task.stop_when_done()
        self._components.inbox_watcher.stop()
        self._components.params_watcher.stop()


async def run_voice_pipeline(session_name: Optional[str] = None) -> None:
    config_manager = ConfigManager()
    ensure_audio_device_preferences(config_manager)
    session_paths = new_session(config_manager.config, session_name=session_name)
    event_logger = EventLogger(session_paths.event_log)
    metrics = MetricsTracker(event_logger)
    max_history_messages = _coerce_max_history_messages(config_manager.config.pipeline.max_history_messages)
    history = ConversationHistory(
        session_paths.transcript,
        clean_transcript_path=session_paths.llm_transcript,
        max_messages=max_history_messages,
    )
    history.set_system_message(config_manager.config.llm.system_prompt)

    controller = VoicePipelineController(
        config_manager,
        session_paths,
        history,
        event_logger,
        metrics,
        actions_path=Path("runtime/actions.txt"),
        inbox_path=Path("runtime/inbox.txt"),
        params_path=Path("runtime/params_inbox.ndjson"),
    )

    await controller.start()

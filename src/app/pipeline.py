from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

from pipecat.frames.frames import (
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.google.llm import (
    GoogleAssistantContextAggregator,
    GoogleContextAggregatorPair,
    GoogleLLMService,
    GoogleUserContextAggregator,
)
from pipecat.services.google.llm import LLMAssistantAggregatorParams, LLMUserAggregatorParams
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from app.config import ConfigManager, RuntimeConfig, get_api_keys
from app.devices import ensure_devices_selected
from app.filters import ActionExtractorFilter
from app.history import ConversationHistory
from app.inbox_watch import InboxWatcher
from app.logging_io import EventLogger
from app.metrics import MetricsTracker
from app.params_apply import ParamsWatcher
from app.session import SessionPaths, new_session
from services.llm import build_google_llm, create_google_context
from services.stt import build_deepgram_flux_stt
from services.tts import build_deepgram_tts

UserCallback = Callable[[str], Awaitable[None]]


class UserAggregator(GoogleUserContextAggregator):
    def __init__(self, *args, on_message: Optional[UserCallback] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_message = on_message

    async def handle_aggregation(self, aggregation: str):  # type: ignore[override]
        await super().handle_aggregation(aggregation)
        if self._on_message and aggregation:
            await self._on_message(aggregation)


class AssistantAggregator(GoogleAssistantContextAggregator):
    def __init__(self, *args, on_message: Optional[UserCallback] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_message = on_message

    async def push_aggregation(self):  # type: ignore[override]
        text = getattr(self, "_aggregation", "")
        await super().push_aggregation()
        if self._on_message and text:
            await self._on_message(text)


@dataclass
class PipelineComponents:
    pipeline: Pipeline
    task: PipelineTask
    runner: PipelineRunner
    inbox_watcher: InboxWatcher
    params_watcher: ParamsWatcher


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
        self._components: Optional[PipelineComponents] = None
        self._transport: Optional[LocalAudioTransport] = None
        self._stt_service = None
        self._llm_service: Optional[GoogleLLMService] = None
        self._tts_service = None
        self._user_aggregator: Optional[UserAggregator] = None
        self._assistant_aggregator: Optional[AssistantAggregator] = None

    async def _on_user_message(self, text: str) -> None:
        self._history.add("user", text)
        if "turn_start" not in self._metrics.marks:
            self._metrics.mark("turn_start")

    async def _on_assistant_message(self, text: str) -> None:
        self._history.add("assistant", text)

    def _build_pipeline(self, config: RuntimeConfig) -> Pipeline:
        keys = get_api_keys()
        if not keys["deepgram"] or not keys["google"]:
            raise RuntimeError("GOOGLE_API_KEY and DEEPGRAM_API_KEY must be set.")

        transport_params = LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_enabled=True,
            audio_out_sample_rate=config.tts.sample_rate,
            input_device_index=config.audio.input_device_index,
            output_device_index=config.audio.output_device_index,
        )
        self._transport = LocalAudioTransport(transport_params)

        self._stt_service = build_deepgram_flux_stt(config, keys["deepgram"])
        self._llm_service = build_google_llm(config, keys["google"])
        self._tts_service = build_deepgram_tts(config, keys["deepgram"])

        context_pair: GoogleContextAggregatorPair = create_google_context(
            self._llm_service, self._history.export()
        )
        base_user = context_pair.user()
        base_assistant = context_pair.assistant()

        self._user_aggregator = UserAggregator(
            base_user.context,
            params=getattr(base_user, "_params", LLMUserAggregatorParams()),
            on_message=self._on_user_message,
        )
        self._assistant_aggregator = AssistantAggregator(
            base_assistant.context,
            params=getattr(base_assistant, "_params", LLMAssistantAggregatorParams()),
            on_message=self._on_assistant_message,
        )

        processors = [
            self._transport.input(),
            self._stt_service,
            self._user_aggregator,
            self._llm_service,
            self._assistant_aggregator,
            ActionExtractorFilter(self._actions_path, self._event_logger),
            self._tts_service,
            self._transport.output(),
        ]
        return Pipeline(processors)

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
            elif isinstance(frame, TTSAudioRawFrame) and "tts_first_audio" not in metrics.marks:
                metrics.mark("tts_first_audio")
            elif isinstance(frame, TTSStoppedFrame):
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

    def _inbox_callback(self, mode: str, payload: str) -> None:
        if mode == "append" and not self._inbox_buffer:
            mode = "push"
        if mode == "append":
            self._inbox_buffer.append(payload)
            return
        if mode == "push":
            if self._inbox_buffer:
                payload = "\n".join(self._inbox_buffer + [payload])
                self._inbox_buffer = []
            future = asyncio.run_coroutine_threadsafe(self._inject_user_text(payload), self._loop)
            future.add_done_callback(lambda fut: fut.exception())

    async def _inject_user_text(self, text: str) -> None:
        if not self._components:
            return
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        frame = TranscriptionFrame(text=text, user_id="inbox", timestamp=timestamp)
        await self._components.task.queue_frame(frame)

    def _apply_param_updates(self, updates: dict) -> None:
        if not updates:
            return
        if "llm" in updates and self._llm_service:
            llm_updates = updates["llm"]
            model = llm_updates.get("model")
            if model:
                self._llm_service.set_model_name(model)
            if "temperature" in llm_updates or "max_tokens" in llm_updates:
                params = getattr(self._llm_service, "_settings", {})
                if "temperature" in llm_updates:
                    params["temperature"] = llm_updates["temperature"]
                if "max_tokens" in llm_updates:
                    params["max_tokens"] = llm_updates["max_tokens"]
            if "system_prompt" in llm_updates:
                setattr(self._llm_service, "_system_instruction", llm_updates["system_prompt"])
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
        ensure_devices_selected(self._config_manager)
        pipeline = self._build_pipeline(config)

        params = PipelineParams(
            allow_interruptions=False,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=config.tts.sample_rate,
            enable_metrics=True,
        )
        task = PipelineTask(pipeline, params=params, conversation_id=self._session_paths.session_name)
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
        inbox_watcher.start()
        params_watcher.start()

        self._components = PipelineComponents(
            pipeline=pipeline,
            task=task,
            runner=runner,
            inbox_watcher=inbox_watcher,
            params_watcher=params_watcher,
        )

        await runner.run(task)

    async def stop(self) -> None:
        if not self._components:
            return
        await self._components.task.stop_when_done()
        self._components.inbox_watcher.stop()
        self._components.params_watcher.stop()


async def run_voice_pipeline(session_name: Optional[str] = None) -> None:
    config_manager = ConfigManager()
    ensure_devices_selected(config_manager)
    session_paths = new_session(config_manager.config, session_name=session_name)
    event_logger = EventLogger(session_paths.event_log)
    metrics = MetricsTracker(event_logger)
    history = ConversationHistory(session_paths.transcript)

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

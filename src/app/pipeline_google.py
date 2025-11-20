from __future__ import annotations

import asyncio
import inspect
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

from pyaec import Aec
import numpy as np
from scipy.signal import resample_poly
from collections import deque

from pipecat.frames.frames import (
    AudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    InputAudioRawFrame,
    STTMuteFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    BotStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.google.llm import (
    GoogleAssistantContextAggregator,
    GoogleContextAggregatorPair,
    GoogleLLMService,
    GoogleUserContextAggregator,
)
from pipecat.services.google.llm import LLMAssistantAggregatorParams, LLMUserAggregatorParams
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
import app.resampler_patch # fix soxr resampler issue in pipecat

from app.config import ConfigManager, RuntimeConfig, get_api_keys
from app.devices import ensure_devices_selected
from app.filters import ActionExtractorFilter, STTStandaloneIFilter
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
LLM_TEXT_IS_TEXTFRAME = issubclass(LLMTextFrame, TextFrame)

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

class PyAECProcessor(FrameProcessor):
    def __init__(self, frame_size: int = 160, filter_length_secs: float = 0.4, sample_rate: int = 16000, mute_while_tts: bool = False, **kwargs):
        super().__init__(**kwargs)
        filter_length = int(sample_rate * filter_length_secs) # 0.4s
        self._aec = Aec(frame_size, filter_length, sample_rate, True)
        self._sr = sample_rate
        self._tts_sr = 48000
        self._playback_buffer = deque(maxlen=self._tts_sr // 3)
        self._post_tts_timeout = 100  # number of frames to keep AEC after TTS ends
        self._post_tts_counter = 0
        self._mute_while_tts = mute_while_tts
        self._muted = False
        
    def add_tts_audio(self, tts_audio: np.ndarray):
        """Feed TTS playback audio into the AEC buffer (downsampled to 16kHz mono)."""
        if tts_audio is None or len(tts_audio) == 0:
            return
        # Limit buffer growth
        if len(self._playback_buffer) + len(tts_audio) > self._playback_buffer.maxlen:
            excess = len(self._playback_buffer) + len(tts_audio) - self._playback_buffer.maxlen
            for _ in range(excess):
                self._playback_buffer.popleft()
        self._playback_buffer.extend(tts_audio)

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            if self._post_tts_counter == self._post_tts_timeout:
                self._playback_buffer.clear()
                self._post_tts_counter += 1
                # print("AEC: cleared playback buffer after TTS end")
            elif self._post_tts_counter < self._post_tts_timeout:
                self._post_tts_counter += 1

            mic_audio = np.frombuffer(frame.audio, dtype=np.int16)
            cleaned = mic_audio.copy()
            tts_audio = np.array(self._playback_buffer, dtype=np.int16)

            if len(tts_audio) > 0:
                if self._mute_while_tts:
                    if not self._muted:
                        # Mute mic audio while TTS is playing
                        cleaned = np.zeros_like(mic_audio)
                        # emit mute frame upstream
                        await self.push_frame(STTMuteFrame(True), FrameDirection.DOWNSTREAM)
                        self._muted = True
                else:
                    # print("========== AEC:", len(mic_audio), "mic samples;", len(tts_audio), "TTS samples")
                    # print dots to indicate AEC activity
                    # print(".", end="", flush=True)
                    if len(tts_audio) < len(mic_audio):
                        tts_audio = np.pad(tts_audio, (0, len(mic_audio) - len(tts_audio)))
                        cleaned = self._aec.cancel_echo(mic_audio, tts_audio)
                    if len(tts_audio) >= len(mic_audio):
                        # split buffer into chunks of mic_audio length and clean the mic against each
                        num_chunks = len(tts_audio) // len(mic_audio)
                        max_rms = 1000
                        for i in range(num_chunks):
                            tts_chunk = tts_audio[i * len(mic_audio) : (i + 1) * len(mic_audio)]
                            cleaned = np.array(self._aec.cancel_echo(cleaned, tts_chunk), dtype=np.int16)
                            rms = np.sqrt(np.mean(cleaned.astype(np.float32)**2))
                            if rms > max_rms:
                                max_rms = rms
                            if i > 3 and rms < max_rms * 0.3:
                                # print(f"AEC: early exit after {i+1} chunks")
                                # print(i, end="", flush=True)
                                # Early exit if signal is sufficiently cleaned
                                break

                        # print("Done after", i + 1, "chunks; max RMS:", int(max_rms))
            else: # no TTS audio
                if self._mute_while_tts and self._muted:
                    # emit unmute frame upstream
                    await self.push_frame(STTMuteFrame(False), FrameDirection.DOWNSTREAM)
                    self._muted = False
            cleaned = np.clip(cleaned, -32768, 32767).astype(np.int16)
            frame.audio = cleaned.tobytes()

        await self.push_frame(frame, direction)


class PushUpTTSFrameProcessor(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process_frame(self, frame, direction: FrameDirection): 
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            # print("TTS STARTED")
            self._aec_ref._post_tts_counter = self._aec_ref._post_tts_timeout + 1
            if self._aec_ref._mute_while_tts:
                await self.push_frame(STTMuteFrame(True), FrameDirection.UPSTREAM)
        #     await self.push_frame(frame, FrameDirection.UPSTREAM)
        if isinstance(frame, TTSStoppedFrame) or isinstance(frame, UserStartedSpeakingFrame):
            # print("TTS STOPPED")
            self._aec_ref._post_tts_counter = 0
            # await self.push_frame(frame, FrameDirection.UPSTREAM)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TTSAudioRawFrame):
            tts_frame = np.frombuffer(frame.audio, dtype=np.int16)
            tts_sr = frame.sample_rate

            # Convert stereo → mono
            if frame.num_channels == 2:
                tts_frame = tts_frame.reshape(-1, 2).mean(axis=1)

            # Resample to 16kHz for AEC
            down = tts_sr // 16000
            resampled = resample_poly(tts_frame, up=1, down=down)
            # resampled = np.clip(resampled, -32768, 32767).astype(np.int16)

            # Feed into AEC processor (instead of global)
            if hasattr(self, "_aec_ref") and self._aec_ref:
                self._aec_ref.add_tts_audio(resampled)
        await self.push_frame(frame, direction)

class VoiceSwitcher(FrameProcessor):
    def __init__(self,
                 switch_voice: Optional[UserCallback] = None,
                 trigger_next_roleplay_turn: Optional[UserCallback] = None,
                  **kwargs):
        super().__init__(**kwargs)
        self._switch_voice = switch_voice
        self._trigger_next_roleplay_turn = trigger_next_roleplay_turn

    async def process_frame(self, frame, direction: FrameDirection): 
        await super().process_frame(frame, direction)
        # print("VoiceSwitcher processing frame:", type(frame).__name__)
        if isinstance(frame, BotStoppedSpeakingFrame):
            print("==== BOT STOPPED SPEAKING, switch voice =====")
            # Switch voice based on current speaker
            if self._switch_voice:
                await self._switch_voice()
                await asyncio.sleep(0)  # yield to allow voice switch to take effect
            # Trigger next roleplay turn
            if self._trigger_next_roleplay_turn:
                await self._trigger_next_roleplay_turn("")

        await self.push_frame(frame, direction)

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
        self._inbox_buffer_lock = threading.Lock()
        self._components: Optional[PipelineComponents] = None
        self._transport: Optional[LocalAudioTransport] = None
        self._stt_service = None
        self._llm_service: Optional[GoogleLLMService] = None
        self._tts_service = None
        self._aec_proc: Optional[PyAECProcessor] = None
        self._push_up_tts_proc: Optional[PushUpTTSFrameProcessor] = None
        self._voice_switcher: Optional[VoiceSwitcher] = None
        self._speech_gate: Optional[AssistantSpeechGate] = None
        self._user_aggregator: Optional[UserAggregator] = None
        self._assistant_aggregator: Optional[AssistantAggregator] = None

        self._current_speaker = "persona1"
        self._dialogue_file = Path("runtime/dialogue.txt")
        self._full_memory = ""

    async def _on_user_message(self, text: str) -> None:
        if self._components:
            self._components.params_watcher.drain_pending()
        # config = self._config_manager.config
        # if config.llm.mode == "2personas":
        #     await self._trigger_next_roleplay_turn(text)
        # else:
        self._history.add("user", text)
        if "turn_start" not in self._metrics.marks:
            self._metrics.mark("turn_start")

    async def _on_assistant_message(self, text: str) -> None:
        self._history.add("assistant", text, replace_last=True)
        
        config = self._config_manager.config
        if config.llm.mode == "2personas":
            
            #Get response from Persona 2 and append to dialogue.txt
            with self._dialogue_file.open("a", encoding="utf8") as f:
                f.write(f"{self._current_speaker.upper()}: {text}\n")

            # Trigger next round of conversation
            # await self._trigger_next_roleplay_turn(text)

    async def _on_assistant_partial(self, text: str) -> None:
        self._history.add_partial("assistant", text)

    def _build_pipeline(self, config: RuntimeConfig) -> Pipeline:
        keys = get_api_keys()
        if not keys["deepgram"] or not keys["google"]:
            raise RuntimeError("GOOGLE_API_KEY and DEEPGRAM_API_KEY must be set.")

        transport_params = LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_enabled=True,
            audio_out_sample_rate=config.audio.output_sample_rate or config.tts.sample_rate,
            input_device_index=config.audio.input_device_index,
            output_device_index=config.audio.output_device_index,
        )
        self._transport = LocalAudioTransport(transport_params)
        
        self._stt_service = build_deepgram_flux_stt(config, keys["deepgram"])
        self._llm_service = build_google_llm(config, keys["google"])
        self._tts_service = build_deepgram_tts(config, keys["deepgram"])
        self._aec_proc = PyAECProcessor(mute_while_tts=(config.audio.aec == "mute_while_tts"))
        self._push_up_tts_proc = PushUpTTSFrameProcessor()
        self._voice_switcher = VoiceSwitcher(
            switch_voice=self._switch_voice,
            trigger_next_roleplay_turn=self._trigger_next_roleplay_turn,
        )
        self._speech_gate = AssistantSpeechGate()

        context_pair: GoogleContextAggregatorPair = create_google_context(
            self._llm_service, self._history.export()
        )
        base_user = context_pair.user()
        base_assistant = context_pair.assistant()

        self._user_aggregator = UserAggregator(
            base_user.context,
            params=getattr(base_user, "_params", LLMUserAggregatorParams()),
            on_message=self._on_user_message,
            transform=self._consume_inbox_buffer,
        )
        self._assistant_aggregator = AssistantAggregator(
            base_assistant.context,
            params=getattr(base_assistant, "_params", LLMAssistantAggregatorParams()),
            on_message=self._on_assistant_message,
            on_partial=self._on_assistant_partial,
        )

        if config.audio.aec == "off":
            # replace AEC with no-op
            class _NoAEC(FrameProcessor):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self._post_tts_counter = 0
                    self._post_tts_timeout = -1
                async def process_frame(self, frame, direction: FrameDirection):
                    await super().process_frame(frame, direction)
                    await self.push_frame(frame, direction)
                def add_tts_audio(self, tts_audio: np.ndarray):
                    # no-op when AEC is disabled
                    return
            self._aec_proc = _NoAEC()
        self._push_up_tts_proc._aec_ref = self._aec_proc

        processors = [
            self._transport.input(),
            self._aec_proc,
            self._speech_gate,
            self._stt_service,
            STTStandaloneIFilter(event_logger=self._event_logger),
            self._user_aggregator,
            self._llm_service,
            self._assistant_aggregator,
            ActionExtractorFilter(self._actions_path, self._event_logger),
            self._tts_service,
            self._transport.output(),
            self._push_up_tts_proc,
            self._voice_switcher,
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
            elif (
                isinstance(frame, TextFrame)
                and not isinstance(frame, LLMTextFrame)
                and "llm_first_token" not in metrics.marks
            ):
                metrics.mark("llm_first_token")
            elif isinstance(frame, TTSStartedFrame):
                if self._speech_gate:
                    self._speech_gate.start_speaking()
            elif isinstance(frame, TTSAudioRawFrame) and "tts_first_audio" not in metrics.marks:
                metrics.mark("tts_first_audio")
            elif isinstance(frame, TTSStoppedFrame):
                if self._speech_gate:
                    self._speech_gate.stop_speaking()
                metrics.mark("turn_complete")
                print("=== TURN COMPLETE ===")
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

    def _consume_inbox_buffer(self, text: str) -> str:
        with self._inbox_buffer_lock:
            if not self._inbox_buffer:
                return text
            extras = "\n".join(self._inbox_buffer)
            self._inbox_buffer.clear()
        if text:
            return f"{text}\n{extras}"
        return extras

    def _inbox_callback(self, mode: str, payload: str) -> None:
        if mode == "append":
            with self._inbox_buffer_lock:
                self._inbox_buffer.append(payload)
            return
        if mode == "push":
            text = self._consume_inbox_buffer(payload)
            future = asyncio.run_coroutine_threadsafe(self._handle_inbox_push(text), self._loop)
            future.add_done_callback(lambda fut: fut.exception())

    async def _handle_inbox_push(self, text: str) -> None:
        await self._interrupt_assistant_if_needed()
        if text:
            await self._inject_user_turn(text)

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
        except Exception as exc:  # pragma: no cover - defensive
            if self._event_logger:
                self._event_logger.emit("inbox_interrupt_error", {"error": repr(exc)})

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
            new_prompt = llm_updates.get("system_prompt")
            if new_prompt is not None:
                setattr(self._llm_service, "_system_instruction", new_prompt)
                if self._assistant_aggregator:
                    context = getattr(self._assistant_aggregator, "context", None)
                    if context is not None and hasattr(context, "system_message"):
                        context.system_message = new_prompt
                if self._user_aggregator:
                    context = getattr(self._user_aggregator, "context", None)
                    if context is not None and hasattr(context, "system_message"):
                        context.system_message = new_prompt
                self._history.set_system_message(new_prompt)
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
            allow_interruptions=(config.audio.aec != "mute_while_tts"),
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

        if config.llm.mode == "2personas":
            persona_voice = config.llm.persona1["voice"]
            self._tts_service.set_voice(persona_voice)

            # Inject the conversation starter
            await self._inject_user_turn(config.llm.persona1['opening'])

            # Save the line into the dialogue file
            with self._dialogue_file.open("a", encoding="utf8") as f:
                f.write(f"{self._current_speaker.upper()}: {config.llm.persona1['opening']}\n")

            # Switch speakers
            self._current_speaker = "persona2"

        await runner.run(task)

    async def stop(self) -> None:
        if not self._components:
            return
        await self._components.task.stop_when_done()
        self._components.inbox_watcher.stop()
        self._components.params_watcher.stop()

    async def _trigger_next_roleplay_turn(self, last_reply: str):
        config = self._config_manager.config

        # Swap speaker
        if self._current_speaker == "persona2":
            self._current_speaker = "persona1"
            system_prompt = config.llm.persona1["prompt"]
            persona_voice = config.llm.persona1["voice"]
        else:
            self._current_speaker = "persona2"
            system_prompt = config.llm.persona2["prompt"]
            persona_voice = config.llm.persona2["voice"]

        # 🔊 Switch TTS voice based on active persona
        # self._tts_service.set_voice(persona_voice)

        try:
            if self._dialogue_file.exists():
                with self._dialogue_file.open("r", encoding="utf8") as f:
                    self._full_memory = f.read()
            else:
                self._full_memory = ""
        except Exception:
            self._full_memory = ""

        # Construct persona-specific input
        persona_input = f"System:\n{system_prompt}\n\nThis is the history of the conversation, take account of it when formulating your answer:\n{self._full_memory}\n\nReply only to the last line of your conversation partner. Stay in character! Your turn:"

        # Inject the next speaking turn
        await self._inject_user_turn(persona_input)

    async def _switch_voice(self):
        config = self._config_manager.config
        persona = self._current_speaker
        print("Switching voice to:", persona)
        if persona == "persona1":
            persona_voice = config.llm.persona1["voice"]
        else:
            persona_voice = config.llm.persona2["voice"]

        # 🔊 Switch TTS voice based on active persona
        self._tts_service.set_voice(persona_voice)


async def run_voice_pipeline(session_name: Optional[str] = None) -> None:
    config_manager = ConfigManager()
    ensure_devices_selected(config_manager)
    session_paths = new_session(config_manager.config, session_name=session_name)
    event_logger = EventLogger(session_paths.event_log)
    metrics = MetricsTracker(event_logger)
    history = ConversationHistory(
        session_paths.transcript,
        clean_transcript_path=session_paths.llm_transcript,
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
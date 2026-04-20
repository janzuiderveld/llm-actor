import asyncio
import os
from typing import Optional
from collections import deque
import numpy as np
from scipy.signal import resample_poly

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.services.groq.llm import GroqLLMService
from pipecat.frames.frames import (
    InputAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    STTMuteFrame,
)
from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pyaec import Aec

import app.resampler_patch  # fix soxr resampler issue in pipecat

from services.tts_kokoro import KokoroTTSService
from services.stt_moonshine import MoonshineSTTService
from services.llm import build_groq_llm
from app.config import ConfigManager, get_api_keys


class PyAECFilter(BaseAudioFilter):
    def __init__(self, frame_size: int = 160, filter_length_secs: float = 0.4, sample_rate: int = 16000, **kwargs):
        super().__init__()
        filter_length = int(sample_rate * filter_length_secs)
        self._aec = Aec(frame_size, filter_length, sample_rate, True)
        self._sr = sample_rate
        self._tts_sr = 48000
        self._playback_buffer = deque(maxlen=self._tts_sr // 3)
        self._post_tts_timeout = 100
        self._post_tts_counter = 0

    async def start(self, sample_rate: int):
        pass

    async def stop(self):
        pass

    async def process_frame(self, frame):
        pass

    def add_tts_audio(self, tts_audio: np.ndarray):
        if tts_audio is None or len(tts_audio) == 0:
            return
        if len(self._playback_buffer) + len(tts_audio) > self._playback_buffer.maxlen:
            excess = len(self._playback_buffer) + len(tts_audio) - self._playback_buffer.maxlen
            for _ in range(excess):
                self._playback_buffer.popleft()
        self._playback_buffer.extend(tts_audio)

    async def filter(self, audio: bytes) -> bytes:
        if self._post_tts_counter == self._post_tts_timeout:
            self._playback_buffer.clear()
            self._post_tts_counter += 1
        elif self._post_tts_counter < self._post_tts_timeout:
            self._post_tts_counter += 1

        mic_audio = np.frombuffer(audio, dtype=np.int16)
        cleaned = mic_audio.copy()
        tts_audio = np.array(self._playback_buffer, dtype=np.int16)

        if len(tts_audio) > 0:
            if len(tts_audio) < len(mic_audio):
                tts_audio = np.pad(tts_audio, (0, len(mic_audio) - len(tts_audio)))
                cleaned = self._aec.cancel_echo(mic_audio, tts_audio)
            if len(tts_audio) >= len(mic_audio):
                num_chunks = len(tts_audio) // len(mic_audio)
                max_rms = 1000
                for i in range(num_chunks):
                    tts_chunk = tts_audio[i * len(mic_audio) : (i + 1) * len(mic_audio)]
                    cleaned = np.array(self._aec.cancel_echo(cleaned, tts_chunk), dtype=np.int16)
                    rms = np.sqrt(np.mean(cleaned.astype(np.float32)**2))
                    if rms > max_rms:
                        max_rms = rms
                    if i > 3 and rms < max_rms * 0.3:
                        break

        cleaned = np.clip(cleaned, -32768, 32767).astype(np.int16)
        return cleaned.tobytes()


class PushUpTTSFrameProcessor(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process_frame(self, frame, direction: FrameDirection): 
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            self._aec_ref._post_tts_counter = self._aec_ref._post_tts_timeout + 1
        if isinstance(frame, TTSStoppedFrame) or isinstance(frame, UserStartedSpeakingFrame):
            self._aec_ref._post_tts_counter = 0
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TTSAudioRawFrame):
            tts_frame = np.frombuffer(frame.audio, dtype=np.int16)
            tts_sr = frame.sample_rate

            if frame.num_channels == 2:
                tts_frame = tts_frame.reshape(-1, 2).mean(axis=1)

            down = tts_sr // 16000
            resampled = resample_poly(tts_frame, up=1, down=down)

            if hasattr(self, "_aec_ref") and self._aec_ref:
                self._aec_ref.add_tts_audio(resampled)
        await self.push_frame(frame, direction)


async def run_voice_pipeline(session_name: Optional[str] = None) -> None:
    config_manager = ConfigManager()
    config = config_manager.config
    keys = get_api_keys()

    groq_api_key = keys.get("groq") or os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is required to run the Groq pipeline.")

    aec_filter = PyAECFilter()

    transport_params = LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=16000,
        audio_out_enabled=True,
        audio_out_sample_rate=config.tts.sample_rate,
        vad_analyzer=SileroVADAnalyzer(),
        audio_in_filter=aec_filter,
    )
    transport = LocalAudioTransport(transport_params)

    # Basic Components
    stt = MoonshineSTTService(
        model_name="moonshine/tiny",
        language="en",
        vad_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    )
    
    llm = build_groq_llm(config, groq_api_key)
    
    tts = KokoroTTSService(
        model_path="assets/kokoro-v1.0.onnx",
        voices_path="assets/voices-v1.0.bin",
        voice_id=config.tts.voice,
        sample_rate=config.tts.sample_rate,
    )

    push_up_tts_proc = PushUpTTSFrameProcessor()
    push_up_tts_proc._aec_ref = aec_filter

    context = OpenAILLMContext([{"role": "system", "content": config.llm.system_prompt}])
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            push_up_tts_proc,
            context_aggregator.assistant(),
        ]
    )

    params = PipelineParams(
        allow_interruptions=True,
        audio_in_sample_rate=16000,
        audio_out_sample_rate=config.tts.sample_rate,
    )
    task = PipelineTask(pipeline, params=params, conversation_id=session_name)
    runner = PipelineRunner()

    await runner.run(task)

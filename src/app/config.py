from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path("runtime/config.json")
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default
    return value


@dataclass
class AudioConfig:
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None


@dataclass
class STTConfig:
    model: str = field(default_factory=lambda: _env("PIPECAT_DEFAULT_STT_MODEL", "deepgram-flux"))
    language: str = "en-US"
    eager_eot_threshold: float = 0.5
    eot_threshold: float = 0.85
    eot_timeout_ms: int = 1500


@dataclass
class LLMConfig:
    model: str = field(default_factory=lambda: _env("PIPECAT_DEFAULT_LLM_MODEL", "gemini-2.5-flash"))
    temperature: float = 0.6
    max_tokens: int = 1024
    system_prompt: str = (
        "You are a real-time voice assistant. Speak concisely.\n"
        "To request external actions, include them inside <...> within your reply. "
        "Keep them short and machine-readable. Do not speak the text inside <...>."
    )


@dataclass
class TTSConfig:
    voice: str = field(default_factory=lambda: _env("PIPECAT_DEFAULT_VOICE", "aura-2-thalia-en"))
    encoding: str = "linear16"
    sample_rate: int = 24000


@dataclass
class RuntimeConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfigManager:
    """Handles persisted runtime configuration as JSON."""

    def __init__(self, path: Path = CONFIG_PATH):
        self._path = path
        self._config = self._load()

    def _load(self) -> RuntimeConfig:
        if self._path.exists():
            data = json.loads(self._path.read_text())
            return RuntimeConfig(
                audio=AudioConfig(**data.get("audio", {})),
                stt=STTConfig(**data.get("stt", {})),
                llm=LLMConfig(**data.get("llm", {})),
                tts=TTSConfig(**data.get("tts", {})),
            )
        return RuntimeConfig()

    @property
    def config(self) -> RuntimeConfig:
        return self._config

    def save(self) -> None:
        self._path.write_text(json.dumps(self._config.as_dict(), indent=2))

    def set_audio_devices(self, input_index: int, output_index: int) -> None:
        self._config.audio.input_device_index = input_index
        self._config.audio.output_device_index = output_index
        self.save()

    def apply_updates(self, *, stt: Optional[Dict[str, Any]] = None, llm: Optional[Dict[str, Any]] = None,
                      tts: Optional[Dict[str, Any]] = None, audio: Optional[Dict[str, Any]] = None) -> None:
        if stt:
            for key, value in stt.items():
                if hasattr(self._config.stt, key):
                    setattr(self._config.stt, key, value)
        if llm:
            for key, value in llm.items():
                if hasattr(self._config.llm, key):
                    setattr(self._config.llm, key, value)
        if tts:
            for key, value in tts.items():
                if hasattr(self._config.tts, key):
                    setattr(self._config.tts, key, value)
        if audio:
            for key, value in audio.items():
                if hasattr(self._config.audio, key):
                    setattr(self._config.audio, key, value)
        self.save()


def get_api_keys() -> Dict[str, Optional[str]]:
    return {
        "google": _env("GOOGLE_API_KEY"),
        "deepgram": _env("DEEPGRAM_API_KEY"),
    }

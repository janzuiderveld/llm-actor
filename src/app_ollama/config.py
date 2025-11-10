from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path("runtime/config.json")
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default
    return value


def _normalize_device_index(value: Any) -> Optional[int]:
    try:
        if isinstance(value, int) and value >= 0:
            return value
        if value is not None:
            candidate = int(value)
            if candidate >= 0:
                return candidate
    except (TypeError, ValueError):
        return None
    return None


def detect_default_audio_device_indices() -> Tuple[Optional[int], Optional[int]]:
    """Return preferred input/output device indices, preferring Krisp when available."""
    try:
        import sounddevice as sd  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None, None

    try:
        devices = sd.query_devices()
    except Exception:  # pragma: no cover - runtime environment dependent
        return None, None

    def has_channels(info: Dict[str, Any], key: str) -> bool:
        try:
            return int(info.get(key, 0)) > 0
        except Exception:
            return False

    def find_device(predicate) -> Optional[int]:
        for idx, info in enumerate(devices):
            if not isinstance(info, dict):
                continue
            if predicate(idx, info):
                return idx
        return None

    krisp_input = find_device(
        lambda _idx, info: "krisp" in str(info.get("name", "")).lower() and has_channels(info, "max_input_channels")
    )
    krisp_output = find_device(
        lambda _idx, info: "krisp" in str(info.get("name", "")).lower() and has_channels(info, "max_output_channels")
    )

    default_input = None
    default_output = None

    hostapi_index = getattr(getattr(sd, "default", object()), "hostapi", None)
    if isinstance(hostapi_index, int) and hostapi_index >= 0:
        try:
            hostapi = sd.query_hostapis(hostapi_index)
        except Exception:  # pragma: no cover - runtime environment dependent
            hostapi = None
        if isinstance(hostapi, dict):
            default_input = _normalize_device_index(hostapi.get("default_input_device"))
            default_output = _normalize_device_index(hostapi.get("default_output_device"))

    try:
        sd_defaults = sd.default.device  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - runtime environment dependent
        sd_defaults = None
    if isinstance(sd_defaults, (list, tuple)):
        if default_input is None and len(sd_defaults) > 0:
            default_input = _normalize_device_index(sd_defaults[0])
        if default_output is None and len(sd_defaults) > 1:
            default_output = _normalize_device_index(sd_defaults[1])

    fallback_input = find_device(lambda _idx, info: has_channels(info, "max_input_channels"))
    fallback_output = find_device(lambda _idx, info: has_channels(info, "max_output_channels"))

    input_index = krisp_input if krisp_input is not None else default_input if default_input is not None else fallback_input
    output_index = (
        krisp_output if krisp_output is not None else default_output if default_output is not None else fallback_output
    )

    return input_index, output_index


@dataclass
class AudioConfig:
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None
    output_sample_rate: Optional[int] = None
    auto_select_devices: bool = True


@dataclass
class STTConfig:
    model: str = field(default_factory=lambda: _env("PIPECAT_DEFAULT_STT_MODEL", "flux-general-en"))
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


def load_or_initialize_runtime_config(path: Path = CONFIG_PATH) -> RuntimeConfig:
    path_exists = path.exists()
    data: Dict[str, Any] = {}

    if path_exists:
        maybe_data = json.loads(path.read_text())
        if isinstance(maybe_data, dict):
            data = maybe_data

    if data:
        config = RuntimeConfig(
            audio=AudioConfig(**data.get("audio", {})),
            stt=STTConfig(**data.get("stt", {})),
            llm=LLMConfig(**data.get("llm", {})),
            tts=TTSConfig(**data.get("tts", {})),
        )
    else:
        config = RuntimeConfig()

    updated = False

    if config.audio.auto_select_devices:
        input_index, output_index = detect_default_audio_device_indices()

        if input_index is not None and config.audio.input_device_index != input_index:
            config.audio.input_device_index = input_index
            updated = True
        if output_index is not None and config.audio.output_device_index != output_index:
            config.audio.output_device_index = output_index
            updated = True

    if not path_exists or not data or updated:
        path.write_text(json.dumps(config.as_dict(), indent=2))

    return config


class ConfigManager:
    """Handles persisted runtime configuration as JSON."""

    def __init__(self, path: Path = CONFIG_PATH):
        self._path = path
        self._config = self._load()

    def _load(self) -> RuntimeConfig:
        return load_or_initialize_runtime_config(self._path)

    @property
    def config(self) -> RuntimeConfig:
        return self._config

    @property
    def path(self) -> Path:
        return self._path

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
        "ollama_base_url": _env("OLLAMA_BASE_URL"),
    }

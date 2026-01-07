import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from app.config import DEFAULT_TTS_PROVIDER, TTSConfig


def test_default_tts_provider_constant_matches_platform() -> None:
    expected = "macos_say" if sys.platform == "darwin" else "deepgram"
    assert DEFAULT_TTS_PROVIDER == expected


def test_tts_config_defaults_to_default_provider_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIPECAT_DEFAULT_TTS_PROVIDER", raising=False)
    assert TTSConfig().provider == DEFAULT_TTS_PROVIDER


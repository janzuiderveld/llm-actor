import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from app.audio_events import should_log_tts_audio_event


def test_should_log_tts_audio_event_rejects_empty_buffers() -> None:
    assert not should_log_tts_audio_event(0.0, 10.0, 0, 1.0)


def test_should_log_tts_audio_event_respects_min_interval() -> None:
    assert not should_log_tts_audio_event(9.5, 10.0, 10, 1.0)
    assert should_log_tts_audio_event(8.5, 10.0, 10, 1.0)

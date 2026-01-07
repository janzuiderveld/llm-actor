import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from COFFEE_MACHINE.keepalive import (
    AUDIO_EVENT_NAME,
    BootSupervisor,
    find_latest_event_log,
    find_latest_transcript_log,
    is_quit_command,
    is_audio_event,
    latest_audio_timestamp_from_lines,
    latest_assistant_timestamp_from_lines,
)


def test_find_latest_event_log_picks_newest(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    session_old = conversations / "2024-01-01"
    session_new = conversations / "2024-01-02"
    session_old.mkdir(parents=True)
    session_new.mkdir(parents=True)
    old_log = session_old / "event_log.ndjson"
    new_log = session_new / "event_log.ndjson"
    old_log.write_text("old", encoding="utf-8")
    new_log.write_text("new", encoding="utf-8")
    os.utime(old_log, (1, 1))
    os.utime(new_log, (2, 2))

    assert find_latest_event_log(conversations) == new_log


def test_find_latest_transcript_log_picks_newest(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    session_old = conversations / "2024-01-01"
    session_new = conversations / "2024-01-02"
    session_old.mkdir(parents=True)
    session_new.mkdir(parents=True)
    old_log = session_old / "transcript.jsonl"
    new_log = session_new / "transcript.jsonl"
    old_log.write_text("old", encoding="utf-8")
    new_log.write_text("new", encoding="utf-8")
    os.utime(old_log, (1, 1))
    os.utime(new_log, (2, 2))

    assert find_latest_transcript_log(conversations) == new_log


def test_is_audio_event_requires_positive_audio_bytes() -> None:
    assert is_audio_event({"event": AUDIO_EVENT_NAME, "audio_bytes": 16})
    assert not is_audio_event({"event": AUDIO_EVENT_NAME, "audio_bytes": 0})
    assert not is_audio_event({"event": "other", "audio_bytes": 16})


def test_is_quit_command_detects_q() -> None:
    assert is_quit_command("q\n")
    assert is_quit_command(" Q ")
    assert not is_quit_command("quit")


def test_latest_audio_timestamp_from_lines_ignores_empty_buffers() -> None:
    lines = [
        '{"event": "tts_audio_buffer", "audio_bytes": 0, "timestamp": 1}',
        '{"event": "tts_audio_buffer", "audio_bytes": 12, "timestamp": 2}',
        '{"event": "other", "timestamp": 5}',
        '{"event": "tts_audio_buffer", "audio_bytes": 8, "timestamp": 3}',
    ]
    assert latest_audio_timestamp_from_lines(lines) == 3.0


def test_latest_assistant_timestamp_from_lines_ignores_empty_entries() -> None:
    lines = [
        '{"role": "assistant", "content": "", "ts": 1}',
        '{"role": "assistant", "content": "Hi", "ts": 2}',
        '{"role": "user", "content": "Yo", "ts": 5}',
        '{"role": "assistant", "content": "Done", "ts": 3}',
    ]
    assert latest_assistant_timestamp_from_lines(lines) == 3.0


def test_keepalive_terminate_process_stops_child(tmp_path) -> None:
    supervisor = BootSupervisor(
        silence_timeout_s=1.0,
        check_interval_s=0.1,
        conversations_dir=tmp_path,
        boot_path=Path("boot.py"),
    )
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        assert process.poll() is None
        supervisor._terminate_process(process)
        assert process.poll() is not None
    finally:
        supervisor._terminate_process(process)


def test_keepalive_terminate_process_sends_sigint_first(tmp_path) -> None:
    supervisor = BootSupervisor(
        silence_timeout_s=1.0,
        check_interval_s=0.1,
        conversations_dir=tmp_path,
        boot_path=Path("boot.py"),
    )
    flag_path = tmp_path / "sigint.txt"
    script = (
        "import signal, sys, time\n"
        "from pathlib import Path\n"
        "flag = Path(sys.argv[1])\n"
        "def on_int(_signum, _frame):\n"
        "    flag.write_text('sigint')\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGINT, on_int)\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(60)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(flag_path)],
        start_new_session=True,
    )
    try:
        time.sleep(0.1)
        supervisor._terminate_process(process)
        for _ in range(20):
            if flag_path.exists():
                break
            time.sleep(0.05)
        assert flag_path.read_text(encoding="utf-8") == "sigint"
    finally:
        supervisor._terminate_process(process)

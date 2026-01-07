import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from COFFEE_MACHINE.keepalive import AUDIO_EVENT_NAME
import COFFEE_MACHINE.keepalive_stability as stability
from COFFEE_MACHINE.keepalive_stability import (
    AudioEventMonitor,
    TranscriptMonitor,
    append_inbox_push,
    build_say_command,
    build_say_play_command,
    choose_say_audio_device,
    choose_cycle_interval_seconds,
    compute_followup_at,
    delay_before_first_cycle,
    ensure_speech_audio_deepgram,
    load_dotenv_if_available,
    parse_say_audio_devices,
    prepare_audio_command,
    resolve_audio_player,
    select_turns,
    wait_for_assistant_settle,
    wait_for_runtime_ready,
)


def _write_event(path: Path, *, event_name: str, audio_bytes: int, timestamp: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": event_name,
        "audio_bytes": audio_bytes,
        "timestamp": timestamp,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload))
        fh.write("\n")


def _write_transcript(path: Path, *, role: str, content: str, timestamp: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": timestamp, "role": role, "content": content}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload))
        fh.write("\n")


def test_build_say_command_includes_voice_and_rate(tmp_path) -> None:
    output_path = tmp_path / "speech.aiff"
    command = build_say_command(
        text="Hello there",
        output_path=output_path,
        voice="Alex",
        rate_wpm=180,
    )

    assert command == [
        "say",
        "-o",
        str(output_path),
        "-v",
        "Alex",
        "-r",
        "180",
        "Hello there",
    ]


def test_append_inbox_push_writes_prefixed_line(tmp_path) -> None:
    path = tmp_path / "inbox.txt"
    append_inbox_push(path, "Hello there")
    assert path.read_text(encoding="utf-8") == "P: Hello there\n"


def test_compute_followup_at_uses_latest_deadline() -> None:
    assert compute_followup_at(100.0, 105.0, 10.0, 2.0) == 110.0
    assert compute_followup_at(100.0, 120.0, 10.0, 2.0) == 122.0


def test_select_turns_uses_long_every_interval() -> None:
    assert select_turns(1, 2, 15, 10) == 2
    assert select_turns(10, 2, 15, 10) == 15
    assert select_turns(20, 2, 15, 10) == 15
    assert select_turns(3, 2, 15, 0) == 2


def test_choose_cycle_interval_seconds_uses_fixed_interval() -> None:
    calls = {}

    def fake_random(_low: float, _high: float) -> float:
        calls["called"] = True
        return 999.0

    assert (
        choose_cycle_interval_seconds(
            fixed_interval_s=42.0,
            min_interval_s=60.0,
            max_interval_s=1200.0,
            random_fn=fake_random,
        )
        == 42.0
    )
    assert calls == {}


def test_choose_cycle_interval_seconds_uses_random_range() -> None:
    calls = {}

    def fake_random(low: float, high: float) -> float:
        calls["args"] = (low, high)
        return 90.0

    assert (
        choose_cycle_interval_seconds(
            fixed_interval_s=None,
            min_interval_s=60.0,
            max_interval_s=1200.0,
            random_fn=fake_random,
        )
        == 90.0
    )
    assert calls["args"] == (60.0, 1200.0)


def test_choose_cycle_interval_seconds_swaps_bounds() -> None:
    calls = {}

    def fake_random(low: float, high: float) -> float:
        calls["args"] = (low, high)
        return 1.0

    assert (
        choose_cycle_interval_seconds(
            fixed_interval_s=None,
            min_interval_s=1200.0,
            max_interval_s=60.0,
            random_fn=fake_random,
        )
        == 1.0
    )
    assert calls["args"] == (60.0, 1200.0)


def test_parse_say_audio_devices_parses_ids_and_names() -> None:
    output = "  193 UMC404HD 192k\n  153 MacBook Pro Speakers\n  173 krisp speaker\n"
    assert parse_say_audio_devices(output) == [
        ("193", "UMC404HD 192k"),
        ("153", "MacBook Pro Speakers"),
        ("173", "krisp speaker"),
    ]


def test_choose_say_audio_device_prefers_non_krisp() -> None:
    devices = [
        ("173", "krisp speaker"),
        ("153", "MacBook Pro Speakers"),
    ]
    assert choose_say_audio_device(devices, preferred=None, allow_krisp=False) == (
        "153",
        "MacBook Pro Speakers",
    )


def test_choose_say_audio_device_prefers_macbook_speakers_over_other_devices() -> None:
    devices = [
        ("208", "UMC404HD 192k"),
        ("153", "MacBook Pro Speakers"),
        ("173", "krisp speaker"),
    ]
    assert choose_say_audio_device(devices, preferred=None, allow_krisp=False) == (
        "153",
        "MacBook Pro Speakers",
    )


def test_choose_say_audio_device_honors_preferred() -> None:
    devices = [
        ("173", "krisp speaker"),
        ("153", "MacBook Pro Speakers"),
    ]
    assert choose_say_audio_device(devices, preferred="153", allow_krisp=False) == (
        "153",
        "MacBook Pro Speakers",
    )


def test_build_say_play_command_includes_device_and_voice() -> None:
    command = build_say_play_command(
        text="Test line",
        audio_device="153",
        voice="Alex",
        rate_wpm=180,
    )
    assert command == ["say", "-a", "153", "-v", "Alex", "-r", "180", "Test line"]


def test_resolve_audio_player_prefers_afplay(tmp_path) -> None:
    audio_path = tmp_path / "clip.aiff"

    def which(name: str):
        return "/usr/bin/afplay" if name == "afplay" else None

    assert resolve_audio_player(audio_path, which=which) == ["afplay", str(audio_path)]


def test_resolve_audio_player_override(tmp_path) -> None:
    audio_path = tmp_path / "clip.aiff"
    command = resolve_audio_player(
        audio_path,
        override="ffplay -nodisp -autoexit {audio_path}",
    )
    assert command == [
        "ffplay",
        "-nodisp",
        "-autoexit",
        str(audio_path),
    ]


def test_load_dotenv_if_available_missing_module() -> None:
    def fake_import(_name: str):
        raise ModuleNotFoundError("dotenv")

    assert load_dotenv_if_available(import_module=fake_import) is False


def test_load_dotenv_if_available_loads_path(tmp_path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("DEEPGRAM_API_KEY=test", encoding="utf-8")
    called = {}

    class DummyModule:
        def load_dotenv(self, dotenv_path=None) -> None:
            called["path"] = dotenv_path

    def fake_import(_name: str):
        return DummyModule()

    assert load_dotenv_if_available(dotenv_path=dotenv_path, import_module=fake_import) is True
    assert called["path"] == str(dotenv_path)


def test_ensure_speech_audio_deepgram_regenerates(tmp_path) -> None:
    audio_path = tmp_path / "speech.wav"
    calls = {}

    def fake_fetch(**kwargs):
        calls["fetch"] = kwargs
        return b"\x01\x02" * 4

    def fake_write(path: Path, audio_bytes: bytes, *, sample_rate: int) -> None:
        calls["write"] = (path, audio_bytes, sample_rate)

    ensure_speech_audio_deepgram(
        audio_path=audio_path,
        text="Test input",
        api_key="test-key",
        model="aura-2-thalia-en",
        encoding="linear16",
        sample_rate=16000,
        container="none",
        regenerate=True,
        fetch_audio_fn=fake_fetch,
        write_wav_fn=fake_write,
    )

    assert calls["fetch"]["text"] == "Test input"
    assert calls["write"] == (audio_path, b"\x01\x02" * 4, 16000)


def test_ensure_speech_audio_deepgram_skips_when_existing(tmp_path) -> None:
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"existing")
    calls = {"fetch": 0}

    def fake_fetch(**_kwargs):
        calls["fetch"] += 1
        return b""

    ensure_speech_audio_deepgram(
        audio_path=audio_path,
        text="Test input",
        api_key="test-key",
        model="aura-2-thalia-en",
        encoding="linear16",
        sample_rate=16000,
        container="none",
        regenerate=False,
        fetch_audio_fn=fake_fetch,
    )

    assert calls["fetch"] == 0


def test_prepare_audio_command_uses_say_on_macos(tmp_path) -> None:
    args = argparse.Namespace(
        playback_mode="say",
        skip_speech=False,
        play_command=None,
        speech_text="Coffee machine test.",
        speech_audio=tmp_path / "speech.aiff",
        say_audio_device=None,
        allow_krisp=False,
        say_voice=None,
        say_rate_wpm=None,
        regenerate_audio=False,
        deepgram_tts_model="aura-2-thalia-en",
        deepgram_tts_sample_rate=16000,
    )
    calls = {"ensure": 0, "resolve": 0}

    def fake_list_devices():
        return [("153", "MacBook Pro Speakers")]

    def fake_choose_devices(devices, preferred, allow_krisp):
        return ("153", "MacBook Pro Speakers")

    def fake_build_say_play_command(**kwargs):
        return ["say", "-a", "153", "Coffee machine test."]

    def fake_ensure_speech_audio(**kwargs):
        calls["ensure"] += 1

    def fake_resolve_audio_player(*_args, **_kwargs):
        calls["resolve"] += 1
        return ["afplay", "speech.aiff"]

    logger = logging.getLogger("test_prepare_audio_command_uses_say_on_macos")
    logger.addHandler(logging.NullHandler())

    command = prepare_audio_command(
        args,
        logger,
        platform="darwin",
        list_say_audio_devices_fn=fake_list_devices,
        choose_say_audio_device_fn=fake_choose_devices,
        build_say_play_command_fn=fake_build_say_play_command,
        ensure_speech_audio_fn=fake_ensure_speech_audio,
        resolve_audio_player_fn=fake_resolve_audio_player,
    )

    assert command == ["say", "-a", "153", "Coffee machine test."]
    assert calls == {"ensure": 0, "resolve": 0}


def test_prepare_audio_command_uses_deepgram(monkeypatch) -> None:
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    args = argparse.Namespace(
        playback_mode="deepgram",
        skip_speech=False,
        play_command=None,
        speech_text="Coffee machine test.",
        speech_audio=stability.DEFAULT_AUDIO_PATH,
        say_audio_device=None,
        allow_krisp=False,
        say_voice=None,
        say_rate_wpm=None,
        regenerate_audio=False,
        deepgram_tts_model="aura-2-thalia-en",
        deepgram_tts_sample_rate=16000,
    )
    calls = {}

    def fake_ensure_speech_audio_deepgram(**kwargs):
        calls["ensure"] = kwargs["audio_path"]

    def fake_resolve_audio_player(audio_path: Path, **_kwargs):
        calls["resolve"] = audio_path
        return ["afplay", str(audio_path)]

    logger = logging.getLogger("test_prepare_audio_command_uses_deepgram")
    logger.addHandler(logging.NullHandler())

    command = prepare_audio_command(
        args,
        logger,
        platform="darwin",
        ensure_speech_audio_deepgram_fn=fake_ensure_speech_audio_deepgram,
        resolve_audio_player_fn=fake_resolve_audio_player,
    )

    assert command == ["afplay", str(stability.DEFAULT_DEEPGRAM_AUDIO_PATH)]
    assert calls["ensure"] == stability.DEFAULT_DEEPGRAM_AUDIO_PATH
    assert calls["resolve"] == stability.DEFAULT_DEEPGRAM_AUDIO_PATH


def test_audio_event_monitor_tracks_latest_audio(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    log_one = conversations / "2024-01-01" / "event_log.ndjson"
    log_two = conversations / "2024-01-02" / "event_log.ndjson"

    _write_event(log_one, event_name=AUDIO_EVENT_NAME, audio_bytes=12, timestamp=10.0)
    os.utime(log_one, (1, 1))

    monitor = AudioEventMonitor(conversations)
    monitor.sync()
    assert monitor.last_audio_timestamp() == 10.0

    _write_event(log_one, event_name="other", audio_bytes=0, timestamp=11.0)
    os.utime(log_one, (1, 1))
    assert monitor.poll() is False
    assert monitor.last_audio_timestamp() == 10.0

    _write_event(log_one, event_name=AUDIO_EVENT_NAME, audio_bytes=8, timestamp=15.0)
    os.utime(log_one, (1, 1))
    assert monitor.poll() is True
    assert monitor.last_audio_timestamp() == 15.0

    _write_event(log_two, event_name=AUDIO_EVENT_NAME, audio_bytes=8, timestamp=20.0)
    os.utime(log_two, (2, 2))

    assert monitor.poll() is True
    assert monitor.last_audio_timestamp() == 20.0
    monitor.close()


def test_wait_for_runtime_ready_detects_boot_files(tmp_path, monkeypatch) -> None:
    runtime_root = tmp_path / "runtime"
    buttons = runtime_root / "coffee_machine_buttons.txt"
    lock_file = runtime_root / "coffee_machine_action_watcher.lock"
    monkeypatch.setattr(stability, "BUTTON_PRESSES_FILE", buttons)
    monkeypatch.setattr(stability, "WATCHER_LOCK_FILE", lock_file)

    started_at = time.time()
    buttons.parent.mkdir(parents=True, exist_ok=True)
    buttons.write_text("", encoding="utf-8")
    os.utime(buttons, (started_at + 1, started_at + 1))
    lock_file.write_text(str(os.getpid()), encoding="utf-8")

    logger = logging.getLogger("test_wait_for_runtime_ready")
    logger.addHandler(logging.NullHandler())
    assert wait_for_runtime_ready(started_at, timeout_s=0.2, logger=logger) is True


def test_transcript_monitor_tracks_assistant_lines(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    log_one = conversations / "2024-01-01" / "transcript.jsonl"
    log_two = conversations / "2024-01-02" / "transcript.jsonl"

    _write_transcript(log_one, role="system", content="Prompt", timestamp=1.0)
    _write_transcript(log_one, role="assistant", content="Hello", timestamp=2.0)
    os.utime(log_one, (1, 1))

    monitor = TranscriptMonitor(conversations)
    monitor.sync()
    assert monitor.last_assistant_timestamp() == 2.0

    _write_transcript(log_one, role="user", content="Hi", timestamp=3.0)
    os.utime(log_one, (1, 1))
    assert monitor.poll() is False
    assert monitor.last_assistant_timestamp() == 2.0

    _write_transcript(log_one, role="assistant", content="Brewing.", timestamp=4.0)
    os.utime(log_one, (1, 1))
    assert monitor.poll() is True
    assert monitor.last_assistant_timestamp() == 4.0

    _write_transcript(log_two, role="assistant", content="Done.", timestamp=6.0)
    os.utime(log_two, (2, 2))
    assert monitor.poll() is True
    assert monitor.last_assistant_timestamp() == 6.0
    monitor.close()


def test_delay_before_first_cycle_waits(monkeypatch) -> None:
    now = {"t": 0.0}
    sleeps = []

    def fake_time() -> float:
        return now["t"]

    def fake_sleep(duration: float) -> None:
        sleeps.append(duration)
        now["t"] += duration

    monkeypatch.setattr(stability.time, "time", fake_time)
    monkeypatch.setattr(stability.time, "sleep", fake_sleep)

    logger = logging.getLogger("test_delay_before_first_cycle_waits")
    logger.addHandler(logging.NullHandler())
    assert delay_before_first_cycle(1.2, lambda: False, logger) is True
    assert sum(sleeps) >= 1.2


def test_delay_before_first_cycle_stops_early(monkeypatch) -> None:
    now = {"t": 0.0}
    sleeps = []

    def fake_time() -> float:
        return now["t"]

    def fake_sleep(duration: float) -> None:
        sleeps.append(duration)
        now["t"] += duration

    monkeypatch.setattr(stability.time, "time", fake_time)
    monkeypatch.setattr(stability.time, "sleep", fake_sleep)

    logger = logging.getLogger("test_delay_before_first_cycle_stops_early")
    logger.addHandler(logging.NullHandler())
    assert delay_before_first_cycle(2.0, lambda: True, logger) is False
    assert sleeps == []


def test_wait_for_assistant_settle_waits_for_quiet() -> None:
    now = {"t": 0.0}

    class DummyMonitor:
        def __init__(self) -> None:
            self._last = 1.0

        def poll(self) -> bool:
            t = now["t"]
            if t >= 0.2:
                self._last = 2.0
            if t >= 0.45:
                self._last = 3.0
            return True

        def last_assistant_timestamp(self):
            return self._last

    def fake_time() -> float:
        return now["t"]

    def fake_sleep(duration: float) -> None:
        now["t"] += duration

    monitor = DummyMonitor()
    result = wait_for_assistant_settle(
        transcript_monitor=monitor,
        baseline_assistant_ts=1.0,
        settle_seconds=0.3,
        max_wait_seconds=2.0,
        poll_interval_s=0.1,
        stop_requested=lambda: False,
        time_fn=fake_time,
        sleep_fn=fake_sleep,
    )

    assert result == 3.0
    assert now["t"] >= 0.75


def test_wait_for_assistant_settle_returns_baseline_when_disabled() -> None:
    monitor = TranscriptMonitor(Path("/tmp"))
    assert (
        wait_for_assistant_settle(
            transcript_monitor=monitor,
            baseline_assistant_ts=4.0,
            settle_seconds=0.0,
            max_wait_seconds=1.0,
            poll_interval_s=0.1,
            stop_requested=lambda: False,
        )
        == 4.0
    )

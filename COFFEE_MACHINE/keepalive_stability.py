"""Long-running stability harness for COFFEE_MACHINE keepalive."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import random
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from COFFEE_MACHINE.keepalive import (
    EventLogTailer,
    find_latest_event_log,
    is_audio_event,
)

RUNTIME_ROOT = REPO_ROOT / "runtime"
CONVERSATIONS_DIR = RUNTIME_ROOT / "conversations"
BUTTON_PRESSES_FILE = RUNTIME_ROOT / "coffee_machine_buttons.txt"
INBOX_FILE = RUNTIME_ROOT / "inbox.txt"
ACTIONS_FILE = RUNTIME_ROOT / "actions.txt"
WATCHER_LOCK_FILE = RUNTIME_ROOT / "coffee_machine_action_watcher.lock"
KEEPALIVE_PATH = REPO_ROOT / "COFFEE_MACHINE" / "keepalive.py"
DEFAULT_AUDIO_PATH = RUNTIME_ROOT / "coffee_machine_stability_speech.aiff"
DEFAULT_LOG_PATH = RUNTIME_ROOT / "coffee_machine_keepalive_stability.log"

DEFAULT_INTERVAL_MIN_S = 1 * 60
DEFAULT_INTERVAL_MAX_S = 20 * 60
DEFAULT_SPEECH_DELAY_S = 10.0
DEFAULT_RESPONSE_TIMEOUT_S = 10.0
DEFAULT_POLL_INTERVAL_S = 1.0
DEFAULT_STARTUP_TIMEOUT_S = 60.0
DEFAULT_STARTUP_DELAY_S = 5.0
DEFAULT_REPLY_DELAY_S = 2.0
DEFAULT_TURNS_PER_CYCLE = 2
DEFAULT_LONG_TURNS = 6
DEFAULT_LONG_TURNS_EVERY = 1
DEFAULT_ASSISTANT_SETTLE_S = 1.0
DEFAULT_INPUT_MODE = "audio"
DEFAULT_PLAYBACK_MODE = "auto"
DEFAULT_KRISP_KEYWORD = "krisp"
DEFAULT_SAY_DEVICE_PREFERENCES = ("macbook pro speakers",)
DEFAULT_BUTTON_LINE = "P: Coffee Button"
DEFAULT_SPEECH_TEXT = "Hello Hello Hello Hello Hello."
DEFAULT_DEEPGRAM_TTS_MODEL = "aura-2-thalia-en"
DEFAULT_DEEPGRAM_TTS_ENCODING = "linear16"
DEFAULT_DEEPGRAM_TTS_SAMPLE_RATE = 16000
DEFAULT_DEEPGRAM_TTS_CONTAINER = "none"
DEFAULT_DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"
DEFAULT_DEEPGRAM_AUDIO_PATH = RUNTIME_ROOT / "coffee_machine_stability_speech.wav"


def _read_lock_pid(lock_path: Path) -> Optional[int]:
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _format_file_stats(paths: dict[str, Path]) -> str:
    summaries: list[str] = []
    for label, path in paths.items():
        try:
            stat = path.stat()
        except FileNotFoundError:
            summaries.append(f"{label}=missing")
            continue
        summaries.append(f"{label}=size:{stat.st_size} mtime:{stat.st_mtime:.1f}")
    return ", ".join(summaries)


def _format_ts(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


class AudioEventMonitor:
    def __init__(self, conversations_dir: Path, logger: Optional[logging.Logger] = None) -> None:
        self._conversations_dir = conversations_dir
        self._current_log: Optional[Path] = None
        self._tailer: Optional[EventLogTailer] = None
        self._last_audio_ts: Optional[float] = None
        self._logger = logger

    def close(self) -> None:
        if self._tailer:
            self._tailer.close()
            self._tailer = None

    def last_audio_timestamp(self) -> Optional[float]:
        return self._last_audio_ts

    def current_log_path(self) -> Optional[Path]:
        return self._current_log

    def refresh(self) -> bool:
        saw_audio = False
        latest_log = find_latest_event_log(self._conversations_dir)
        if latest_log and latest_log != self._current_log:
            self._current_log = latest_log
            if self._tailer:
                self._tailer.close()
            self._tailer = EventLogTailer(latest_log)
            self._tailer.open()
            existing_lines = self._tailer.read_new_lines()
            saw_audio = self._update_from_lines(existing_lines)
            if self._logger:
                self._logger.debug("Audio event log switched to %s.", latest_log)
        return saw_audio

    def sync(self) -> None:
        self.refresh()
        if not self._tailer:
            return
        self._update_from_lines(self._tailer.read_new_lines())

    def poll(self) -> bool:
        saw_audio = self.refresh()
        if not self._tailer:
            return saw_audio
        return self._update_from_lines(self._tailer.read_new_lines()) or saw_audio

    def _update_from_lines(self, lines: list[str]) -> bool:
        saw_audio = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not is_audio_event(event):
                continue
            saw_audio = True
            ts = event.get("timestamp")
            if isinstance(ts, (int, float)):
                ts_value = float(ts)
            else:
                ts_value = time.time()
            if self._last_audio_ts is None or ts_value > self._last_audio_ts:
                self._last_audio_ts = ts_value
        return saw_audio


def find_latest_transcript_log(conversations_dir: Path) -> Optional[Path]:
    if not conversations_dir.exists():
        return None
    candidates = list(conversations_dir.glob("*/transcript.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


class TranscriptMonitor:
    def __init__(self, conversations_dir: Path, logger: Optional[logging.Logger] = None) -> None:
        self._conversations_dir = conversations_dir
        self._current_log: Optional[Path] = None
        self._tailer: Optional[EventLogTailer] = None
        self._last_assistant_ts: Optional[float] = None
        self._logger = logger

    def close(self) -> None:
        if self._tailer:
            self._tailer.close()
            self._tailer = None

    def last_assistant_timestamp(self) -> Optional[float]:
        return self._last_assistant_ts

    def current_log_path(self) -> Optional[Path]:
        return self._current_log

    def refresh(self) -> bool:
        saw_assistant = False
        latest_log = find_latest_transcript_log(self._conversations_dir)
        if latest_log and latest_log != self._current_log:
            self._current_log = latest_log
            if self._tailer:
                self._tailer.close()
            self._tailer = EventLogTailer(latest_log)
            self._tailer.open()
            existing_lines = self._tailer.read_new_lines()
            saw_assistant = self._update_from_lines(existing_lines)
            if self._logger:
                self._logger.debug("Transcript log switched to %s.", latest_log)
        return saw_assistant

    def sync(self) -> None:
        self.refresh()
        if not self._tailer:
            return
        self._update_from_lines(self._tailer.read_new_lines())

    def poll(self) -> bool:
        saw_assistant = self.refresh()
        if not self._tailer:
            return saw_assistant
        return self._update_from_lines(self._tailer.read_new_lines()) or saw_assistant

    def _update_from_lines(self, lines: list[str]) -> bool:
        saw_assistant = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("role") != "assistant":
                continue
            content = event.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            saw_assistant = True
            ts = event.get("ts")
            if isinstance(ts, (int, float)):
                ts_value = float(ts)
            else:
                ts_value = time.time()
            if self._last_assistant_ts is None or ts_value > self._last_assistant_ts:
                self._last_assistant_ts = ts_value
        return saw_assistant


@dataclass(frozen=True)
class ResponseDetection:
    timestamp: float
    source: str


def parse_say_audio_devices(output: str) -> list[tuple[str, str]]:
    devices: list[tuple[str, str]] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if not parts:
            continue
        if not parts[0].isdigit():
            continue
        device_id = parts[0]
        name = " ".join(parts[1:]).strip()
        if name:
            devices.append((device_id, name))
    return devices


def list_say_audio_devices() -> list[tuple[str, str]]:
    try:
        result = subprocess.run(
            ["say", "-a", "?"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Failed to list macOS say audio devices.") from exc
    output = "\n".join([result.stdout, result.stderr]).strip()
    return parse_say_audio_devices(output)


def choose_say_audio_device(
    devices: list[tuple[str, str]],
    *,
    preferred: Optional[str],
    allow_krisp: bool,
) -> tuple[str, str]:
    if not devices:
        raise RuntimeError("No macOS say audio devices found.")

    if preferred:
        lowered = preferred.lower()
        for device_id, name in devices:
            if device_id == preferred or name.lower().startswith(lowered):
                return device_id, name
        raise RuntimeError(f"Requested audio device not found: {preferred}")

    def is_krisp(device_name: str) -> bool:
        return DEFAULT_KRISP_KEYWORD in device_name.lower()

    def is_preferred_device(device_name: str) -> bool:
        lowered = device_name.lower()
        return any(keyword in lowered for keyword in DEFAULT_SAY_DEVICE_PREFERENCES)

    if not allow_krisp:
        for device_id, name in devices:
            if is_preferred_device(name) and not is_krisp(name):
                return device_id, name

    if allow_krisp:
        return devices[0]

    for device_id, name in devices:
        if not is_krisp(name):
            return device_id, name

    raise RuntimeError("Only Krisp audio devices found; refusing to play audio.")


def select_turns(cycle_index: int, base_turns: int, long_turns: int, long_every: int) -> int:
    base = max(1, int(base_turns))
    if long_every > 0 and cycle_index % long_every == 0:
        return max(1, int(long_turns))
    return base


def choose_cycle_interval_seconds(
    *,
    fixed_interval_s: Optional[float],
    min_interval_s: float,
    max_interval_s: float,
    random_fn: Callable[[float, float], float] = random.uniform,
) -> float:
    if fixed_interval_s is not None:
        return max(0.0, fixed_interval_s)
    low = max(0.0, min_interval_s)
    high = max(0.0, max_interval_s)
    if high < low:
        low, high = high, low
    return random_fn(low, high)


def compute_followup_at(
    button_pressed_at: float,
    response_at: float,
    speech_delay_s: float,
    reply_delay_s: float,
) -> float:
    min_delay = max(0.0, speech_delay_s)
    reply_delay = max(0.0, reply_delay_s)
    return max(button_pressed_at + min_delay, response_at + reply_delay)


def sleep_until(
    target_time: float,
    *,
    poll_interval_s: float,
    stop_requested: Callable[[], bool],
) -> bool:
    while time.time() < target_time:
        if stop_requested():
            return False
        remaining = max(0.0, target_time - time.time())
        time.sleep(min(poll_interval_s, remaining))
    return True


def wait_for_response(
    *,
    monitor: AudioEventMonitor,
    transcript_monitor: TranscriptMonitor,
    baseline_audio_ts: float,
    baseline_assistant_ts: float,
    deadline: float,
    poll_interval_s: float,
    stop_requested: Callable[[], bool],
) -> Optional[ResponseDetection]:
    while time.time() < deadline and not stop_requested():
        audio_updated = monitor.poll()
        transcript_updated = transcript_monitor.poll()
        latest_audio_ts = monitor.last_audio_timestamp() or 0.0
        latest_assistant_ts = transcript_monitor.last_assistant_timestamp() or 0.0

        candidates: list[ResponseDetection] = []
        if audio_updated and latest_audio_ts > baseline_audio_ts:
            candidates.append(ResponseDetection(latest_audio_ts, "tts_audio_buffer"))
        if transcript_updated and latest_assistant_ts > baseline_assistant_ts:
            candidates.append(ResponseDetection(latest_assistant_ts, "assistant_transcript"))
        if candidates:
            return max(candidates, key=lambda item: item.timestamp)
        time.sleep(poll_interval_s)
    return None


def wait_for_assistant_settle(
    *,
    transcript_monitor: TranscriptMonitor,
    baseline_assistant_ts: float,
    settle_seconds: float,
    max_wait_seconds: float,
    poll_interval_s: float,
    stop_requested: Callable[[], bool],
    time_fn: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> float:
    if settle_seconds <= 0:
        return baseline_assistant_ts
    deadline = time_fn() + max(0.0, max_wait_seconds)
    last_ts = baseline_assistant_ts
    quiet_until = time_fn() + settle_seconds
    while time_fn() < deadline and not stop_requested():
        transcript_monitor.poll()
        latest_ts = transcript_monitor.last_assistant_timestamp() or last_ts
        if latest_ts > last_ts:
            last_ts = latest_ts
            quiet_until = time_fn() + settle_seconds
        if time_fn() >= quiet_until:
            return last_ts
        sleep_fn(poll_interval_s)
    return last_ts


def build_say_command(
    *,
    text: str,
    output_path: Path,
    voice: Optional[str],
    rate_wpm: Optional[int],
) -> list[str]:
    command = ["say", "-o", str(output_path)]
    if voice:
        command.extend(["-v", voice])
    if rate_wpm:
        command.extend(["-r", str(rate_wpm)])
    command.append(text)
    return command


def build_say_play_command(
    *,
    text: str,
    audio_device: str,
    voice: Optional[str],
    rate_wpm: Optional[int],
) -> list[str]:
    command = ["say", "-a", audio_device]
    if voice:
        command.extend(["-v", voice])
    if rate_wpm is not None:
        command.extend(["-r", str(rate_wpm)])
    command.append(text)
    return command


def resolve_audio_player(
    audio_path: Path,
    *,
    override: Optional[str] = None,
    which: Callable[[str], Optional[str]] = shutil.which,
) -> list[str]:
    if override:
        return [part.format(audio_path=str(audio_path)) for part in shlex.split(override)]
    if which("afplay"):
        return ["afplay", str(audio_path)]
    if which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", str(audio_path)]
    if which("play"):
        return ["play", str(audio_path)]
    raise RuntimeError("No supported audio player found (afplay, ffplay, play).")


def load_dotenv_if_available(
    dotenv_path: Optional[Path] = None,
    *,
    import_module: Callable[[str], object] = importlib.import_module,
) -> bool:
    try:
        module = import_module("dotenv")
    except ModuleNotFoundError:
        return False
    load_dotenv = getattr(module, "load_dotenv", None)
    if not callable(load_dotenv):
        return False
    path = dotenv_path or (REPO_ROOT / ".env")
    if path.exists():
        load_dotenv(dotenv_path=str(path))
    else:
        load_dotenv()
    return True


def build_deepgram_tts_request(
    *,
    api_key: str,
    text: str,
    model: str,
    encoding: str,
    sample_rate: int,
    container: str,
    base_url: str = DEFAULT_DEEPGRAM_TTS_URL,
) -> urllib.request.Request:
    params = {
        "model": model,
        "encoding": encoding,
        "sample_rate": str(sample_rate),
        "container": container,
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    payload = json.dumps({"text": text}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("authorization", f"token {api_key}")
    request.add_header("content-type", "application/json")
    return request


def fetch_deepgram_tts_audio(
    *,
    api_key: str,
    text: str,
    model: str,
    encoding: str,
    sample_rate: int,
    container: str,
    request_fn: Callable[..., object] = urllib.request.urlopen,
    timeout_s: float = 30.0,
) -> bytes:
    request = build_deepgram_tts_request(
        api_key=api_key,
        text=text,
        model=model,
        encoding=encoding,
        sample_rate=sample_rate,
        container=container,
    )
    try:
        response = request_fn(request, timeout=timeout_s)
        try:
            status = response.getcode()
            payload = response.read()
        finally:
            response.close()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"Deepgram TTS request failed ({exc.code}): {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Deepgram TTS request failed: {exc}") from exc

    if status != 200:
        body = payload.decode("utf-8", "replace")
        raise RuntimeError(f"Deepgram TTS request failed ({status}): {body}")
    if not payload:
        raise RuntimeError("Deepgram TTS returned empty audio.")
    return payload


def write_wav_file(audio_path: Path, audio_bytes: bytes, *, sample_rate: int) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)


def ensure_speech_audio(
    *,
    audio_path: Path,
    text: str,
    voice: Optional[str],
    rate_wpm: Optional[int],
    regenerate: bool,
) -> None:
    if audio_path.exists() and not regenerate:
        return
    say_path = shutil.which("say")
    if say_path is None:
        raise RuntimeError("macOS 'say' is required to generate the speech sample.")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_say_command(text=text, output_path=audio_path, voice=voice, rate_wpm=rate_wpm)
    subprocess.run(command, check=True)


def ensure_speech_audio_deepgram(
    *,
    audio_path: Path,
    text: str,
    api_key: str,
    model: str,
    encoding: str,
    sample_rate: int,
    container: str,
    regenerate: bool,
    fetch_audio_fn: Callable[..., bytes] = fetch_deepgram_tts_audio,
    write_wav_fn: Callable[..., None] = write_wav_file,
) -> None:
    if audio_path.exists() and not regenerate:
        return
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY must be set to generate Deepgram speech audio.")
    if encoding != "linear16":
        raise RuntimeError("Deepgram speech audio generation expects linear16 encoding.")
    if container != "none":
        raise RuntimeError("Deepgram speech audio generation expects container=none.")
    audio_bytes = fetch_audio_fn(
        api_key=api_key,
        text=text,
        model=model,
        encoding=encoding,
        sample_rate=sample_rate,
        container=container,
    )
    write_wav_fn(audio_path, audio_bytes, sample_rate=sample_rate)


def prepare_audio_command(
    args: argparse.Namespace,
    logger: logging.Logger,
    *,
    platform: str = sys.platform,
    list_say_audio_devices_fn: Callable[[], list[tuple[str, str]]] = list_say_audio_devices,
    choose_say_audio_device_fn: Callable[..., tuple[str, str]] = choose_say_audio_device,
    build_say_play_command_fn: Callable[..., list[str]] = build_say_play_command,
    ensure_speech_audio_fn: Callable[..., None] = ensure_speech_audio,
    ensure_speech_audio_deepgram_fn: Callable[..., None] = ensure_speech_audio_deepgram,
    resolve_audio_player_fn: Callable[..., list[str]] = resolve_audio_player,
) -> Optional[list[str]]:
    if args.skip_speech:
        return None

    playback_mode = args.playback_mode
    if playback_mode == "auto":
        playback_mode = "say" if platform == "darwin" else "file"

    if playback_mode == "say":
        if platform != "darwin":
            raise RuntimeError("Playback mode 'say' is only supported on macOS.")
        if args.play_command:
            logger.warning("Ignoring --play-command because playback mode is 'say'.")
        preferred_device = args.say_audio_device
        if preferred_device and preferred_device.lower() == "auto":
            preferred_device = None
        devices = list_say_audio_devices_fn()
        device_id, device_name = choose_say_audio_device_fn(
            devices,
            preferred=preferred_device,
            allow_krisp=args.allow_krisp,
        )
        logger.info("Using say audio device: %s (%s).", device_name, device_id)
        return build_say_play_command_fn(
            text=args.speech_text,
            audio_device=device_id,
            voice=args.say_voice,
            rate_wpm=args.say_rate_wpm,
        )

    audio_path = args.speech_audio
    if playback_mode == "deepgram":
        if audio_path == DEFAULT_AUDIO_PATH:
            audio_path = DEFAULT_DEEPGRAM_AUDIO_PATH
        ensure_speech_audio_deepgram_fn(
            audio_path=audio_path,
            text=args.speech_text,
            api_key=os.getenv("DEEPGRAM_API_KEY", ""),
            model=args.deepgram_tts_model,
            encoding=DEFAULT_DEEPGRAM_TTS_ENCODING,
            sample_rate=args.deepgram_tts_sample_rate,
            container=DEFAULT_DEEPGRAM_TTS_CONTAINER,
            regenerate=args.regenerate_audio,
        )
        logger.info(
            "Deepgram speech sample generated; playback uses the current system output device."
        )
    else:
        ensure_speech_audio_fn(
            audio_path=audio_path,
            text=args.speech_text,
            voice=args.say_voice,
            rate_wpm=args.say_rate_wpm,
            regenerate=args.regenerate_audio,
        )
    return resolve_audio_player_fn(
        audio_path,
        override=args.play_command,
    )


def append_button_press(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = line.rstrip("\n")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(payload)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())


def append_inbox_push(path: Path, text: str) -> None:
    payload = text.strip()
    if not payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"P: {payload}\n")
        fh.flush()
        os.fsync(fh.fileno())


def start_keepalive(extra_args: list[str]) -> subprocess.Popen[str]:
    command = [sys.executable, str(KEEPALIVE_PATH), *extra_args]
    return subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        start_new_session=True,
        stdin=subprocess.PIPE,
        text=True,
    )


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.kill()
    process.wait(timeout=5)


def stop_keepalive(process: subprocess.Popen[str], logger: logging.Logger) -> None:
    if process.poll() is not None:
        return
    try:
        if process.stdin:
            process.stdin.write("q\n")
            process.stdin.flush()
    except Exception as exc:
        logger.debug("Failed to send quit to keepalive (%s).", exc)
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        terminate_process(process)


def play_audio(command: list[str], logger: logging.Logger) -> None:
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        logger.warning("Audio playback failed (%s).", exc)


def run_cycle(
    *,
    cycle_index: int,
    monitor: AudioEventMonitor,
    transcript_monitor: TranscriptMonitor,
    button_line: str,
    turns: int,
    input_mode: str,
    inbox_text: str,
    speech_delay_s: float,
    reply_delay_s: float,
    assistant_settle_s: float,
    response_timeout_s: float,
    poll_interval_s: float,
    audio_command: Optional[list[str]],
    logger: logging.Logger,
    stop_requested: Callable[[], bool],
) -> bool:
    monitor.sync()
    transcript_monitor.sync()
    baseline_audio_ts = monitor.last_audio_timestamp() or 0.0
    baseline_assistant_ts = transcript_monitor.last_assistant_timestamp() or 0.0
    audio_log = monitor.current_log_path()
    transcript_log = transcript_monitor.current_log_path()
    logger.debug(
        "Cycle %s: baseline audio ts=%s assistant ts=%s audio_log=%s transcript_log=%s.",
        cycle_index,
        _format_ts(baseline_audio_ts),
        _format_ts(baseline_assistant_ts),
        str(audio_log) if audio_log else "none",
        str(transcript_log) if transcript_log else "none",
    )
    append_button_press(BUTTON_PRESSES_FILE, button_line)
    logger.info("Cycle %s: button press appended.", cycle_index)
    button_pressed_at = time.time()

    total_turns = max(1, int(turns))
    logger.info("Cycle %s: waiting for response (turn 1/%s).", cycle_index, total_turns)
    detection = wait_for_response(
        monitor=monitor,
        transcript_monitor=transcript_monitor,
        baseline_audio_ts=baseline_audio_ts,
        baseline_assistant_ts=baseline_assistant_ts,
        deadline=button_pressed_at + response_timeout_s,
        poll_interval_s=poll_interval_s,
        stop_requested=stop_requested,
    )
    if not detection:
        logger.warning(
            "Cycle %s: no audio response within %.1f seconds.",
            cycle_index,
            response_timeout_s,
        )
        logger.info(
            "Cycle %s: response timeout diagnostics (last_audio_ts=%s, last_assistant_ts=%s, "
            "audio_log=%s, transcript_log=%s, runtime_files=%s).",
            cycle_index,
            _format_ts(monitor.last_audio_timestamp()),
            _format_ts(transcript_monitor.last_assistant_timestamp()),
            str(monitor.current_log_path()) if monitor.current_log_path() else "none",
            str(transcript_monitor.current_log_path()) if transcript_monitor.current_log_path() else "none",
            _format_file_stats(
                {
                    "buttons": BUTTON_PRESSES_FILE,
                    "inbox": INBOX_FILE,
                    "actions": ACTIONS_FILE,
                    "watcher_lock": WATCHER_LOCK_FILE,
                }
            ),
        )
        return False
    logger.info("Cycle %s: response detected (%s) for turn 1.", cycle_index, detection.source)
    response_at = detection.timestamp
    if detection.source == "assistant_transcript":
        settle_ts = wait_for_assistant_settle(
            transcript_monitor=transcript_monitor,
            baseline_assistant_ts=response_at,
            settle_seconds=assistant_settle_s,
            max_wait_seconds=min(5.0, response_timeout_s),
            poll_interval_s=poll_interval_s,
            stop_requested=stop_requested,
        )
        if settle_ts > response_at:
            logger.debug(
                "Cycle %s: assistant settled after %.2f seconds (turn 1).",
                cycle_index,
                settle_ts - response_at,
            )
            response_at = settle_ts

    if total_turns <= 1:
        return True

    if input_mode == "audio" and not audio_command:
        logger.warning(
            "Cycle %s: audio playback disabled; skipping %s follow-up turns.",
            cycle_index,
            total_turns - 1,
        )
        return True
    if input_mode == "inbox" and not inbox_text.strip():
        logger.warning(
            "Cycle %s: inbox text empty; skipping %s follow-up turns.",
            cycle_index,
            total_turns - 1,
        )
        return True

    next_input_at = compute_followup_at(
        button_pressed_at,
        response_at,
        speech_delay_s,
        reply_delay_s,
    )

    turn_index = 2
    while turn_index <= total_turns and not stop_requested():
        if not sleep_until(
            next_input_at,
            poll_interval_s=poll_interval_s,
            stop_requested=stop_requested,
        ):
            return False
        if input_mode == "audio":
            logger.info("Cycle %s: playing speech sample (turn %s/%s).", cycle_index, turn_index, total_turns)
            play_audio(audio_command, logger)
        else:
            logger.info("Cycle %s: injecting inbox text (turn %s/%s).", cycle_index, turn_index, total_turns)
            append_inbox_push(INBOX_FILE, inbox_text)

        baseline_audio_ts = monitor.last_audio_timestamp() or baseline_audio_ts
        baseline_assistant_ts = transcript_monitor.last_assistant_timestamp() or baseline_assistant_ts

        logger.info("Cycle %s: waiting for response (turn %s/%s).", cycle_index, turn_index, total_turns)
        detection = wait_for_response(
            monitor=monitor,
            transcript_monitor=transcript_monitor,
            baseline_audio_ts=baseline_audio_ts,
            baseline_assistant_ts=baseline_assistant_ts,
            deadline=time.time() + response_timeout_s,
            poll_interval_s=poll_interval_s,
            stop_requested=stop_requested,
        )
        if not detection:
            logger.warning(
                "Cycle %s: no audio response within %.1f seconds (turn %s/%s).",
                cycle_index,
                response_timeout_s,
                turn_index,
                total_turns,
            )
            logger.info(
                "Cycle %s: response timeout diagnostics (turn %s/%s, last_audio_ts=%s, "
                "last_assistant_ts=%s, audio_log=%s, transcript_log=%s, runtime_files=%s).",
                cycle_index,
                turn_index,
                total_turns,
                _format_ts(monitor.last_audio_timestamp()),
                _format_ts(transcript_monitor.last_assistant_timestamp()),
                str(monitor.current_log_path()) if monitor.current_log_path() else "none",
                str(transcript_monitor.current_log_path()) if transcript_monitor.current_log_path() else "none",
                _format_file_stats(
                    {
                        "buttons": BUTTON_PRESSES_FILE,
                        "inbox": INBOX_FILE,
                        "actions": ACTIONS_FILE,
                        "watcher_lock": WATCHER_LOCK_FILE,
                    }
                ),
            )
            return False
        logger.info(
            "Cycle %s: response detected (%s) for turn %s/%s.",
            cycle_index,
            detection.source,
            turn_index,
            total_turns,
        )
        response_at = detection.timestamp
        if detection.source == "assistant_transcript":
            settle_ts = wait_for_assistant_settle(
                transcript_monitor=transcript_monitor,
                baseline_assistant_ts=response_at,
                settle_seconds=assistant_settle_s,
                max_wait_seconds=min(5.0, response_timeout_s),
                poll_interval_s=poll_interval_s,
                stop_requested=stop_requested,
            )
            if settle_ts > response_at:
                logger.debug(
                    "Cycle %s: assistant settled after %.2f seconds (turn %s/%s).",
                    cycle_index,
                    settle_ts - response_at,
                    turn_index,
                    total_turns,
                )
                response_at = settle_ts
        turn_index += 1
        next_input_at = response_at + max(0.0, reply_delay_s)
    return True


def wait_for_runtime_ready(started_at: float, timeout_s: float, logger: logging.Logger) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        buttons_ready = False
        if BUTTON_PRESSES_FILE.exists():
            try:
                buttons_ready = BUTTON_PRESSES_FILE.stat().st_mtime >= started_at
            except FileNotFoundError:
                buttons_ready = False
        lock_pid = _read_lock_pid(WATCHER_LOCK_FILE)
        watcher_ready = bool(lock_pid and _is_process_alive(lock_pid))
        if WATCHER_LOCK_FILE.exists() and not watcher_ready:
            logger.debug("Watcher lock present but not alive (pid=%s).", lock_pid)
        if buttons_ready and watcher_ready:
            logger.info("Runtime ready for button presses.")
            return True
        time.sleep(0.5)
    logger.warning(
        "Timed out waiting for runtime readiness (buttons_ready=%s, watcher_ready=%s, watcher_pid=%s).",
        BUTTON_PRESSES_FILE.exists(),
        watcher_ready,
        _read_lock_pid(WATCHER_LOCK_FILE),
    )
    return False


def delay_before_first_cycle(
    delay_s: float,
    stop_requested: Callable[[], bool],
    logger: logging.Logger,
) -> bool:
    if delay_s <= 0:
        return True
    logger.info("Waiting %.1f seconds before first cycle.", delay_s)
    deadline = time.time() + delay_s
    while time.time() < deadline:
        if stop_requested():
            return False
        remaining = max(0.0, deadline - time.time())
        time.sleep(min(0.5, remaining))
    return True


def setup_logging(log_path: Path, verbose: bool) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("coffee_machine_stability")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Long-running stability runner for COFFEE_MACHINE keepalive.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=None,
        help="Fixed seconds between simulation cycles (overrides randomized intervals).",
    )
    parser.add_argument(
        "--min-interval-seconds",
        type=float,
        default=DEFAULT_INTERVAL_MIN_S,
        help="Minimum seconds between simulation cycles when randomized.",
    )
    parser.add_argument(
        "--max-interval-seconds",
        type=float,
        default=DEFAULT_INTERVAL_MAX_S,
        help="Maximum seconds between simulation cycles when randomized.",
    )
    parser.add_argument(
        "--speech-delay-seconds",
        type=float,
        default=DEFAULT_SPEECH_DELAY_S,
        help="Minimum seconds after the button press to inject the first follow-up speech.",
    )
    parser.add_argument(
        "--reply-delay-seconds",
        type=float,
        default=DEFAULT_REPLY_DELAY_S,
        help="Seconds to wait after a response before injecting follow-up speech.",
    )
    parser.add_argument(
        "--assistant-settle-seconds",
        type=float,
        default=DEFAULT_ASSISTANT_SETTLE_S,
        help="Seconds of quiet transcript activity before sending the next follow-up turn.",
    )
    parser.add_argument(
        "--turns-per-cycle",
        type=int,
        default=DEFAULT_TURNS_PER_CYCLE,
        help="Total turns per cycle, including the initial button press.",
    )
    parser.add_argument(
        "--long-turns",
        type=int,
        default=DEFAULT_LONG_TURNS,
        help="Total turns when running a long conversation cycle.",
    )
    parser.add_argument(
        "--long-turns-every",
        type=int,
        default=DEFAULT_LONG_TURNS_EVERY,
        help="Run a long conversation cycle every N cycles (0 disables).",
    )
    parser.add_argument(
        "--response-timeout-seconds",
        type=float,
        default=DEFAULT_RESPONSE_TIMEOUT_S,
        help="Seconds to wait for an audio response before logging a failure.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=DEFAULT_POLL_INTERVAL_S,
        help="Polling interval for audio responses.",
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT_S,
        help="Seconds to wait for the boot/runtime files to appear before the first cycle.",
    )
    parser.add_argument(
        "--startup-delay-seconds",
        type=float,
        default=DEFAULT_STARTUP_DELAY_S,
        help="Extra delay after startup before the first cycle begins.",
    )
    parser.add_argument(
        "--button-line",
        default=DEFAULT_BUTTON_LINE,
        help="Line appended to runtime/coffee_machine_buttons.txt each cycle.",
    )
    parser.add_argument(
        "--speech-text",
        default=DEFAULT_SPEECH_TEXT,
        help="Text used to generate speech samples or inbox injections.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("audio", "inbox"),
        default=DEFAULT_INPUT_MODE,
        help="Inject follow-up turns via audio playback or direct inbox text.",
    )
    parser.add_argument(
        "--playback-mode",
        choices=("auto", "file", "say", "deepgram"),
        default=DEFAULT_PLAYBACK_MODE,
        help="Playback mode for speech samples (audio input mode only).",
    )
    parser.add_argument(
        "--say-audio-device",
        help=(
            "macOS say audio device ID or name prefix (default: prefer MacBook Pro "
            "Speakers, otherwise first non-Krisp)."
        ),
    )
    parser.add_argument(
        "--allow-krisp",
        action="store_true",
        help="Allow selecting a Krisp output device for playback.",
    )
    parser.add_argument(
        "--speech-audio",
        type=Path,
        default=DEFAULT_AUDIO_PATH,
        help="Path to the speech audio file to play.",
    )
    parser.add_argument(
        "--say-voice",
        help="Optional macOS say voice name.",
    )
    parser.add_argument(
        "--say-rate-wpm",
        type=int,
        help="Optional macOS say speech rate in WPM.",
    )
    parser.add_argument(
        "--deepgram-tts-model",
        default=DEFAULT_DEEPGRAM_TTS_MODEL,
        help="Deepgram TTS model/voice used for playback-mode deepgram.",
    )
    parser.add_argument(
        "--deepgram-tts-sample-rate",
        type=int,
        default=DEFAULT_DEEPGRAM_TTS_SAMPLE_RATE,
        help="Deepgram TTS sample rate used for playback-mode deepgram.",
    )
    parser.add_argument(
        "--regenerate-audio",
        action="store_true",
        help="Regenerate the speech audio sample even if it exists.",
    )
    parser.add_argument(
        "--play-command",
        help="Override audio playback command (use {audio_path} placeholder).",
    )
    parser.add_argument(
        "--skip-speech",
        action="store_true",
        help="Skip speech playback entirely (audio input mode only).",
    )
    parser.add_argument(
        "--keepalive-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed to COFFEE_MACHINE/keepalive.py after --.",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run a single simulation cycle and exit.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to the stability log file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level console logging.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv_if_available()
    args = parse_args()
    logger = setup_logging(args.log_path, args.verbose)

    stop_requested = False

    def _handle_signal(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        logger.info("Stop requested; shutting down.")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info(
        "Runtime paths: buttons=%s, inbox=%s, actions=%s, conversations=%s, watcher_lock=%s.",
        BUTTON_PRESSES_FILE,
        INBOX_FILE,
        ACTIONS_FILE,
        CONVERSATIONS_DIR,
        WATCHER_LOCK_FILE,
    )
    input_mode = args.input_mode
    audio_command: Optional[list[str]] = None
    if input_mode == "inbox":
        if args.skip_speech:
            logger.info("Input mode inbox ignores --skip-speech.")
        if args.play_command or args.playback_mode != DEFAULT_PLAYBACK_MODE:
            logger.warning("Input mode inbox ignores playback settings.")
    else:
        audio_command = prepare_audio_command(args, logger)

    logger.info("Input mode: %s.", input_mode)

    extra_keepalive_args = list(args.keepalive_args)
    if extra_keepalive_args[:1] == ["--"]:
        extra_keepalive_args = extra_keepalive_args[1:]
    keepalive_started_at = time.time()
    keepalive = start_keepalive(extra_keepalive_args)
    logger.info("keepalive.py started (pid=%s).", keepalive.pid)

    monitor = AudioEventMonitor(CONVERSATIONS_DIR, logger=logger)
    transcript_monitor = TranscriptMonitor(CONVERSATIONS_DIR, logger=logger)
    cycle_index = 0

    try:
        wait_for_runtime_ready(keepalive_started_at, args.startup_timeout_seconds, logger)
        if not delay_before_first_cycle(args.startup_delay_seconds, lambda: stop_requested, logger):
            return 0

        while not stop_requested:
            if keepalive.poll() is not None:
                logger.error("keepalive.py exited (code=%s).", keepalive.returncode)
                return 1

            cycle_index += 1
            turns = select_turns(
                cycle_index,
                args.turns_per_cycle,
                args.long_turns,
                args.long_turns_every,
            )
            run_cycle(
                cycle_index=cycle_index,
                monitor=monitor,
                transcript_monitor=transcript_monitor,
                button_line=args.button_line,
                turns=turns,
                input_mode=input_mode,
                inbox_text=args.speech_text,
                speech_delay_s=args.speech_delay_seconds,
                reply_delay_s=args.reply_delay_seconds,
                assistant_settle_s=args.assistant_settle_seconds,
                response_timeout_s=args.response_timeout_seconds,
                poll_interval_s=args.poll_interval_seconds,
                audio_command=audio_command,
                logger=logger,
                stop_requested=lambda: stop_requested,
            )

            if args.run_once:
                logger.info("Run-once complete; exiting.")
                break

            interval_s = choose_cycle_interval_seconds(
                fixed_interval_s=args.interval_seconds,
                min_interval_s=args.min_interval_seconds,
                max_interval_s=args.max_interval_seconds,
            )
            logger.info("Sleeping %.1f seconds before next cycle.", interval_s)
            next_cycle_at = time.time() + interval_s
            while time.time() < next_cycle_at and not stop_requested:
                if keepalive.poll() is not None:
                    logger.error("keepalive.py exited (code=%s).", keepalive.returncode)
                    return 1
                time.sleep(1.0)
    finally:
        monitor.close()
        transcript_monitor.close()
        stop_keepalive(keepalive, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Keep the COFFEE_MACHINE boot process alive and restart on silent audio."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CONVERSATIONS = REPO_ROOT / "runtime" / "conversations"
BOOT_PATH = REPO_ROOT / "COFFEE_MACHINE" / "boot.py"

AUDIO_EVENT_NAME = "tts_audio_buffer"
DEFAULT_SILENCE_TIMEOUT_S = 10 * 60
DEFAULT_CHECK_INTERVAL_S = 2.0


def is_quit_command(line: str) -> bool:
    return line.strip().lower() == "q"


def find_latest_event_log(conversations_dir: Path) -> Optional[Path]:
    if not conversations_dir.exists():
        return None
    candidates = list(conversations_dir.glob("*/event_log.ndjson"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_latest_transcript_log(conversations_dir: Path) -> Optional[Path]:
    if not conversations_dir.exists():
        return None
    candidates = list(conversations_dir.glob("*/transcript.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def is_audio_event(event: dict) -> bool:
    if event.get("event") != AUDIO_EVENT_NAME:
        return False
    audio_bytes = event.get("audio_bytes")
    return isinstance(audio_bytes, int) and audio_bytes > 0


def latest_audio_timestamp_from_lines(lines: Iterable[str]) -> Optional[float]:
    latest = None
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
        ts = event.get("timestamp")
        if isinstance(ts, (int, float)):
            latest = float(ts)
    return latest


def latest_assistant_timestamp_from_lines(lines: Iterable[str]) -> Optional[float]:
    latest = None
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
        ts = event.get("ts")
        if isinstance(ts, (int, float)):
            latest = float(ts)
    return latest


@dataclass
class EventLogTailer:
    path: Path
    _handle: Optional[object] = None

    def open(self) -> None:
        if self._handle:
            try:
                self._handle.close()
            except Exception:
                pass
        self._handle = self.path.open("r", encoding="utf-8")

    def read_new_lines(self) -> list[str]:
        if not self._handle:
            self.open()
        assert self._handle is not None
        lines = []
        while True:
            line = self._handle.readline()
            if not line:
                break
            lines.append(line)
        return lines

    def close(self) -> None:
        if self._handle:
            try:
                self._handle.close()
            except Exception:
                pass
            self._handle = None


class BootSupervisor:
    def __init__(
        self,
        *,
        silence_timeout_s: float,
        check_interval_s: float,
        conversations_dir: Path,
        boot_path: Path,
    ) -> None:
        self._silence_timeout_s = silence_timeout_s
        self._check_interval_s = check_interval_s
        self._conversations_dir = conversations_dir
        self._boot_path = boot_path
        self._stop = False
        self._current_log: Optional[Path] = None
        self._tailer: Optional[EventLogTailer] = None
        self._current_transcript: Optional[Path] = None
        self._transcript_tailer: Optional[EventLogTailer] = None
        self._current_process: Optional[subprocess.Popen[str]] = None

    def run(self) -> None:
        try:
            while not self._stop:
                self._current_process = self._start_boot()
                last_audio_at = time.time()
                reason = self._monitor(self._current_process, last_audio_at)
                if self._stop:
                    self._terminate_process(self._current_process)
                    return
                print(f"[keepalive] Restarting boot.py after {reason}.")
        finally:
            if self._current_process:
                self._terminate_process(self._current_process)
            if self._tailer:
                self._tailer.close()
            if self._transcript_tailer:
                self._transcript_tailer.close()

    def stop(self, *_args: object) -> None:
        self._stop = True
        if self._current_process:
            self._terminate_process(self._current_process)

    def _start_boot(self) -> subprocess.Popen[str]:
        print("[keepalive] Starting boot.py.")
        return subprocess.Popen(
            [sys.executable, str(self._boot_path)],
            cwd=str(REPO_ROOT),
            start_new_session=True,
        )

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        for sig, timeout_s in (
            (signal.SIGINT, 3),
            (signal.SIGTERM, 3),
        ):
            if process.poll() is not None:
                return
            try:
                process.send_signal(sig)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=timeout_s)
                return
            except subprocess.TimeoutExpired:
                continue
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.kill()
        process.wait(timeout=5)

    def _monitor(self, process: subprocess.Popen[str], last_audio_at: float) -> str:
        while not self._stop:
            if process.poll() is not None:
                return "process exit"
            last_audio_at = self._update_last_audio(last_audio_at)
            if time.time() - last_audio_at > self._silence_timeout_s:
                self._terminate_process(process)
                return "silence timeout"
            time.sleep(self._check_interval_s)
        return "stop requested"

    def _update_last_audio(self, last_audio_at: float) -> float:
        latest_log = find_latest_event_log(self._conversations_dir)
        if latest_log and latest_log != self._current_log:
            self._current_log = latest_log
            if self._tailer:
                self._tailer.close()
            self._tailer = EventLogTailer(latest_log)
            self._tailer.open()
            existing_lines = self._tailer.read_new_lines()
            latest_ts = latest_audio_timestamp_from_lines(existing_lines)
            if latest_ts:
                last_audio_at = max(last_audio_at, latest_ts)

        if not self._tailer:
            pass
        else:
            for line in self._tailer.read_new_lines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not is_audio_event(event):
                    continue
                ts = event.get("timestamp")
                if isinstance(ts, (int, float)):
                    last_audio_at = max(last_audio_at, float(ts))
                else:
                    last_audio_at = time.time()

        latest_transcript = find_latest_transcript_log(self._conversations_dir)
        if latest_transcript and latest_transcript != self._current_transcript:
            self._current_transcript = latest_transcript
            if self._transcript_tailer:
                self._transcript_tailer.close()
            self._transcript_tailer = EventLogTailer(latest_transcript)
            self._transcript_tailer.open()
            existing_lines = self._transcript_tailer.read_new_lines()
            latest_ts = latest_assistant_timestamp_from_lines(existing_lines)
            if latest_ts:
                last_audio_at = max(last_audio_at, latest_ts)

        if not self._transcript_tailer:
            return last_audio_at

        latest_ts = latest_assistant_timestamp_from_lines(self._transcript_tailer.read_new_lines())
        if latest_ts:
            last_audio_at = max(last_audio_at, latest_ts)
        return last_audio_at


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep COFFEE_MACHINE boot.py alive.")
    parser.add_argument(
        "--silence-timeout-seconds",
        type=float,
        default=DEFAULT_SILENCE_TIMEOUT_S,
        help="Restart if no audio output is detected for this many seconds.",
    )
    parser.add_argument(
        "--check-interval-seconds",
        type=float,
        default=DEFAULT_CHECK_INTERVAL_S,
        help="Polling interval for process and audio checks.",
    )
    return parser.parse_args()


def _stdin_watcher(supervisor: BootSupervisor) -> None:
    try:
        for line in sys.stdin:
            if is_quit_command(line):
                supervisor.stop()
                break
    except Exception:
        supervisor.stop()


def main() -> None:
    args = parse_args()
    supervisor = BootSupervisor(
        silence_timeout_s=args.silence_timeout_seconds,
        check_interval_s=args.check_interval_seconds,
        conversations_dir=RUNTIME_CONVERSATIONS,
        boot_path=BOOT_PATH,
    )
    signal.signal(signal.SIGINT, supervisor.stop)
    signal.signal(signal.SIGTERM, supervisor.stop)
    input_thread = threading.Thread(
        target=_stdin_watcher,
        args=(supervisor,),
        daemon=True,
    )
    input_thread.start()
    supervisor.run()


if __name__ == "__main__":
    main()

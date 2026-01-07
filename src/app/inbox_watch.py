from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

from .logging_io import EventLogger

InboxCallback = Callable[[str, str], None]


class InboxWatcher:
    def __init__(
        self,
        inbox_path: Path,
        callback: InboxCallback,
        poll_interval: float = 0.1,
        event_logger: EventLogger | None = None,
    ):
        self._inbox_path = inbox_path
        self._callback = callback
        self._poll_interval = poll_interval
        self._event_logger = event_logger
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._offset = 0
        self._last_mtime: Optional[float] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1)

    def _reset_offset_if_truncated(self) -> None:
        try:
            stat = self._inbox_path.stat()
        except FileNotFoundError:
            return
        size = stat.st_size
        mtime = stat.st_mtime
        if size < self._offset or (
            self._last_mtime is not None and mtime > self._last_mtime and size <= self._offset
        ):
            self._offset = 0
            if self._event_logger:
                self._event_logger.emit(
                    "inbox_truncated",
                    {"size": size, "mtime": mtime, "timestamp": time.time()},
                )
        self._last_mtime = mtime

    def _run(self) -> None:
        self._inbox_path.parent.mkdir(parents=True, exist_ok=True)
        self._inbox_path.touch(exist_ok=True)
        while not self._stop_event.is_set():
            self._reset_offset_if_truncated()
            with self._inbox_path.open("r", encoding="utf-8") as fh:
                fh.seek(self._offset)
                for line in iter(fh.readline, ""):
                    self._offset = fh.tell()
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("P:"):
                        payload = line[2:].strip()
                        self._emit("push", payload)
                    elif line.startswith("A:"):
                        payload = line[2:].strip()
                        self._emit("append", payload)
                    else:
                        self._emit("push", line)
            time.sleep(self._poll_interval)

    def _emit(self, mode: str, payload: str) -> None:
        if self._event_logger:
            self._event_logger.emit(
                "inbox_push",
                {"mode": mode, "payload": payload, "timestamp": time.time()},
            )
        self._callback(mode, payload)

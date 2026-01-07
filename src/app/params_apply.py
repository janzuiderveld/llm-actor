from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .config import ConfigManager
from .history import ConversationHistory
from .logging_io import EventLogger

ParamsCallback = Callable[[Dict[str, object]], None]


class ParamsWatcher:
    def __init__(
        self,
        path: Path,
        config: ConfigManager,
        history: ConversationHistory,
        apply_callback: ParamsCallback,
        poll_interval: float = 0.2,
        event_logger: EventLogger | None = None,
    ):
        self._path = path
        self._config = config
        self._history = history
        self._apply_callback = apply_callback
        self._poll_interval = poll_interval
        self._event_logger = event_logger
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._offset = 0
        self._pending: List[Dict[str, object]] = []
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

    def _run(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)
        while not self._stop_event.is_set():
            self._reset_offset_if_truncated()
            with self._path.open("r", encoding="utf-8") as fh:
                fh.seek(self._offset)
                for line in iter(fh.readline, ""):
                    self._offset = fh.tell()
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        if isinstance(payload, dict):
                            self._pending.append(payload)
                    except json.JSONDecodeError:
                        if self._event_logger:
                            self._event_logger.emit("params_invalid", {"line": line})
            time.sleep(self._poll_interval)

    def _reset_offset_if_truncated(self) -> None:
        try:
            stat = self._path.stat()
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
                    "params_truncated",
                    {"size": size, "mtime": mtime, "timestamp": time.time()},
                )
        self._last_mtime = mtime

    def drain_pending(self) -> None:
        if not self._pending:
            return
        batch = self._pending
        self._pending = []
        for payload in batch:
            self._apply(payload)
        if self._event_logger:
            self._event_logger.emit("params_applied", {"count": len(batch)})

    def _apply(self, payload: Dict[str, object]) -> None:
        op = payload.get("op")
        if not isinstance(op, str):
            return

        if op == "llm.set":
            updates = {
                k: payload[k]
                for k in ("model", "temperature", "max_tokens", "thinking_level")
                if k in payload
            }
            self._config.apply_updates(llm=updates)
            self._apply_callback({"llm": updates})
        elif op == "llm.system":
            if "text" in payload:
                text = payload.get("text")
                if isinstance(text, str) or text is None:
                    self._config.apply_updates(llm={"system_prompt": text})
                    self._apply_callback({"llm": {"system_prompt": text}})
                    self._history.set_system_message(text)
        elif op == "history.reset":
            current_prompt = self._config.config.llm.system_prompt
            self._history.reset(system_prompt=current_prompt)
        elif op == "history.append":
            role = payload.get("role")
            content = payload.get("content")
            if isinstance(role, str) and isinstance(content, str):
                self._history.add(role, content)
        elif op == "stt.flux":
            updates = {
                k: payload[k]
                for k in ("eager_eot_threshold", "eot_threshold", "eot_timeout_ms")
                if k in payload
            }
            self._config.apply_updates(stt=updates)
            self._apply_callback({"stt": updates})
        elif op == "tts.set":
            updates = {k: payload[k] for k in ("voice", "encoding", "sample_rate") if k in payload}
            self._config.apply_updates(tts=updates)
            self._apply_callback({"tts": updates})

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

_LOCKS: Dict[Path, threading.Lock] = {}


def _lock_for(path: Path) -> threading.Lock:
    if path not in _LOCKS:
        _LOCKS[path] = threading.Lock()
    return _LOCKS[path]


def _atomic_append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _lock_for(path)
    with lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(text)
            if not text.endswith("\n"):
                fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())


@dataclass
class TranscriptWriter:
    path: Path

    def append(self, record: Dict[str, Any]) -> None:
        _atomic_append(self.path, json.dumps(record))


@dataclass
class EventLogger:
    path: Path

    def emit(self, event: str, data: Dict[str, Any] | None = None) -> None:
        payload = {"event": event}
        if data:
            payload.update(data)
        _atomic_append(self.path, json.dumps(payload))

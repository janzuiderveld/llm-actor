from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List

from .logging_io import TranscriptWriter


@dataclass
class HistoryEntry:
    role: str
    content: str

    def as_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class ConversationHistory:
    def __init__(self, transcript_path: Path, max_messages: int = 50):
        self._buffer: Deque[HistoryEntry] = deque(maxlen=max_messages)
        self._transcript = TranscriptWriter(transcript_path)

    def add(self, role: str, content: str) -> None:
        entry = HistoryEntry(role=role, content=content)
        self._buffer.append(entry)
        self._transcript.append({
            "ts": time.time(),
            "role": role,
            "content": content,
        })

    def reset(self) -> None:
        self._buffer.clear()

    def extend(self, entries: Iterable[Dict[str, str]]) -> None:
        for entry in entries:
            if "role" in entry and "content" in entry:
                self.add(entry["role"], entry["content"])

    def export(self) -> List[Dict[str, str]]:
        return [entry.as_dict() for entry in self._buffer]

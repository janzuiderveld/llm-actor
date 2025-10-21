from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List

from .logging_io import TranscriptWriter


@dataclass
class HistoryEntry:
    role: str
    content: str = ""
    chunks: list[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

    def append_chunk(self, text: str) -> None:
        self.chunks.append(text)
        self.content = "".join(self.chunks)

    def replace_last_chunk(self, text: str) -> None:
        if not self.chunks:
            self.chunks.append(text)
            self.content = text
            return

        prefix = "".join(self.chunks[:-1])
        if prefix and text.startswith(prefix):
            # Final aggregation contains entire message; reset chunks.
            self.chunks = [text]
            self.content = text
            return

        old_chunk = self.chunks[-1]
        text = self._preserve_whitespace(old_chunk, text)
        self.chunks[-1] = text
        self.content = "".join(self.chunks)

    @staticmethod
    def _preserve_whitespace(old: str, new: str) -> str:
        if not old:
            return new

        prefix = ""
        suffix = ""

        i = 0
        while i < len(old) and old[i].isspace():
            prefix += old[i]
            i += 1

        j = len(old)
        while j > 0 and old[j - 1].isspace():
            suffix = old[j - 1] + suffix
            j -= 1

        if prefix and not new.startswith(prefix):
            new = prefix + new
        if suffix and not new.endswith(suffix):
            new = new + suffix
        return new


class ConversationHistory:
    def __init__(
        self,
        transcript_path: Path,
        *,
        clean_transcript_path: Path | None = None,
        context_path: Path | None = None,
        max_messages: int = 50,
    ):
        self._buffer: Deque[HistoryEntry] = deque(maxlen=max_messages)
        self._transcript = TranscriptWriter(transcript_path)
        self._clean_transcript_path = clean_transcript_path
        self._context_writer = TranscriptWriter(context_path) if context_path else None

    def add(self, role: str, content: str, *, replace_last: bool = False) -> None:
        if not content:
            return

        replaced = False
        if replace_last and self._buffer and self._buffer[-1].role == role:
            entry = self._buffer[-1]
            entry.replace_last_chunk(content)
            replaced = True
        else:
            entry = HistoryEntry(role=role)
            entry.append_chunk(content)
            self._buffer.append(entry)

        record = {
            "ts": time.time(),
            "role": role,
            "content": content,
        }
        if replaced:
            record["replace"] = True
        self._transcript.append(record)
        self._write_clean_transcript()
        self._write_context_snapshot()

    def add_partial(self, role: str, content: str) -> None:
        if not content:
            return

        if self._buffer and self._buffer[-1].role == role:
            entry = self._buffer[-1]
        else:
            entry = HistoryEntry(role=role)
            self._buffer.append(entry)

        delta = self._extract_partial_delta(entry.content, content)
        if not delta:
            return

        for chunk in self._split_partial_content(delta):
            entry.append_chunk(chunk)

            self._transcript.append({
                "ts": time.time(),
                "role": role,
                "content": chunk,
                "partial": True,
            })

    @staticmethod
    def _extract_partial_delta(existing: str, new_text: str) -> str:
        if not new_text:
            return ""
        if not existing:
            return new_text
        if new_text.startswith(existing):
            return new_text[len(existing) :]
        if existing.startswith(new_text):
            return ""

        max_len = min(len(existing), len(new_text))
        prefix_len = 0
        while prefix_len < max_len and existing[prefix_len] == new_text[prefix_len]:
            prefix_len += 1
        if prefix_len:
            return new_text[prefix_len:]
        return new_text

    @staticmethod
    def _split_partial_content(text: str) -> list[str]:
        if not text:
            return []

        subsentence_delims = {";", ":", "?", "!", "\u2013", "\u2014"}
        ellipsis_char = "\u2026"
        chunks: list[str] = []
        buffer: list[str] = []
        in_action = False
        length = len(text)

        def flush() -> None:
            if buffer:
                chunks.append("".join(buffer))
                buffer.clear()

        i = 0
        while i < length:
            char = text[i]

            if not in_action and char == "<":
                flush()
                in_action = True
                buffer.append(char)
                i += 1
                continue

            if in_action and char == ">":
                buffer.append(char)
                flush()
                in_action = False
                i += 1
                continue

            buffer.append(char)

            if not in_action:
                if char in ("\n", "\r"):
                    flush()
                    i += 1
                    continue

                if text.startswith("...", i):
                    buffer.extend([".", "."])
                    j = i + 3
                    while j < length and text[j] == ".":
                        buffer.append(".")
                        j += 1
                    while j < length and text[j].isspace() and text[j] not in ("\n", "\r"):
                        buffer.append(text[j])
                        j += 1
                    flush()
                    i = j
                    continue

                if char == ".":
                    next_char = text[i + 1] if i + 1 < length else ""
                    if next_char == "." or next_char.isdigit():
                        i += 1
                        continue
                    j = i + 1
                    while j < length and text[j].isspace() and text[j] not in ("\n", "\r"):
                        buffer.append(text[j])
                        j += 1
                    flush()
                    i = j
                    continue

                if char == "," or char in subsentence_delims or char == ellipsis_char:
                    j = i + 1
                    while j < length and text[j].isspace() and text[j] not in ("\n", "\r"):
                        buffer.append(text[j])
                        j += 1
                    flush()
                    i = j
                    continue

            i += 1

        flush()
        return [chunk for chunk in chunks if chunk]


    def reset(self) -> None:
        self._buffer.clear()
        self._write_clean_transcript()
        self._write_context_snapshot()

    def extend(self, entries: Iterable[Dict[str, str]]) -> None:
        for entry in entries:
            if "role" in entry and "content" in entry:
                self.add(entry["role"], entry["content"])

    def export(self) -> List[Dict[str, str]]:
        return [entry.as_dict() for entry in self._buffer]

    def _write_clean_transcript(self) -> None:
        if not self._clean_transcript_path:
            return

        self._clean_transcript_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(entry.as_dict()) for entry in self._buffer]
        if lines:
            payload = "\n".join(lines) + "\n"
        else:
            payload = ""
        self._clean_transcript_path.write_text(payload, encoding="utf-8")

    def _write_context_snapshot(self) -> None:
        if not self._context_writer:
            return

        snapshot = {
            "ts": time.time(),
            "context": self.export(),
        }
        self._context_writer.append(snapshot)

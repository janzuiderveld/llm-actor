from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, cast

from .logging_io import TranscriptWriter

_SYSTEM_UNCHANGED = object()

_SENTENCE_PATTERN = re.compile(r"\s*[^.!?]*[.!?]")
_PUNCTUATION_SPACE_PATTERN = re.compile(r"\s+([,.;:!?\"'’])")
_MULTISPACE_PATTERN = re.compile(r"\s+")


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

    def replace_last_chunk(self, text: str) -> str:
        if not self.chunks:
            combined = text
        else:
            prefix = "".join(self.chunks[:-1])
            if prefix and text.startswith(prefix):
                combined = text
            else:
                old_chunk = self.chunks[-1]
                updated_tail = self._preserve_whitespace(old_chunk, text)
                combined = prefix + updated_tail

        deduped = self._dedupe_repeated_sentences(combined)
        if deduped:
            self.chunks = [deduped]
            self.content = deduped
        else:
            # Preserve empty aggregation state if content collapsed to empty.
            self.chunks = []
            self.content = ""
        return self.content

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

    @staticmethod
    def _normalize_sentence(sentence: str) -> str:
        stripped = sentence.strip()
        if not stripped:
            return ""
        stripped = _PUNCTUATION_SPACE_PATTERN.sub(r"\1", stripped)
        stripped = _MULTISPACE_PATTERN.sub(" ", stripped)
        return stripped

    @classmethod
    def _dedupe_repeated_sentences(cls, text: str) -> str:
        if not text:
            return text

        segments: List[str] = []
        last_end = 0
        for match in _SENTENCE_PATTERN.finditer(text):
            segments.append(match.group(0))
            last_end = match.end()
        remainder = text[last_end:]

        normalized_segments = [cls._normalize_sentence(segment) for segment in segments]

        while True:
            repeat_len = cls._find_repeated_prefix(normalized_segments)
            if not repeat_len:
                break
            segments = segments[:repeat_len] + segments[repeat_len * 2 :]
            normalized_segments = normalized_segments[:repeat_len] + normalized_segments[repeat_len * 2 :]

        deduped_segments: List[str] = []
        previous_normalized: Optional[str] = None

        for segment, normalized in zip(segments, normalized_segments):
            if normalized and previous_normalized == normalized:
                continue
            deduped_segments.append(segment)
            if normalized:
                previous_normalized = normalized

        deduped_segments.append(remainder)
        return "".join(deduped_segments)

    @staticmethod
    def _find_repeated_prefix(items: List[str]) -> int:
        max_len = len(items) // 2
        for candidate in range(max_len, 0, -1):
            if not any(items[:candidate]):
                continue
            if items[:candidate] == items[candidate : candidate * 2]:
                return candidate
        return 0


class ConversationHistory:
    def __init__(
        self,
        transcript_path: Path,
        *,
        clean_transcript_path: Path | None = None,
        max_messages: int = 50,
    ):
        self._buffer: Deque[HistoryEntry] = deque(maxlen=max_messages)
        self._system_message: Optional[str] = None
        self._transcript = TranscriptWriter(transcript_path)
        self._clean_transcript_path = clean_transcript_path

    def add(self, role: str, content: str, *, replace_last: bool = False) -> None:
        if not content:
            return

        if role == "system":
            self.set_system_message(content)
            return

        replaced = False
        if replace_last and self._buffer and self._buffer[-1].role == role:
            entry = self._buffer[-1]
            content = entry.replace_last_chunk(content)
            replaced = True
            if not content:
                self._buffer.pop()
                return
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

    def set_system_message(self, prompt: Optional[str]) -> None:
        if prompt == self._system_message:
            return

        self._system_message = prompt

        record = {
            "ts": time.time(),
            "role": "system",
            "content": prompt if prompt is not None else "",
        }
        if prompt is None:
            record["cleared"] = True

        self._transcript.append(record)
        self._write_clean_transcript()

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


    def reset(self, *, system_prompt: Optional[str] | object = _SYSTEM_UNCHANGED) -> None:
        self._buffer.clear()
        if system_prompt is not _SYSTEM_UNCHANGED:
            self._system_message = cast(Optional[str], system_prompt)
        self._write_clean_transcript()

    def extend(self, entries: Iterable[Dict[str, str]]) -> None:
        for entry in entries:
            if "role" in entry and "content" in entry:
                self.add(entry["role"], entry["content"])

    def export(self) -> List[Dict[str, str]]:
        exported: List[Dict[str, str]] = []
        if self._system_message is not None:
            exported.append({"role": "system", "content": self._system_message})
        exported.extend(entry.as_dict() for entry in self._buffer)
        return exported

    def _write_clean_transcript(self) -> None:
        if not self._clean_transcript_path:
            return

        self._clean_transcript_path.parent.mkdir(parents=True, exist_ok=True)
        entries = self.export()
        lines = [json.dumps(entry) for entry in entries]
        if lines:
            payload = "\n".join(lines) + "\n"
        else:
            payload = ""
        self._clean_transcript_path.write_text(payload, encoding="utf-8")
        
        
    def get_last_assistant(self) -> str:
        """
        Return the most recent assistant message as a string.
        Returns an empty string if there are no assistant messages.
        """
        for entry in reversed(self._buffer):
            if entry.role == "assistant":
                return entry.content
        return ""

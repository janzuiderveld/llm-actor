from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .logging_io import EventLogger


@dataclass
class ActionExtractionState:
    raw_text: str = ""
    actions: List[str] = field(default_factory=list)

    def reset(self) -> None:
        self.raw_text = ""
        self.actions.clear()

    def append_raw(self, text: str) -> None:
        if text:
            self.raw_text += text

    def record_actions(self, actions: List[str]) -> None:
        if actions:
            self.actions.extend(actions)

    def snapshot(self) -> tuple[str, List[str]]:
        return self.raw_text, list(self.actions)


class ReasoningTraceFilter(FrameProcessor):
    """Strips hidden reasoning traces like <think>...</think> before further processing.

    Any content inside known reasoning tags is removed from the stream so it never reaches:
    - Action parsing (<...> directives)
    - TTS
    - Conversation history
    """

    _DEFAULT_TRACE_TAGS = {"think", "analysis", "reasoning", "thought"}

    def __init__(self, *, trace_tags: Optional[set[str]] = None, **kwargs):
        super().__init__(enable_direct_mode=True, **kwargs)
        self._trace_tags = {tag.lower() for tag in (trace_tags or self._DEFAULT_TRACE_TAGS)}
        self._last_raw_text: str = ""
        self._current_tag: Optional[List[str]] = None
        self._in_trace = False

    async def process_frame(self, frame, direction: FrameDirection):  # type: ignore[override]
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, (LLMFullResponseStartFrame, LLMFullResponseEndFrame)):
                self._reset_stream_state()
            elif isinstance(frame, (LLMTextFrame, TextFrame)):
                frame.text = self._strip_reasoning(frame.text or "")
                if getattr(frame, "is_final", False):
                    self._reset_stream_state()

        await self.push_frame(frame, direction)

    def _strip_reasoning(self, raw_text: str) -> str:
        delta = self._extract_stream_delta(raw_text)
        if not delta:
            return ""

        output_chars: List[str] = []

        for char in delta:
            if self._current_tag is not None:
                if char == ">":
                    raw_body = "".join(self._current_tag)
                    self._current_tag = None
                    self._handle_tag(raw_body, output_chars)
                else:
                    self._current_tag.append(char)
                continue

            if char == "<":
                self._current_tag = []
                continue

            if self._in_trace:
                continue

            output_chars.append(char)

        return "".join(output_chars)

    def _extract_stream_delta(self, raw_text: str) -> str:
        if not raw_text:
            self._reset_stream_state()
            return ""

        delta = raw_text
        if self._last_raw_text and raw_text.startswith(self._last_raw_text):
            delta = raw_text[len(self._last_raw_text) :]

        self._last_raw_text = raw_text
        return delta

    def _handle_tag(self, raw_body: str, output_chars: List[str]) -> None:
        normalized = raw_body.strip()
        token = normalized.split(maxsplit=1)[0].lower() if normalized else ""
        is_closing = token.startswith("/")
        name = token[1:] if is_closing else token
        is_self_closing = name.endswith("/")
        if is_self_closing:
            name = name.rstrip("/")

        if name in self._trace_tags:
            if is_closing:
                self._in_trace = False
            elif not is_self_closing:
                self._in_trace = True
            return

        if self._in_trace:
            return

        output_chars.extend(["<", raw_body, ">"])

    def _reset_stream_state(self) -> None:
        self._last_raw_text = ""
        self._current_tag = None
        self._in_trace = False


class ActionExtractorFilter(FrameProcessor):
    """Strips <...> directives from LLM text before TTS and logs them."""

    _ACTION_ADJACENT_PUNCT = {",", ";", ":"}
    _WORDISH_TRAILERS = {'"', "'", ")", "]", "}"}

    def __init__(
        self,
        actions_path: Path,
        event_logger: Optional[EventLogger] = None,
        action_state: Optional[ActionExtractionState] = None,
        **kwargs,
    ):
        super().__init__(enable_direct_mode=True, **kwargs)
        self._actions_path = actions_path
        self._event_logger = event_logger
        self._action_state = action_state
        self._last_raw_text: str = ""
        self._current_action: Optional[List[str]] = None
        self._last_spoken_char: str = ""
        self._last_spoken_nonspace: str = ""
        self._skipped_punct = False

    async def process_frame(self, frame, direction: FrameDirection):  # type: ignore[override]
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, (LLMFullResponseStartFrame, LLMFullResponseEndFrame)):
            self._reset_stream_state()
            if isinstance(frame, LLMFullResponseStartFrame) and self._action_state:
                self._action_state.reset()

        if isinstance(frame, LLMTextFrame) and direction == FrameDirection.DOWNSTREAM:
            raw_text = frame.text or ""
            delta = self._extract_stream_delta(raw_text)
            if self._action_state and delta:
                self._action_state.append_raw(delta)

            sanitized, actions = self._extract_actions_from_delta(delta)
            if actions:
                self._append_actions(actions)
                if self._event_logger:
                    self._event_logger.emit("actions_extracted", {"actions": actions})
                if self._action_state:
                    self._action_state.record_actions([action.strip() for action in actions])
            frame.text = sanitized
            if getattr(frame, "is_final", False):
                self._reset_stream_state()
        await self.push_frame(frame, direction)

    def _append_actions(self, actions: List[str]) -> None:
        self._actions_path.parent.mkdir(parents=True, exist_ok=True)
        with self._actions_path.open("a", encoding="utf-8") as fh:
            for action in actions:
                fh.write(action.strip() + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _extract_stream_delta(self, raw_text: str) -> str:
        if not raw_text:
            self._reset_stream_state()
            return ""

        delta = raw_text
        if self._last_raw_text and raw_text.startswith(self._last_raw_text):
            delta = raw_text[len(self._last_raw_text) :]

        self._last_raw_text = raw_text
        return delta

    def _extract_actions_from_delta(self, delta: str) -> Tuple[str, List[str]]:
        if not delta:
            return "", []

        spoken_chars: List[str] = []
        actions: List[str] = []
        last_spoken_char = self._last_spoken_char
        last_spoken_nonspace = self._last_spoken_nonspace
        skipped_punct = self._skipped_punct
        prior_last_spoken_char = last_spoken_char

        def append_char(ch: str) -> None:
            nonlocal last_spoken_char, last_spoken_nonspace
            spoken_chars.append(ch)
            last_spoken_char = ch
            if not ch.isspace():
                last_spoken_nonspace = ch

        def drop_trailing_space() -> None:
            nonlocal last_spoken_char
            if spoken_chars and spoken_chars[-1].isspace():
                spoken_chars.pop()
                last_spoken_char = spoken_chars[-1] if spoken_chars else prior_last_spoken_char

        for char in delta:
            if self._current_action is not None:
                if char == ">":
                    action = "".join(self._current_action)
                    actions.append(action)
                    self._current_action = None
                else:
                    self._current_action.append(char)
                continue

            if char == "<":
                self._current_action = []
                continue

            if char in self._ACTION_ADJACENT_PUNCT:
                # Avoid speaking stray punctuation that often wraps action tags.
                if not last_spoken_nonspace or not self._is_wordish_char(last_spoken_nonspace):
                    skipped_punct = True
                    continue
                drop_trailing_space()
                append_char(char)
                skipped_punct = False
                continue

            if char.isspace():
                skipped_punct = False
                append_char(char)
                continue

            if skipped_punct and char.isalnum() and last_spoken_char and not last_spoken_char.isspace():
                append_char(" ")
            skipped_punct = False
            append_char(char)

        self._last_spoken_char = last_spoken_char
        self._last_spoken_nonspace = last_spoken_nonspace
        self._skipped_punct = skipped_punct

        return "".join(spoken_chars), actions

    def _reset_stream_state(self) -> None:
        self._last_raw_text = ""
        self._current_action = None
        self._last_spoken_char = ""
        self._last_spoken_nonspace = ""
        self._skipped_punct = False

    @classmethod
    def _is_wordish_char(cls, ch: str) -> bool:
        return ch.isalnum() or ch in cls._WORDISH_TRAILERS


class STTStandaloneIFilter(FrameProcessor):
    """Drops standalone 'I' transcripts produced by the STT service."""

    def __init__(self, event_logger: Optional[EventLogger] = None, **kwargs):
        super().__init__(**kwargs)
        self._event_logger = event_logger

    async def process_frame(self, frame, direction: FrameDirection):  # type: ignore[override]
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            if text and text.lower() == "i":
                if self._event_logger:
                    self._event_logger.emit("stt_drop_standalone_i", {"text": frame.text})
                return
        await self.push_frame(frame, direction)

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

from pipecat.frames.frames import LLMTextFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .logging_io import EventLogger


class ActionExtractorFilter(FrameProcessor):
    """Strips <...> directives from LLM text before TTS and logs them."""

    def __init__(self, actions_path: Path, event_logger: Optional[EventLogger] = None, **kwargs):
        super().__init__(enable_direct_mode=True, **kwargs)
        self._actions_path = actions_path
        self._event_logger = event_logger
        self._last_raw_text: str = ""
        self._current_action: Optional[List[str]] = None

    async def process_frame(self, frame, direction: FrameDirection):  # type: ignore[override]
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMTextFrame) and direction == FrameDirection.DOWNSTREAM:
            sanitized, actions = self._extract_actions(frame.text or "")
            if actions:
                self._append_actions(actions)
                if self._event_logger:
                    self._event_logger.emit("actions_extracted", {"actions": actions})
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

    def _extract_actions(self, raw_text: str) -> Tuple[str, List[str]]:
        if not raw_text:
            self._reset_stream_state()
            return "", []

        prefix_length = 0
        if self._last_raw_text:
            common_prefix = os.path.commonprefix([self._last_raw_text, raw_text])
            prefix_length = len(common_prefix)
            if prefix_length < len(self._last_raw_text):
                if self._current_action is not None:
                    prefix_length = 0
                else:
                    self._reset_stream_state()
                    prefix_length = 0

        delta = raw_text[prefix_length:]
        self._last_raw_text = raw_text

        if not delta:
            return "", []

        spoken_chars: List[str] = []
        actions: List[str] = []

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

            spoken_chars.append(char)

        return "".join(spoken_chars), actions

    def _reset_stream_state(self) -> None:
        self._last_raw_text = ""
        self._current_action = None


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

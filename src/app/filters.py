from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .logging_io import EventLogger

ACTION_PATTERN = re.compile(r"<([^<>]+)>")


class ActionExtractorFilter(FrameProcessor):
    """Strips <...> directives from LLM text before TTS and logs them."""

    def __init__(self, actions_path: Path, event_logger: Optional[EventLogger] = None, **kwargs):
        super().__init__(enable_direct_mode=True, **kwargs)
        self._actions_path = actions_path
        self._event_logger = event_logger

    async def process_frame(self, frame, direction: FrameDirection):  # type: ignore[override]
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMTextFrame) and direction == FrameDirection.DOWNSTREAM:
            original = frame.text
            actions = ACTION_PATTERN.findall(original)
            if actions:
                self._append_actions(actions)
                frame.text = ACTION_PATTERN.sub("", original)
                if self._event_logger:
                    self._event_logger.emit("actions_extracted", {"actions": actions})
        await self.push_frame(frame, direction)

    def _append_actions(self, actions: list[str]) -> None:
        self._actions_path.parent.mkdir(parents=True, exist_ok=True)
        with self._actions_path.open("a", encoding="utf-8") as fh:
            for action in actions:
                fh.write(action.strip() + "\n")
            fh.flush()
            os.fsync(fh.fileno())

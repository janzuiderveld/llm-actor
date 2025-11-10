from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from .logging_io import EventLogger


@dataclass
class MetricsTracker:
    event_logger: EventLogger
    marks: Dict[str, int] = field(default_factory=dict)

    def mark(self, name: str) -> None:
        timestamp = time.monotonic_ns()
        self.marks[name] = timestamp
        self.event_logger.emit("metric_mark", {"name": name, "monotonic_ns": timestamp})

    def compute_turn_metrics(self) -> Dict[str, float]:
        def delta_ms(start: str, end: str) -> Optional[float]:
            if start not in self.marks or end not in self.marks:
                return None
            return (self.marks[end] - self.marks[start]) / 1_000_000.0

        metrics = {
            "audio_to_llm_first_ms": delta_ms("audio_in_last_packet", "llm_first_token"),
            "llm_to_tts_first_ms": delta_ms("llm_first_token", "tts_first_audio"),
            "audio_to_tts_first_ms": delta_ms("audio_in_last_packet", "tts_first_audio"),
            "turn_total_ms": delta_ms("turn_start", "turn_complete"),
        }
        filtered = {k: v for k, v in metrics.items() if v is not None}
        if filtered:
            self.event_logger.emit("latency_metrics", filtered)
        return filtered

    def reset(self) -> None:
        self.marks.clear()

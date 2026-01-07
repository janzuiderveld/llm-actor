from __future__ import annotations


def should_log_tts_audio_event(
    last_event_ts: float,
    now_ts: float,
    audio_bytes: int,
    min_interval_s: float,
) -> bool:
    if audio_bytes <= 0:
        return False
    if now_ts - last_event_ts < min_interval_s:
        return False
    return True

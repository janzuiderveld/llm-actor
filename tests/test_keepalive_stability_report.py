import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from COFFEE_MACHINE.keepalive_stability_report import (
    extract_sessions,
    parse_pid,
    select_session,
    summarize_session_entries,
)


def test_parse_pid_accepts_prefix_or_digits() -> None:
    assert parse_pid("pid=52260") == 52260
    assert parse_pid("PID = 42") == 42
    assert parse_pid("  7 ") == 7
    assert parse_pid("nope") is None


def test_summarize_session_entries_for_pid() -> None:
    lines = [
        "2025-12-31 09:30:00,000 [INFO] keepalive.py started (pid=1).",
        "2025-12-31 09:30:01,000 [INFO] Cycle 1: button press appended.",
        "2025-12-31 09:30:10,000 [WARNING] Cycle 1: no audio response within 10.0 seconds.",
        "2025-12-31 10:10:00,000 [INFO] keepalive.py started (pid=2).",
        "2025-12-31 10:10:01,000 [INFO] Cycle 1: button press appended.",
        "2025-12-31 10:10:05,000 [INFO] Cycle 1: response detected (assistant transcript).",
        "2025-12-31 10:30:00,000 [INFO] Cycle 2: button press appended.",
        "2025-12-31 10:30:05,000 [WARNING] Cycle 2: no audio response within 10.0 seconds.",
    ]
    sessions = extract_sessions(lines)
    session = select_session(sessions, 2)
    assert session is not None
    stats = summarize_session_entries(session.pid, session.entries)

    assert stats.response_successes == 1
    assert stats.response_failures == 1
    assert stats.cycle_starts == 2
    assert stats.cycles_with_results == 2
    assert stats.cycles_with_failures == 1


def test_select_session_prefers_latest_pid() -> None:
    lines = [
        "2025-12-31 11:00:00,000 [INFO] keepalive.py started (pid=1).",
        "2025-12-31 11:00:01,000 [INFO] Cycle 1: button press appended.",
        "2025-12-31 11:00:10,000 [INFO] Cycle 1: response detected (assistant transcript).",
        "2025-12-31 11:30:00,000 [INFO] keepalive.py started (pid=2).",
        "2025-12-31 11:30:01,000 [INFO] Cycle 1: button press appended.",
        "2025-12-31 11:30:10,000 [WARNING] Cycle 1: no audio response within 10.0 seconds.",
        "2025-12-31 12:00:00,000 [INFO] keepalive.py started (pid=1).",
        "2025-12-31 12:00:01,000 [INFO] Cycle 1: button press appended.",
        "2025-12-31 12:00:10,000 [WARNING] Cycle 1: no audio response within 10.0 seconds.",
    ]
    sessions = extract_sessions(lines)
    session = select_session(sessions, 1)
    assert session is not None
    stats = summarize_session_entries(session.pid, session.entries)

    assert stats.response_successes == 0
    assert stats.response_failures == 1
    assert stats.cycle_starts == 1

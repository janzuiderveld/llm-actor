"""Summarize keepalive stability failures from the runtime log."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = REPO_ROOT / "runtime" / "coffee_machine_keepalive_stability.log"

LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} \[[A-Z]+\] (?P<message>.*)$"
)
KEEPALIVE_START_RE = re.compile(r"keepalive\.py started \(pid=(?P<pid>\d+)\)\.")
CYCLE_RE = re.compile(r"^Cycle (?P<cycle>\d+): (?P<event>.*)$")
PID_RE = re.compile(r"pid\s*=\s*(\d+)", re.IGNORECASE)


@dataclass
class Session:
    pid: int
    entries: list[tuple[dt.datetime, str]]

    @property
    def start_time(self) -> dt.datetime:
        return self.entries[0][0]

    @property
    def end_time(self) -> dt.datetime:
        return self.entries[-1][0]


@dataclass
class FailureStats:
    pid: int
    session_start: dt.datetime
    session_end: dt.datetime
    response_successes: int
    response_failures: int
    cycle_starts: int
    cycles_with_results: int
    cycles_with_failures: int

    @property
    def response_attempts(self) -> int:
        return self.response_successes + self.response_failures

    @property
    def response_failure_ratio(self) -> Optional[float]:
        if self.response_attempts == 0:
            return None
        return self.response_failures / self.response_attempts

    @property
    def cycle_failure_ratio(self) -> Optional[float]:
        if self.cycles_with_results == 0:
            return None
        return self.cycles_with_failures / self.cycles_with_results

    @property
    def session_duration(self) -> dt.timedelta:
        return self.session_end - self.session_start


def parse_log_line(line: str) -> Optional[tuple[dt.datetime, str]]:
    match = LOG_LINE_RE.match(line.strip())
    if not match:
        return None
    ts = dt.datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S")
    return ts, match.group("message")


def parse_cycle_event(message: str) -> Optional[tuple[int, str]]:
    match = CYCLE_RE.match(message)
    if not match:
        return None
    return int(match.group("cycle")), match.group("event")


def classify_cycle_event(event: str) -> Optional[str]:
    lowered = event.lower()
    if "button press appended" in lowered:
        return "start"
    if "response detected" in lowered:
        return "success"
    if "no audio response within" in lowered:
        return "failure"
    return None


def parse_pid(raw: str) -> Optional[int]:
    cleaned = raw.strip()
    if not cleaned:
        return None
    match = PID_RE.search(cleaned)
    if match:
        return int(match.group(1))
    if cleaned.isdigit():
        return int(cleaned)
    return None


def extract_sessions(lines: Iterable[str]) -> list[Session]:
    sessions: list[Session] = []
    current_pid: Optional[int] = None
    current_entries: list[tuple[dt.datetime, str]] = []

    for line in lines:
        parsed = parse_log_line(line)
        if not parsed:
            continue
        timestamp, message = parsed
        start_match = KEEPALIVE_START_RE.search(message)
        if start_match:
            pid = int(start_match.group("pid"))
            if current_pid is not None:
                sessions.append(Session(pid=current_pid, entries=current_entries))
            current_pid = pid
            current_entries = [(timestamp, message)]
            continue
        if current_pid is None:
            continue
        current_entries.append((timestamp, message))

    if current_pid is not None:
        sessions.append(Session(pid=current_pid, entries=current_entries))
    return sessions


def select_session(sessions: Iterable[Session], pid: int) -> Optional[Session]:
    matches = [session for session in sessions if session.pid == pid]
    if not matches:
        return None
    return matches[-1]


def summarize_session_entries(pid: int, entries: list[tuple[dt.datetime, str]]) -> FailureStats:
    response_successes = 0
    response_failures = 0
    cycle_starts = 0
    cycles_with_results: set[int] = set()
    cycles_with_failures: set[int] = set()

    for _timestamp, message in entries:
        cycle_info = parse_cycle_event(message)
        if not cycle_info:
            continue
        cycle_number, event_text = cycle_info
        event_type = classify_cycle_event(event_text)
        if not event_type:
            continue
        if event_type == "start":
            cycle_starts += 1
        elif event_type == "success":
            response_successes += 1
            cycles_with_results.add(cycle_number)
        elif event_type == "failure":
            response_failures += 1
            cycles_with_results.add(cycle_number)
            cycles_with_failures.add(cycle_number)

    return FailureStats(
        pid=pid,
        session_start=entries[0][0],
        session_end=entries[-1][0],
        response_successes=response_successes,
        response_failures=response_failures,
        cycle_starts=cycle_starts,
        cycles_with_results=len(cycles_with_results),
        cycles_with_failures=len(cycles_with_failures),
    )


def summarize_log_lines(lines: Iterable[str], pid: int) -> FailureStats:
    sessions = extract_sessions(lines)
    session = select_session(sessions, pid)
    if session is None:
        raise ValueError(f"PID {pid} not found.")
    return summarize_session_entries(pid, session.entries)


def format_ratio(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def format_duration(value: dt.timedelta) -> str:
    total_seconds = int(value.total_seconds())
    if total_seconds < 0:
        total_seconds = 0
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def format_report(stats: FailureStats, *, log_path: Path) -> str:
    window = (
        f"{stats.session_start.strftime('%Y-%m-%d %H:%M:%S')} -> "
        f"{stats.session_end.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    return "\n".join(
        [
            f"Log: {log_path}",
            f"PID: {stats.pid}",
            f"Session: {window} ({format_duration(stats.session_duration)})",
            (
                "Responses: "
                f"{stats.response_successes} successes, "
                f"{stats.response_failures} failures, "
                f"failure ratio {format_ratio(stats.response_failure_ratio)}"
            ),
            (
                "Cycles: "
                f"{stats.cycles_with_results} with results, "
                f"{stats.cycles_with_failures} with failures, "
                f"failure ratio {format_ratio(stats.cycle_failure_ratio)}"
            ),
            f"Cycle starts in session: {stats.cycle_starts}",
        ]
    )


def prompt_pid() -> int:
    while True:
        raw = input("PID to summarize (e.g. 52260 or pid=52260): ")
        pid = parse_pid(raw)
        if pid is None:
            print("Enter a PID like 52260 or pid=52260.")
            continue
        return pid


def format_pid_list(sessions: Iterable[Session]) -> str:
    seen = set()
    ordered = []
    for session in sessions:
        if session.pid in seen:
            continue
        seen.add(session.pid)
        ordered.append(str(session.pid))
    return ", ".join(ordered)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize coffee machine keepalive stability failures.",
    )
    parser.add_argument(
        "--pid",
        type=str,
        help="Keepalive PID to summarize (prompts if omitted).",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to the keepalive stability log.",
    )
    args = parser.parse_args()

    if args.pid is None:
        pid = prompt_pid()
    else:
        pid = parse_pid(args.pid)
        if pid is None:
            print("PID must look like 52260 or pid=52260.", file=sys.stderr)
            return 2

    if not args.log_path.exists():
        print(f"Log file not found: {args.log_path}", file=sys.stderr)
        return 1

    lines = args.log_path.read_text(encoding="utf-8").splitlines()
    sessions = extract_sessions(lines)
    if not sessions:
        print("No keepalive sessions found in the log.", file=sys.stderr)
        return 1
    session = select_session(sessions, pid)
    if session is None:
        print(f"PID {pid} not found in the log.", file=sys.stderr)
        available = format_pid_list(sessions)
        if available:
            print(f"Available PIDs: {available}", file=sys.stderr)
        return 1
    stats = summarize_session_entries(pid, session.entries)
    print(format_report(stats, log_path=args.log_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

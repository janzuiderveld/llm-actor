import json
import os
import sys
import time
import types
from pathlib import Path
import dataclasses
import inspect

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

if "dotenv" not in sys.modules:
    stub = types.ModuleType("dotenv")
    stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = stub

if "slots" not in inspect.signature(dataclasses.dataclass).parameters:
    _orig_dataclass = dataclasses.dataclass

    def _dataclass_wrapper(*args, **kwargs):
        kwargs.pop("slots", None)
        return _orig_dataclass(*args, **kwargs)

    dataclasses.dataclass = _dataclass_wrapper

from app.config import ConfigManager
from app.history import ConversationHistory
from app.logging_io import EventLogger
from app.params_apply import ParamsWatcher
from app.inbox_watch import InboxWatcher
from projects.utils import tail_line


def _wait_for(predicate, timeout_s: float = 1.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_tail_line_resets_on_truncation(tmp_path: Path) -> None:
    path = tmp_path / "stream.txt"
    path.write_text("first\n", encoding="utf-8")

    with path.open("r", encoding="utf-8") as reader:
        reader.seek(0, os.SEEK_END)
        with path.open("a", encoding="utf-8") as writer:
            writer.write("second\n")
        assert tail_line(reader) == "second"

        path.write_text("reset\n", encoding="utf-8")
        assert tail_line(reader) == "reset"


def test_inbox_watcher_recovers_after_truncation(tmp_path: Path) -> None:
    inbox_path = tmp_path / "inbox.txt"
    events_path = tmp_path / "events.ndjson"
    logger = EventLogger(events_path)
    received: list[tuple[str, str]] = []

    def callback(mode: str, payload: str) -> None:
        received.append((mode, payload))

    watcher = InboxWatcher(inbox_path, callback, poll_interval=0.01, event_logger=logger)
    watcher.start()
    try:
        inbox_path.write_text("P: hello\n", encoding="utf-8")
        assert _wait_for(lambda: ("push", "hello") in received)
        received.clear()

        inbox_path.write_text("P: after\n", encoding="utf-8")
        assert _wait_for(lambda: ("push", "after") in received)
    finally:
        watcher.stop()

    events = events_path.read_text(encoding="utf-8").splitlines()
    assert any('"event": "inbox_truncated"' in line for line in events)


def test_params_watcher_recovers_after_truncation(tmp_path: Path) -> None:
    params_path = tmp_path / "params.ndjson"
    config_path = tmp_path / "config.json"
    events_path = tmp_path / "params_events.ndjson"
    manager = ConfigManager(path=config_path)
    history = ConversationHistory(
        tmp_path / "transcript.jsonl",
        clean_transcript_path=tmp_path / "llm_transcript.jsonl",
        max_messages=manager.config.pipeline.max_history_messages,
    )
    applied: list[dict] = []

    def apply_callback(payload: dict) -> None:
        applied.append(payload)

    watcher = ParamsWatcher(
        params_path,
        manager,
        history,
        apply_callback=apply_callback,
        poll_interval=0.01,
        event_logger=EventLogger(events_path),
    )
    watcher.start()
    try:
        params_path.write_text(json.dumps({"op": "llm.set", "model": "gemini-2.5-flash"}) + "\n", encoding="utf-8")
        assert _wait_for(lambda: bool(watcher._pending))
        watcher.drain_pending()
        applied.clear()

        params_path.write_text(json.dumps({"op": "llm.set", "temperature": 0.4}) + "\n", encoding="utf-8")
        assert _wait_for(lambda: bool(watcher._pending))
        watcher.drain_pending()
        assert applied
    finally:
        watcher.stop()

    events = events_path.read_text(encoding="utf-8").splitlines()
    assert any('"event": "params_truncated"' in line for line in events)

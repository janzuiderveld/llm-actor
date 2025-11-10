from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

from .config import RuntimeConfig


class SessionPaths:
    def __init__(self, base_dir: Path, session_name: str):
        self.base_dir = base_dir
        self.session_name = session_name
        self.session_dir = base_dir / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.transcript = self.session_dir / "transcript.jsonl"
        self.llm_transcript = self.session_dir / "transcript.llm.jsonl"
        self.event_log = self.session_dir / "event_log.ndjson"
        self.config_snapshot = self.session_dir / "config.snapshot.json"
        self.input_wav = self.session_dir / "in.wav"
        self.output_wav = self.session_dir / "out.wav"

    def snapshot_config(self, config: RuntimeConfig) -> None:
        self.config_snapshot.write_text(json.dumps(config.as_dict(), indent=2))


def create_session(base_dir: Path, config: RuntimeConfig, session_name: str | None = None) -> SessionPaths:
    if session_name is None:
        session_name = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    session_paths = SessionPaths(base_dir, session_name)
    session_paths.snapshot_config(config)
    return session_paths


DEFAULT_RUNTIME_DIR = Path("runtime/conversations")
DEFAULT_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def new_session(config: RuntimeConfig, session_name: str | None = None) -> SessionPaths:
    return create_session(DEFAULT_RUNTIME_DIR, config, session_name)

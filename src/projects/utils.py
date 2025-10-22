from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

from app.config import CONFIG_PATH, ConfigManager, load_or_initialize_runtime_config

RUNTIME_ROOT = Path("runtime")
_DEFAULT_CONFIG_FILENAME = "project_config.json"


@dataclass(slots=True)
class ProjectConfig:
    """Container for project-level settings loaded from ``project_config.json``."""

    runtime_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProjectConfig":
        runtime_overrides = {
            section: dict(values) for section, values in data.get("runtime", {}).items() if isinstance(values, Mapping)
        }
        metadata = {key: value for key, value in data.items() if key != "runtime"}
        return cls(runtime_overrides=runtime_overrides, metadata=dict(metadata))


def load_project_config(project_dir: Path, filename: str = _DEFAULT_CONFIG_FILENAME) -> ProjectConfig:
    """Load ``project_config.json`` from ``project_dir``."""
    config_path = project_dir / filename
    if not config_path.exists():
        raise FileNotFoundError(f"Project config not found: {config_path}")

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ValueError(f"Project config must be a JSON object: {config_path}")

    return ProjectConfig.from_mapping(raw)


def ensure_runtime_state(runtime_dir: Path = RUNTIME_ROOT) -> None:
    """Make sure the runtime directory and expected files exist."""
    runtime_dir.mkdir(parents=True, exist_ok=True)
    load_or_initialize_runtime_config(CONFIG_PATH)
    for filename in ("actions.txt", "inbox.txt", "params_inbox.ndjson"):
        path = runtime_dir / filename
        if not path.exists():
            path.write_text("", encoding="utf-8")
    conversations_dir = runtime_dir / "conversations"
    conversations_dir.mkdir(parents=True, exist_ok=True)


def reset_runtime_state(runtime_dir: Path = RUNTIME_ROOT, *, clear_conversations: bool = False) -> None:
    """Reset append-only runtime files and optionally clear stored conversations."""
    ensure_runtime_state(runtime_dir)
    for filename in ("actions.txt", "inbox.txt", "params_inbox.ndjson"):
        path = runtime_dir / filename
        path.write_text("", encoding="utf-8")

    if clear_conversations:
        conversations_dir = runtime_dir / "conversations"
        for entry in conversations_dir.iterdir():
            if entry.is_dir():
                for sub in entry.rglob("*"):
                    if sub.is_file():
                        sub.unlink(missing_ok=True)
                entry.rmdir()
            elif entry.is_file():
                entry.unlink(missing_ok=True)


def _filtered_section(section_values: Mapping[str, Any], valid_keys: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only keys present on the runtime config dataclass section."""
    allowed = set(valid_keys.__dict__.keys())
    return {key: value for key, value in section_values.items() if key in allowed}


def apply_runtime_config_overrides(overrides: Mapping[str, Mapping[str, Any]], config_path: Path = CONFIG_PATH) -> None:
    """Apply project-provided overrides to the runtime config file."""
    if not overrides:
        load_or_initialize_runtime_config(config_path)
        return

    manager = ConfigManager(path=config_path)
    current = manager.config
    audio_updates = _filtered_section(overrides.get("audio", {}), current.audio)
    stt_updates = _filtered_section(overrides.get("stt", {}), current.stt)
    llm_updates = _filtered_section(overrides.get("llm", {}), current.llm)
    tts_updates = _filtered_section(overrides.get("tts", {}), current.tts)
    manager.apply_updates(audio=audio_updates, stt=stt_updates, llm=llm_updates, tts=tts_updates)


def spawn_subprocess(
    args: list[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[MutableMapping[str, str]] = None,
    new_session: bool = True,
) -> subprocess.Popen:
    """Spawn a child process suitable for long-running helpers."""
    final_env = os.environ.copy()
    if env:
        final_env.update(env)
    return subprocess.Popen(
        args,
        cwd=str(cwd) if cwd else None,
        env=final_env,
        stdout=None,
        stderr=None,
        stdin=None,
        start_new_session=new_session,
    )


def python_module_args(module: str, *extra: str) -> list[str]:
    """Convenience for launching ``python -m module`` via :func:`spawn_subprocess`."""
    return [sys.executable, "-m", module, *extra]

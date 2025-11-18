from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

from app.config import CONFIG_PATH, ConfigManager, load_or_initialize_runtime_config

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
RUNTIME_ROOT = REPO_ROOT / "runtime"
ACTIONS_FILE = RUNTIME_ROOT / "actions.txt"
INBOX_FILE = RUNTIME_ROOT / "inbox.txt"
PARAMS_FILE = RUNTIME_ROOT / "params_inbox.ndjson"
CONVERSATIONS_DIR = RUNTIME_ROOT / "conversations"

if CONFIG_PATH.is_absolute():
    CONFIG_FILE = CONFIG_PATH
else:
    CONFIG_FILE = (REPO_ROOT / CONFIG_PATH).resolve()

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


def ensure_runtime_state() -> None:
    """Make sure the runtime directory and expected files exist."""
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    load_or_initialize_runtime_config(CONFIG_FILE)
    for path in (ACTIONS_FILE, INBOX_FILE, PARAMS_FILE):
        if not path.exists():
            path.write_text("", encoding="utf-8")
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)


def reset_runtime_state(*, clear_conversations: bool = False) -> None:
    """Reset append-only runtime files and optionally clear stored conversations."""
    ensure_runtime_state()
    for path in (ACTIONS_FILE, INBOX_FILE, PARAMS_FILE):
        path.write_text("", encoding="utf-8")

    if clear_conversations and CONVERSATIONS_DIR.exists():
        for entry in CONVERSATIONS_DIR.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            elif entry.is_file():
                entry.unlink(missing_ok=True)


def _filtered_section(section_values: Mapping[str, Any], valid_keys: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only keys present on the runtime config dataclass section."""
    allowed = set(valid_keys.__dict__.keys())
    return {key: value for key, value in section_values.items() if key in allowed}


def apply_runtime_config_overrides(overrides: Mapping[str, Mapping[str, Any]], config_path: Path = CONFIG_FILE) -> None:
    """Apply project-provided overrides to the runtime config file."""
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    manager = ConfigManager(path=config_path)
    current = manager.config
    audio_updates = _filtered_section(overrides.get("audio", {}), current.audio)
    stt_updates = _filtered_section(overrides.get("stt", {}), current.stt)
    llm_updates = _filtered_section(overrides.get("llm", {}), current.llm)
    tts_updates = _filtered_section(overrides.get("tts", {}), current.tts)
    manager.apply_updates(audio=audio_updates, stt=stt_updates, llm=llm_updates, tts=tts_updates)


def append_json_line(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


def append_inbox_line(text: str) -> None:
    INBOX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with INBOX_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"{text}\n")


def append_action(tag: str) -> None:
    ACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with ACTIONS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"{tag}\n")


def set_system_prompt(prompt: str, *, reset_history: bool = False) -> None:
    append_json_line(PARAMS_FILE, {"op": "llm.system", "text": prompt})
    if reset_history:
        append_json_line(PARAMS_FILE, {"op": "history.reset"})


def tail_line(handle) -> Optional[str]:
    """Read the next line from an open file handle, returning a stripped string."""
    line = handle.readline()
    if not line:
        return None
    return line.strip()


def ensure_runtime_files() -> None:
    for path in (ACTIONS_FILE, INBOX_FILE, PARAMS_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)


def ensure_pythonpath(env: MutableMapping[str, str]) -> None:
    current = env.get("PYTHONPATH")
    parts = [] if not current else current.split(os.pathsep)
    if str(SRC_ROOT) not in parts:
        parts.insert(0, str(SRC_ROOT))
    env["PYTHONPATH"] = os.pathsep.join(parts) if parts else str(SRC_ROOT)


def spawn_subprocess(
    args: list[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[MutableMapping[str, str]] = None,
    new_session: bool = True,
) -> subprocess.Popen:
    """Spawn a child process suitable for long-running helpers."""
    final_env: MutableMapping[str, str] = os.environ.copy()
    if env:
        final_env.update(env)
    ensure_pythonpath(final_env)
    return subprocess.Popen(
        args,
        cwd=str(cwd) if cwd else str(REPO_ROOT),
        env=final_env,
        stdout=None,
        stderr=None,
        stdin=None,
        start_new_session=new_session,
    )


def python_module_args(module: str, *extra: str) -> list[str]:
    """Convenience for launching ``python -m module`` via :func:`spawn_subprocess`."""
    return [sys.executable, "-m", module, *extra]


def launch_module(
    module: str,
    *extra: str,
    pipeline: Optional[str] = "google",
    cwd: Optional[Path] = None,
    new_session: bool = True,
    env: Optional[MutableMapping[str, str]] = None,
) -> subprocess.Popen:
    """Launch a module in a background process."""
    return spawn_subprocess(python_module_args(module, *extra), cwd=cwd, new_session=new_session, env=env)


class TerminalSessionHandle:
    """Lightweight handle for helpers launched via an external terminal window (macOS)."""

    def __init__(self, title: str) -> None:
        self._title = title

    def poll(self) -> None:
        """Pretend the session is always active so terminate() is attempted on shutdown."""
        return None

    def terminate(self) -> None:
        """Close any terminal windows matching the custom title."""
        if sys.platform != "darwin":
            return
        escaped_title = _escape_applescript(self._title)
        script = [
            "osascript",
            "-e",
            'tell application "Terminal"',
            "-e",
            f'set targetWindows to every window whose custom title is "{escaped_title}"',
            "-e",
            "repeat with win in targetWindows",
            "-e",
            "try",
            "-e",
            "close win",
            "-e",
            "end try",
            "-e",
            "end repeat",
            "-e",
            "end tell",
        ]
        try:
            subprocess.run(script, check=False)
        except Exception:
            pass


ProcessHandle = subprocess.Popen | TerminalSessionHandle


def _escape_applescript(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _macos_terminal_command(
    module: str,
    extra: tuple[str, ...],
    cwd: Optional[Path],
    env: Optional[MutableMapping[str, str]],
) -> str:
    base_dir = Path(cwd) if cwd else REPO_ROOT
    final_env: MutableMapping[str, str] = os.environ.copy()
    if env:
        final_env.update(env)
    ensure_pythonpath(final_env)

    exports: dict[str, str] = {}
    pythonpath = final_env.get("PYTHONPATH")
    if pythonpath:
        exports["PYTHONPATH"] = pythonpath
    if env:
        for key, value in env.items():
            if value is not None:
                exports[key] = value

    python_cmd = " ".join(shlex.quote(arg) for arg in python_module_args(module, *extra))
    shell_parts = [f"cd {shlex.quote(str(base_dir))}"]
    for key, value in exports.items():
        shell_parts.append(f"export {key}={shlex.quote(value)}")
    shell_parts.append(python_cmd)
    return " && ".join(shell_parts)


def launch_module_in_terminal(
    module: str,
    *extra: str,
    cwd: Optional[Path] = None,
    env: Optional[MutableMapping[str, str]] = None,
    title: Optional[str] = None,
) -> ProcessHandle:
    """Launch a module in a new terminal window when supported (currently macOS)."""
    if sys.platform != "darwin":
        return launch_module(module, *extra, cwd=cwd, env=env)

    custom_title = title or f"{module.split('.')[-1]}-{uuid.uuid4().hex[:8]}"
    command = _macos_terminal_command(module, extra, cwd, env)
    escaped_command = _escape_applescript(command)
    escaped_title = _escape_applescript(custom_title)
    script = [
        "osascript",
        "-e",
        'tell application "Terminal"',
        "-e",
        "activate",
        "-e",
        f'set newWindow to do script "{escaped_command}"',
        "-e",
        f'set custom title of newWindow to "{escaped_title}"',
        "-e",
        "end tell",
    ]
    try:
        subprocess.run(script, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Fall back to background launch if we cannot open a terminal window.
        return launch_module(module, *extra, cwd=cwd, env=env)
    return TerminalSessionHandle(custom_title)


def terminate_processes(processes: Iterable[Optional[ProcessHandle]]) -> None:
    for proc in processes:
        if not proc:
            continue
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            continue

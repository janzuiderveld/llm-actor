from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from projects.utils import load_project_config

RUNTIME_DIR = Path("runtime")
ACTIONS_PATH = RUNTIME_DIR / "actions.txt"
PARAMS_PATH = RUNTIME_DIR / "params_inbox.ndjson"
INBOX_PATH = RUNTIME_DIR / "inbox.txt"


def _append_params(payload: Dict[str, object]) -> None:
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PARAMS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


def _append_inbox(message: str) -> None:
    INBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INBOX_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"P:{message}\n")


def _play_sound(label: str, sound_path: Optional[Path]) -> None:
    if not sound_path:
        print(f"[DoorWatcher] {label} (no sound configured)")
        return

    if not sound_path.exists():
        print(f"[DoorWatcher] {label} (missing file: {sound_path})")
        return

    command: Optional[list[str]] = None
    if sys.platform == "darwin":
        command = ["afplay", str(sound_path)]
    elif sys.platform.startswith("linux"):
        command = ["paplay", str(sound_path)]
        if not _command_exists(command[0]):
            command = ["aplay", str(sound_path)]
    elif sys.platform.startswith("win"):
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"[console]::OutputEncoding=[System.Text.Encoding]::UTF8;$p=new-object media.soundplayer '{sound_path}';$p.PlaySync()",
        ]

    if command and _command_exists(command[0]):
        try:
            print(f"[DoorWatcher] {label} (playing {sound_path})")
            os.spawnvp(os.P_NOWAIT, command[0], command)
            return
        except Exception as exc:  # pragma: no cover - exec platform dependent
            print(f"[DoorWatcher] Failed to play sound ({exc}); falling back to placeholder.")

    print(f"[DoorWatcher] {label} (pretend sound: {sound_path.name})")


def _command_exists(name: str) -> bool:
    from shutil import which

    return which(name) is not None


def _reset_and_set_prompt(prompt: str) -> None:
    _append_params({"op": "llm.system", "text": prompt})
    _append_params({"op": "history.reset"})


def _announce_status(prefix: str, message: str) -> None:
    _append_inbox(f"{prefix} {message}")


def _handle_unlock(door_settings: Dict[str, str], project_dir: Path) -> None:
    sound_path = _optional_path(door_settings.get("unlock_sound"), project_dir)
    _play_sound("Door unlocking", sound_path)
    _reset_and_set_prompt(door_settings["unlocked_prompt"])
    _announce_status(door_settings.get("status_inbox_prefix", "[system] Door status:"), "Unlocked and welcoming.")


def _handle_lock(door_settings: Dict[str, str], project_dir: Path) -> None:
    sound_path = _optional_path(door_settings.get("lock_sound"), project_dir)
    _play_sound("Door locking", sound_path)
    _reset_and_set_prompt(door_settings["locked_prompt"])
    _announce_status(door_settings.get("status_inbox_prefix", "[system] Door status:"), "Locked and exclusive.")


def _optional_path(value: Optional[str], base_dir: Path) -> Optional[Path]:
    if not value:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def watch_actions() -> None:
    project_dir = Path(__file__).resolve().parent
    config = load_project_config(project_dir)
    door_settings_raw = config.metadata.get("door", {})
    if not isinstance(door_settings_raw, dict):
        raise ValueError("Door metadata missing or invalid in project_config.json")

    required_keys = ("locked_prompt", "unlocked_prompt")
    for key in required_keys:
        if key not in door_settings_raw:
            raise KeyError(f"door.{key} missing in project_config.json")
    door_settings: Dict[str, object] = dict(door_settings_raw)

    ACTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIONS_PATH.touch(exist_ok=True)
    PARAMS_PATH.touch(exist_ok=True)
    INBOX_PATH.touch(exist_ok=True)

    print("[DoorWatcher] Watching for <LOCK> and <UNLOCK> actions.")
    with ACTIONS_PATH.open("r", encoding="utf-8") as handle:
        handle.seek(0, os.SEEK_END)
        while True:
            line = handle.readline()
            if not line:
                time.sleep(0.2)
                continue
            action = line.strip()
            if action == "<UNLOCK>":
                _handle_unlock(door_settings, project_dir)
            elif action == "<LOCK>":
                _handle_lock(door_settings, project_dir)


def main() -> None:
    try:
        watch_actions()
    except KeyboardInterrupt:
        print("[DoorWatcher] Stopped by user.")


if __name__ == "__main__":
    main()

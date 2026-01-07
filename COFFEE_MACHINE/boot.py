"""Entry point script for the COFFEE_MACHINE project.

This project demonstrates file-driven automation:

- Write button press lines into ``runtime/coffee_machine_buttons.txt``.
- The watcher forwards those presses into ``runtime/inbox.txt``.
- When the agent decides to brew it emits ``<Make_Coffee>``, which is logged to
  ``runtime/actions.txt`` and mirrored to ``runtime/coffee_machine_commands.txt``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
# Make sure the shared src/ folder is importable when running this file directly.
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from app.prompt_templates import CLEAN_TIME_TOKEN
from projects.utils import (
    apply_runtime_config_overrides,
    launch_module,
    launch_module_in_terminal,
    reset_runtime_state,
    terminate_processes,
)
from app.devices import load_audio_device_preferences

PROJECT_SLUG = "COFFEE_MACHINE"

# Action emitted by the assistant when a brew should start.
ACTION_MAKE_COFFEE = "Make_Coffee"

RUNTIME_ROOT = REPO_ROOT / "runtime"
# Append-only file written by external systems to simulate hardware button presses.
BUTTON_PRESSES_FILE = RUNTIME_ROOT / "coffee_machine_buttons.txt"
# Append-only file written by the watcher when <Make_Coffee> is observed.
COMMANDS_FILE = RUNTIME_ROOT / "coffee_machine_commands.txt"

PROMPT_ROOT = REPO_ROOT / "COFFEE_MACHINE" / "prompts"
PROMPT_VENUES_DIR = PROMPT_ROOT / "venues"
PROMPT_TEMPLATES_DIR = PROMPT_ROOT / "templates"
DEFAULT_PROMPT_TEMPLATE = PROMPT_ROOT / "template.txt"
DEFAULT_PROMPT_PROFILE = PROMPT_VENUES_DIR / "default.json"
REQUIRED_PROMPT_FIELDS = ("event_name", "location_name", "organizer_name", "drink", "context")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Boot the COFFEE_MACHINE project.")
    parser.add_argument(
        "--venue",
        help="Load COFFEE_MACHINE/prompts/venues/<venue>.json for the system prompt.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Path to a prompt profile JSON file (overrides --venue).",
    )
    parser.add_argument(
        "--template-name",
        help="Select COFFEE_MACHINE/prompts/templates/<name>.txt (overrides profile template fields).",
    )
    parser.add_argument(
        "--template-file",
        help="Path to a prompt template file (overrides --template-name and profile template fields).",
    )
    return parser.parse_args(argv)


def _resolve_prompt_profile_path(*, venue: Optional[str], prompt_file: Optional[str]) -> Path:
    if prompt_file:
        path = Path(prompt_file)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        return path
    if venue:
        return PROMPT_VENUES_DIR / f"{venue}.json"
    return DEFAULT_PROMPT_PROFILE


def _resolve_template_override_path(
    *,
    template_name: Optional[str],
    template_file: Optional[str],
) -> Optional[Path]:
    if template_file:
        path = Path(template_file)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path
    if template_name:
        path = PROMPT_TEMPLATES_DIR / f"{template_name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path
    return None


def _load_prompt_profile(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt profile not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Prompt profile must be a JSON object: {path}")
    missing = [field for field in REQUIRED_PROMPT_FIELDS if field not in data]
    if missing:
        raise ValueError(f"Prompt profile missing fields {missing}: {path}")
    for field in REQUIRED_PROMPT_FIELDS:
        if not isinstance(data.get(field), str):
            raise ValueError(f"Prompt profile field '{field}' must be a string: {path}")
    prompt_append = data.get("prompt_append", "")
    if prompt_append is None:
        prompt_append = ""
    if not isinstance(prompt_append, str):
        raise ValueError("Prompt profile field 'prompt_append' must be a string.")
    data["prompt_append"] = prompt_append
    prompt_template = data.get("prompt_template")
    if prompt_template is not None and not isinstance(prompt_template, str):
        raise ValueError("Prompt profile field 'prompt_template' must be a string.")
    template_name = data.get("template_name")
    if template_name is not None and not isinstance(template_name, str):
        raise ValueError("Prompt profile field 'template_name' must be a string.")
    name = data.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError("Prompt profile field 'name' must be a string.")
    return cast(dict[str, str], data)


def _resolve_prompt_template_path(profile: dict[str, str], profile_path: Path) -> Path:
    template_override = profile.get("prompt_template")
    if template_override:
        template_path = Path(template_override)
        if not template_path.is_absolute():
            template_path = (profile_path.parent / template_override).resolve()
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        return template_path
    template_name = profile.get("template_name")
    if template_name:
        template_path = PROMPT_TEMPLATES_DIR / f"{template_name}.txt"
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        return template_path
    if not DEFAULT_PROMPT_TEMPLATE.exists():
        raise FileNotFoundError(f"Prompt template not found: {DEFAULT_PROMPT_TEMPLATE}")
    return DEFAULT_PROMPT_TEMPLATE


def _current_time_string(now: Optional[datetime.datetime] = None) -> str:
    timestamp = now or datetime.datetime.now()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def _render_system_prompt(
    profile: dict[str, str],
    *,
    template_path: Path,
    clean_time: Optional[str] = None,
) -> str:
    if clean_time is None:
        clean_time = CLEAN_TIME_TOKEN
    template = template_path.read_text(encoding="utf-8")
    prompt = template.format(
        event_name=profile["event_name"],
        location_name=profile["location_name"],
        organizer_name=profile["organizer_name"],
        drink=profile["drink"],
        context=profile["context"],
        make_cmd=ACTION_MAKE_COFFEE,
        clean_time=clean_time,
    )
    prompt_append = profile.get("prompt_append", "").strip()
    if prompt_append:
        prompt = f"{prompt}\n\n{prompt_append}"
    return prompt


def _load_system_prompt(profile_path: Path, *, template_path: Optional[Path] = None) -> str:
    profile = _load_prompt_profile(profile_path)
    if template_path is None:
        template_path = _resolve_prompt_template_path(profile, profile_path)
    return _render_system_prompt(profile, template_path=template_path)


SYSTEM_PROMPT = _load_system_prompt(DEFAULT_PROMPT_PROFILE)

RUNTIME_CONFIG = {
    "audio": {
        "input_device_index": 7,
        "output_device_index": 8,
        "output_sample_rate": 48000,
        "auto_select_devices": False,
    },
    "pipeline": {
        "idle_timeout_secs": 30,
        "cancel_on_idle_timeout": False,
        "pause_stt_on_idle": True,
        "history_on_idle": "reset",
        "max_history_messages": 50,
    },
    "stt": {
        # "model": "macos-hear",
        "model": "deepgram-flux",
        "language": "en-US",
        "eager_eot_threshold": 0.7,
        "eot_threshold": 0.85,
        "eot_timeout_ms": 1500,
        "hear_on_device": True,
        "hear_punctuation": True,
        "hear_input_device_id": None,
        "hear_final_silence_sec": 1.2,
        "hear_restart_on_final": True,
        "hear_keep_mic_open": True,
    },
    "llm": {
        "model": "gemini-3-flash-preview",
        # "model": "gemini-2.5-flash",
        # "model": "openai-gpt-5.2-chat-latest",
        # "model": "ollama-gemma3:4b",
        "system_prompt": SYSTEM_PROMPT,
        "temperature": 1,
        # "temperature": 0.2,
        "max_tokens": 1280,
        "thinking_level": "MINIMAL",
        "request_timeout_s": 30.0,
    },
    "tts": {
        "provider": "macos_say",
        "voice": "aura-2-thalia-en",
        "encoding": "linear16",
        "sample_rate": 24000,
        "cutoff_marker": "[.. cut off by user utterance]",
        "say_voice": None,
        "say_rate_wpm": None,
        "say_audio_device": None,
        "say_interactive": True,
    },
}


def _parse_say_audio_devices(output: str) -> list[tuple[str, str]]:
    devices: list[tuple[str, str]] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if not parts or not parts[0].isdigit():
            continue
        device_id = parts[0]
        name = " ".join(parts[1:]).strip()
        if name:
            devices.append((device_id, name))
    return devices


def _list_say_audio_devices(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[tuple[str, str]]:
    try:
        result = run(
            ["say", "-a", "?"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    output = "\n".join([result.stdout, result.stderr]).strip()
    return _parse_say_audio_devices(output)


def _resolve_say_audio_device(
    preferences_path: Path,
    *,
    list_devices_fn: Callable[[], list[tuple[str, str]]] = _list_say_audio_devices,
) -> Optional[str]:
    preferences = load_audio_device_preferences(preferences_path)
    if not preferences:
        return None
    devices = list_devices_fn()
    if not devices:
        return None
    preferred = preferences.output_device_name
    lowered = preferred.casefold()
    for device_id, name in devices:
        if device_id == preferred or name.casefold().startswith(lowered):
            return device_id
    return None


def _reset_project_io_files() -> None:
    for path in (BUTTON_PRESSES_FILE, COMMANDS_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    profile_path = _resolve_prompt_profile_path(
        venue=args.venue or os.getenv("COFFEE_MACHINE_VENUE"),
        prompt_file=args.prompt_file or os.getenv("COFFEE_MACHINE_PROMPT_FILE"),
    )
    template_path = _resolve_template_override_path(
        template_name=args.template_name or os.getenv("COFFEE_MACHINE_TEMPLATE_NAME"),
        template_file=args.template_file or os.getenv("COFFEE_MACHINE_TEMPLATE_FILE"),
    )
    system_prompt = _load_system_prompt(profile_path, template_path=template_path)

    # Start fresh so stale state from previous runs does not interfere.
    reset_runtime_state()
    _reset_project_io_files()
    runtime_overrides = {
        **RUNTIME_CONFIG,
        "tts": dict(RUNTIME_CONFIG["tts"]),
        "llm": dict(RUNTIME_CONFIG["llm"]),
    }
    runtime_overrides["llm"]["system_prompt"] = system_prompt
    say_device = _resolve_say_audio_device(RUNTIME_ROOT / "audio_device_preferences.json")
    if say_device:
        runtime_overrides["tts"]["say_audio_device"] = say_device
    apply_runtime_config_overrides(runtime_overrides)

    processes = [
        launch_module("app.cli"),
        launch_module_in_terminal(
            "COFFEE_MACHINE.action_watcher",
            title="Coffee Machine Watcher",
        ),
        launch_module_in_terminal(
            "COFFEE_MACHINE.arduino_bridge",
            title="Coffee Machine Arduino",
        ),
    ]

    try:
        processes[0].wait()
    except KeyboardInterrupt:
        pass
    finally:
        terminate_processes(processes)


if __name__ == "__main__":
    main()

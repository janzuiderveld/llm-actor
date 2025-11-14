"""Entry point script that runs the Velvet Room dual-persona dialogue."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from projects.utils import (
    apply_runtime_config_overrides,
    launch_module,
    launch_module_in_terminal,
    reset_runtime_state,
    terminate_processes,
)

from app.config import RuntimeConfig


# Shared reminder for TTS formatting
PROMPT_APPEND = "Only output plain text to be synthesized by a TTS system, no '*' or emojis."

# Persona definitions
DOOR_SYSTEM = (
    """You are the Door that guards the Velvet Room.
    Speak with crisp, exclusive poise.
    Decline entry unless the king arrives (someone saying he is the King).
    Keep replies brief.
    To unlock the door, output <UNLOCK>."""
)
DRUNK_SYSTEM = (
    """You are a Drunk Uncle who desperately wants to enter the Velvet Room.
    Speak in a slightly slurred, persuasive, but endearing tone.
    You believe it is your life mission to discover how to get through that door.
    Keep replies brief and emotional."""
)

DOOR_SYSTEM += "\n\n" + PROMPT_APPEND
DRUNK_SYSTEM += "\n\n" + PROMPT_APPEND


# Runtime configuration
RUNTIME_CONFIG = {
    "audio": {
        "input_device_index": 1,
        "output_device_index": 3,
        "output_sample_rate": 48000,
        "auto_select_devices": False,
    },
    "stt": {
        "model": "deepgram-flux",
        "language": "en-US",
        "eager_eot_threshold": 0.7,
        "eot_threshold": 0.85,
        "eot_timeout_ms": 1500,
    },
    "llm": {
        "model": "gemini-2.5-flash",
        "temperature": 0.2,
        "max_tokens": 1024,
    },
    "tts": {
        "voice": "aura-2-thalia-en",
        "encoding": "linear16",
        "sample_rate": 24000,
    },
}


def main() -> None:
    reset_runtime_state()
    apply_runtime_config_overrides(RUNTIME_CONFIG)

    processes = [
        launch_module("app.cli"),
    ]

    try:
        processes[0].wait()
    except KeyboardInterrupt:
        pass
    finally:
        terminate_processes(processes)


if __name__ == "__main__":
    main()


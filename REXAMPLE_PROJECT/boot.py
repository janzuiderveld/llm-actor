"""Entry point script that spins up the example agents for the Velvet Room door."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
# Make sure the shared src/ folder is importable when running this file directly.
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from projects.utils import (
    apply_runtime_config_overrides,
    launch_module,
    launch_module_in_terminal,
    reset_runtime_state,
    terminate_processes,
)


# Persona scripts that define how the door behaves in each state.
LOCKED_PROMPT = (
    "You guard the Velvet Room. Speak with crisp, exclusive poise. Decline entry unless a the king arrives (someone saying he is the King). Remember, there is only one king. once he is inside, there cant be another in front of the door, keep imposters out. Keep replies brief. To unlock the door, output <SINE 50HZ> ."
)


# Shared reminder appended to both prompts so the voice stays TTS-friendly.
PROMPT_APPEND = "Only output text to be synthesized by a TTS system, no '*' around words or emojis for example"

LOCKED_PROMPT = LOCKED_PROMPT + "\n\n" + PROMPT_APPEND


# Default runtime settings; tweak these to match your hardware and providers.
RUNTIME_CONFIG = {
    "audio": {
        "input_device_index": 1,
        "output_device_index": 2,
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
        "temperature": 0.6,
        "max_tokens": 1024,
        "system_prompt": LOCKED_PROMPT,
    },
    "tts": {
        "voice": "aura-2-thalia-en",
        "encoding": "linear16",
        "sample_rate": 24000,
    },
}


def main() -> None:
    # Start fresh so stale state from previous runs does not interfere.
    reset_runtime_state()
    # Load our example configuration before launching any helper processes.
    apply_runtime_config_overrides(RUNTIME_CONFIG)


    # Start the CLI plus helper scripts; the terminals make their logs easy to follow.
    processes = [
        launch_module("app.cli"),
        launch_module_in_terminal(
            "EXAMPLE_PROJECT.action_watcher",
            title="Action Watcher",
        ),
        launch_module_in_terminal(
            "EXAMPLE_PROJECT.inbox_writer",
            title="Inbox Writer",
        ),
    ]

    try:
        # Keep the helpers alive while the CLI session runs.
        processes[0].wait()
    except KeyboardInterrupt:
        pass
    finally:
        # Always clean up child processes so the system stays tidy.
        terminate_processes(processes)


if __name__ == "__main__":
    main()

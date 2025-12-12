"""Entry point script that spins up the example agents for the Velvet Room door."""

from __future__ import annotations

import sys
from pathlib import Path
import settings_loader

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
# Make sure the shared src/ folder is importable when running this file directly.
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from projects.utils import (
    apply_runtime_config_overrides,
    launch_module,
    reset_runtime_state,
    terminate_processes,
)

# if runtime/dialogue.txt exists, empty it to start fresh.
dialogue_file = Path("runtime/dialogue.txt")
if dialogue_file.exists():
    dialogue_file.write_text("")

# Persona script.
SYSTEM_PROMPT = settings_loader.sys_prompt

# Shared reminder appended to prompt so the voice stays TTS-friendly.
PROMPT_APPEND = "\n\nOnly output text to be synthesized by a TTS system, no '*' around words or emojis for example."

SYSTEM_PROMPT = SYSTEM_PROMPT + PROMPT_APPEND


# Default runtime settings; tweak these to match your hardware and providers.
RUNTIME_CONFIG = {
    "audio": {
        "input_device_index": settings_loader.input_device_index,
        "output_device_index": settings_loader.output_device_index,
        "output_sample_rate": 48000,
        "auto_select_devices": False,
        "aec": settings_loader.aec_setting,
    },
    "stt": {
        "model": settings_loader.stt_model,
        "language": "en-US",
        "eager_eot_threshold": 0.7,
        "eot_threshold": 0.85,
        "eot_timeout_ms": 1500,
    },
    "llm": {
        "model": settings_loader.model,
        "temperature": settings_loader.temperature,
        "max_tokens": 1024,
        "system_prompt": SYSTEM_PROMPT,
        "mode": settings_loader.mode,
        "persona1": {
            "name": settings_loader.p1_name,
            "opening": settings_loader.p1_opening,
            "prompt": settings_loader.p1_prompt + PROMPT_APPEND,
            "voice": settings_loader.p1_voice,
        },
        "persona2": {
            "name": settings_loader.p2_name,
            "opening": settings_loader.p2_opening,
            "prompt": settings_loader.p2_prompt + PROMPT_APPEND,
            "voice": settings_loader.p2_voice,
        },
        "narrator": {
            "name": settings_loader.n_name,
            "opening": "",
            "prompt": settings_loader.n_prompt + PROMPT_APPEND,
            "voice": settings_loader.n_voice,
        }
    },
    "tts": {
        "model": settings_loader.tts_model,
        "voice": settings_loader.sys_voice,
        "encoding": "linear16",
        "sample_rate": 24000,
    },
}
PIPELINE = settings_loader.pipeline  # options: "google", "groq", "ollama"


def main() -> None:
    # Start fresh so stale state from previous runs does not interfere.
    reset_runtime_state()
    # Load our example configuration before launching any helper processes.
    apply_runtime_config_overrides(RUNTIME_CONFIG)

    # Start the CLI.
    processes = [
        launch_module("app.cli", "--pipeline", PIPELINE),
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

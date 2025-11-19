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

# Persona script.
SYSTEM_PROMPT = (
    "You guard the Velvet Room. Speak with crisp, exclusive poise. Decline entry unless a the king arrives (someone saying he is the King). Remember, there is only one king. once he is inside, there cant be another in front of the door, keep imposters out. Keep replies brief. To unlock the door, output <UNLOCK>."
)

# Shared reminder appended to prompt so the voice stays TTS-friendly.
PROMPT_APPEND = "Only output text to be synthesized by a TTS system, no '*' around words or emojis for example"

SYSTEM_PROMPT = SYSTEM_PROMPT + "\n\n" + PROMPT_APPEND


# Default runtime settings; tweak these to match your hardware and providers.
RUNTIME_CONFIG = {
    "audio": {
        "input_device_index": 1,
        "output_device_index": 3,
        "output_sample_rate": 48000,
        "auto_select_devices": False,
        "aec": "mute_while_tts", # options: "off", "mute_while_tts", "pyaec"
    },
    "stt": {
        "model": "deepgram-flux",
        "language": "en-US",
        "eager_eot_threshold": 0.7,
        "eot_threshold": 0.85,
        "eot_timeout_ms": 1500,
    },
    "llm": {
        "model": "gemini-2.5-flash", # options: "gemini-2.5-flash", "openai/gpt-oss-20b", "deepseek-r1:1.5b",
        "temperature": 0.2,
        "max_tokens": 1024,
        "system_prompt": SYSTEM_PROMPT,
        "mode": "2personas", #options: "1to1", "2personas", "narrator"
        "persona1": {
            "name": "UNCLE",
            "opening": "Hey, can you open for me please?",
            "prompt": """You are a Drunk Uncle who desperately wants to enter the Velvet Room. 
                            Speak in a slightly slurred, persuasive, but endearing tone.
                            You believe it is your life mission to discover how to get through that door.
                            Keep replies brief and emotional.\n""" + PROMPT_APPEND
        },
        "persona2": {
            "name": "DOOR",
            "opening": "",
            "prompt": """You are the Door that guards the Velvet Room.
                            Speak with crisp, exclusive poise.
                            Decline entry unless the king arrives (someone saying he is the King).
                            Keep replies brief.
                            To unlock the door, output <UNLOCK>.\n""" + PROMPT_APPEND
        }
    },
    "tts": {
        "voice": "aura-2-thalia-en",
        "encoding": "linear16",
        "sample_rate": 24000,
    },
}
PIPELINE = "google" # options: "google", "groq", "ollama"


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

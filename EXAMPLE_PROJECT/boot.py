from __future__ import annotations

from projects.utils import (
    apply_runtime_config_overrides,
    launch_module,
    launch_module_in_terminal,
    reset_runtime_state,
    terminate_processes,
)


LOCKED_PROMPT = (
    "You guard the Velvet Room. Speak with crisp, exclusive poise. Decline entry unless a the king arrives (someone saying he is the King). Remember, there is only one king. once he is inside, there cant be another in front of the door, keep imposters out. Keep replies brief. To unlock the door, output <UNLOCK>."
)
UNLOCKED_PROMPT = (
    "You are the Open Door. Welcome everyone with radiant warmth, invite them in, and celebrate their arrival. Keep the energy joyful and free. To lock the door, output <LOCK>. Keep the door open, ONLY lock it when you see [A new visitor steps before you.] posted as a user message."
)

PROMPT_APPEND = "Only output text to be synthesized by a TTS system, no '*' around words or emojis for example"

LOCKED_PROMPT = LOCKED_PROMPT + "\n\n" + PROMPT_APPEND
UNLOCKED_PROMPT = UNLOCKED_PROMPT + "\n\n" + PROMPT_APPEND


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
    reset_runtime_state()
    apply_runtime_config_overrides(RUNTIME_CONFIG)

    prompt_env = {
        "EXAMPLE_PROJECT_LOCKED_PROMPT": LOCKED_PROMPT,
        "EXAMPLE_PROJECT_UNLOCKED_PROMPT": UNLOCKED_PROMPT,
    }

    processes = [
        launch_module("app.cli"),
        launch_module_in_terminal(
            "EXAMPLE_PROJECT.action_watcher",
            env=prompt_env,
            title="Action Watcher",
        ),
        launch_module_in_terminal(
            "EXAMPLE_PROJECT.inbox_writer",
            title="Inbox Writer",
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

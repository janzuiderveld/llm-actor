from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from projects.utils import ACTIONS_FILE, INBOX_FILE, append_inbox_line, ensure_runtime_files, set_system_prompt, tail_line
from .boot import LOCKED_PROMPT as DEFAULT_LOCKED_PROMPT, UNLOCKED_PROMPT as DEFAULT_UNLOCKED_PROMPT

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
OPEN_SOUND = ASSETS_DIR / "opening-door-411632.mp3"
CLOSE_SOUND = ASSETS_DIR / "close-door-382723.mp3"


def play_sound(sound_path: Path) -> None:
    """Fire-and-forget playback using whichever CLI player is available."""
    if not sound_path.exists():
        print(f"Sound not found: {sound_path}")
        return

    command = None
    if shutil.which("afplay"):
        command = ["afplay", str(sound_path)]
    elif shutil.which("ffplay"):
        command = ["ffplay", "-nodisp", "-autoexit", str(sound_path)]
    elif shutil.which("play"):
        command = ["play", str(sound_path)]

    if not command:
        print("No supported audio player found; skipping sound playback.")
        return

    try:
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"Failed to play sound {sound_path}: {exc}")


def main() -> None:
    ensure_runtime_files()
    print("Starting action watcher...")
    with ACTIONS_FILE.open("r", encoding="utf-8") as actions, INBOX_FILE.open(
        "r", encoding="utf-8"
    ) as inbox:
        actions.seek(0, os.SEEK_END)
        inbox.seek(0, os.SEEK_END)

        locked = True
        locked_prompt = (
            os.getenv("EXAMPLE_PROJECT_LOCKED_PROMPT")
            or os.getenv("LOCKED_PROMPT")
            or DEFAULT_LOCKED_PROMPT
        )
        unlocked_prompt = (
            os.getenv("EXAMPLE_PROJECT_UNLOCKED_PROMPT")
            or os.getenv("UNLOCKED_PROMPT")
            or DEFAULT_UNLOCKED_PROMPT
        )

        print("starting loop")

        try:
            while True:
                action_line = tail_line(actions)
                if action_line:
                    if action_line == "UNLOCK":
                        print("UNLOCK found")
                        locked = False
                        set_system_prompt(unlocked_prompt)
                        append_inbox_line("A: [The door unlocked, the personality of the door changed drastically, see current system message]")
                        play_sound(OPEN_SOUND)
                        continue
                    if action_line == "LOCK":
                        print("LOCK found")
                        locked = True
                        set_system_prompt(locked_prompt)
                        append_inbox_line("A: [The door locked, the personality of the door changed drastically, see current system message]")
                        play_sound(CLOSE_SOUND)
                        continue

                time.sleep(0.2)
        except KeyboardInterrupt:
            print("Action watcher stopped by KeyboardInterrupt.")
        except Exception as e:
            print(f"Action watcher stopped by Exception: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(10)


if __name__ == "__main__":
    main()

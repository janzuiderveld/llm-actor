"""Watch COFFEE_MACHINE input/output files and bridge them into runtime automation."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from projects.utils import ACTIONS_FILE, acquire_process_lock, append_inbox_line, ensure_runtime_files, tail_line

from .boot import ACTION_MAKE_COFFEE, BUTTON_PRESSES_FILE, COMMANDS_FILE

BUTTON_PRESS_MARKER = "[ButtonPress]"
LOCK_FILE = BUTTON_PRESSES_FILE.parent / "coffee_machine_action_watcher.lock"
LOG_PATH = BUTTON_PRESSES_FILE.parent / "coffee_machine_action_watcher.log"


def is_make_coffee_action(action_line: str) -> bool:
    return action_line.strip().lower() == ACTION_MAKE_COFFEE.lower()


def format_button_press_inbox_line(button_line: str) -> str:
    pressed = button_line.strip()
    return f"P: {BUTTON_PRESS_MARKER} {pressed}"


def append_command_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("coffee_machine_action_watcher")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _rewind_if_truncated(handle, label: str, logger: logging.Logger) -> None:
    try:
        current_pos = handle.tell()
        size = os.fstat(handle.fileno()).st_size
    except OSError:
        return
    if size < current_pos:
        logger.warning("%s file truncated (size=%s < offset=%s); resetting.", label, size, current_pos)
        handle.seek(0, os.SEEK_SET)


def main(*, poll_interval_s: float = 0.1) -> None:
    ensure_runtime_files()
    for path in (BUTTON_PRESSES_FILE, COMMANDS_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    logger = _setup_logging(LOG_PATH)
    lock = acquire_process_lock(LOCK_FILE)
    if not lock:
        logger.warning("COFFEE_MACHINE watcher already running (lock: %s).", LOCK_FILE)
        return

    logger.info("COFFEE_MACHINE watcher started. Button input: %s", BUTTON_PRESSES_FILE)
    logger.info("COFFEE_MACHINE watcher started. Commands output: %s", COMMANDS_FILE)

    try:
        with ACTIONS_FILE.open("r", encoding="utf-8") as actions, BUTTON_PRESSES_FILE.open(
            "r", encoding="utf-8"
        ) as buttons:
            actions.seek(0, os.SEEK_END)
            buttons.seek(0, os.SEEK_END)

            try:
                while True:
                    handled_any = False
                    _rewind_if_truncated(buttons, "buttons", logger)
                    _rewind_if_truncated(actions, "actions", logger)

                    while True:
                        button_line = tail_line(buttons)
                        if button_line is None:
                            break
                        if not button_line:
                            handled_any = True
                            continue
                        handled_any = True
                        append_inbox_line(format_button_press_inbox_line(button_line))
                        logger.info("Forwarded button press to inbox: %s", button_line)

                    while True:
                        action_line = tail_line(actions)
                        if action_line is None:
                            break
                        if not action_line:
                            handled_any = True
                            continue
                        handled_any = True
                        if is_make_coffee_action(action_line):
                            append_command_line(COMMANDS_FILE, f"<{ACTION_MAKE_COFFEE}>")
                            logger.info("Mirrored make-coffee action: %s", action_line)

                    if not handled_any:
                        time.sleep(poll_interval_s)
            except KeyboardInterrupt:
                logger.info("COFFEE_MACHINE watcher stopped by KeyboardInterrupt.")
    finally:
        lock.release()


if __name__ == "__main__":
    main()

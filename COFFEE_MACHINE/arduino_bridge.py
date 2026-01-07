"""Bridge Arduino serial IO with COFFEE_MACHINE runtime files."""

from __future__ import annotations

import logging
import os
import time
from glob import glob
from pathlib import Path
from typing import Callable, Iterable, Optional

from projects.utils import acquire_process_lock, ensure_runtime_files, tail_line

from .boot import ACTION_MAKE_COFFEE, BUTTON_PRESSES_FILE, COMMANDS_FILE

DEFAULT_PORT_GLOB = "/dev/tty*"
DEFAULT_BAUDRATE = 9600
DEFAULT_TIMEOUT_S = 1.0
DEFAULT_POLL_INTERVAL_S = 0.1
DEFAULT_RECONNECT_INTERVAL_S = 2.0
DEFAULT_COMMAND_DELAY_S = 0.5

LOCK_FILE = BUTTON_PRESSES_FILE.parent / "coffee_machine_arduino_bridge.lock"
LOG_PATH = BUTTON_PRESSES_FILE.parent / "coffee_machine_arduino_bridge.log"

USB_PORT_HINTS = ("usbmodem", "usbserial")

MAKE_COFFEE_SEQUENCE = (b"0\n", b"7\n", b"5\n", b"*\n")

BUTTON_CODE_MESSAGES = {
    "2": "[VISITIOR PUSHED COFFEE BUTTON]",
    "3": "[VISITIOR PUSHED ESPRESSO MARTINI BUTTON]",
    "4": "[VISITIOR PUSHED SUGAR BUTTON]",
    "5": "[VISITIOR PUSHED START BUTTON, TELL THEM TO TAKE IT EASY]",
    "6": "[VISITIOR PUSHED CAFE CREME BUTTON]",
    "7": "[VISITIOR PUSHED ESPRESSO BUTTON]",
    "8": "[VISITIOR PUSHED CAPPUCHINO BUTTON]",
    "9": "[VISITIOR PUSHED CACAO BUTTON]",
    "a": "[VISITIOR PUSHED ESPRESSCHOC BUTTON]",
    "b": "[VISITIOR PUSHED SOUP BUTTON. You don't serve soup. Only coffee]",
    "c": "[VISITIOR PUSHED HOT WATER BUTTON]",
}


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("coffee_machine_arduino_bridge")
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


def list_serial_ports(port_glob: str = DEFAULT_PORT_GLOB) -> list[str]:
    return sorted(glob(port_glob))


def find_arduino_port(ports: Iterable[str]) -> Optional[str]:
    for port in ports:
        lowered = port.lower()
        if any(hint in lowered for hint in USB_PORT_HINTS):
            return port
    return None


def open_serial_connection(
    port: str,
    *,
    baudrate: int = DEFAULT_BAUDRATE,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    serial_factory: Optional[Callable[..., object]] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[object]:
    if serial_factory is None:
        try:
            import serial  # type: ignore
        except ModuleNotFoundError:
            if logger:
                logger.warning("pyserial not installed; skipping Arduino connection.")
            return None
        serial_factory = serial.Serial
    try:
        return serial_factory(port, baudrate, timeout=timeout_s)
    except Exception as exc:
        if logger:
            logger.warning("Failed to open serial port %s: %s", port, exc)
        return None


def _serial_in_waiting(serial_conn: object) -> int:
    if hasattr(serial_conn, "in_waiting"):
        return int(getattr(serial_conn, "in_waiting"))
    if hasattr(serial_conn, "inWaiting"):
        return int(serial_conn.inWaiting())
    return 0


def flush_serial_input(serial_conn: object, *, logger: Optional[logging.Logger] = None) -> None:
    while True:
        waiting = _serial_in_waiting(serial_conn)
        if waiting <= 0:
            return
        if logger:
            logger.info("Flushing %s bytes from serial input buffer.", waiting)
        try:
            serial_conn.read(waiting)
        except Exception:
            return


def connect_to_arduino(
    *,
    ports: Optional[Iterable[str]] = None,
    port_glob: str = DEFAULT_PORT_GLOB,
    baudrate: int = DEFAULT_BAUDRATE,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    serial_factory: Optional[Callable[..., object]] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[object]:
    ports = list_serial_ports(port_glob) if ports is None else list(ports)
    if logger:
        logger.info("Serial ports detected: %s", ports)
    port = find_arduino_port(ports)
    if not port:
        if logger:
            logger.info("No Arduino serial port found.")
        return None
    serial_conn = open_serial_connection(
        port,
        baudrate=baudrate,
        timeout_s=timeout_s,
        serial_factory=serial_factory,
        logger=logger,
    )
    if not serial_conn:
        return None
    flush_serial_input(serial_conn, logger=logger)
    if logger:
        logger.info("Connected to Arduino on %s.", port)
    return serial_conn


def normalize_serial_line(raw_line: object) -> Optional[str]:
    if raw_line is None:
        return None
    if isinstance(raw_line, (bytes, bytearray)):
        text = bytes(raw_line).decode("utf-8", errors="ignore")
    else:
        text = str(raw_line)
    text = text.strip()
    if not text:
        return None
    return text[0]


def should_skip_code(code: str, skip_remaining: int) -> tuple[bool, int]:
    if code == "0":
        return True, 3
    if code == "*":
        return True, 0
    if skip_remaining and code in {"5", "7"}:
        return True, skip_remaining - 1
    return False, skip_remaining


def append_button_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def is_make_coffee_command(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("<") and stripped.endswith(">"):
        stripped = stripped[1:-1].strip()
    return stripped.lower() == ACTION_MAKE_COFFEE.lower()


def send_make_coffee(
    serial_conn: object,
    *,
    delay_s: float = DEFAULT_COMMAND_DELAY_S,
    sleep_fn: Callable[[float], None] = time.sleep,
    logger: Optional[logging.Logger] = None,
) -> None:
    for payload in MAKE_COFFEE_SEQUENCE:
        serial_conn.write(payload)
        if logger:
            logger.info("Sent serial payload: %s", payload.decode("ascii", errors="ignore").strip())
        sleep_fn(delay_s)
    if hasattr(serial_conn, "flush"):
        serial_conn.flush()


def _run_serial_loop(
    serial_conn: object,
    *,
    poll_interval_s: float,
    logger: logging.Logger,
) -> None:
    skip_remaining = 0
    with COMMANDS_FILE.open("r", encoding="utf-8") as commands:
        commands.seek(0, os.SEEK_END)
        while True:
            _rewind_if_truncated(commands, "commands", logger)
            handled_any = False

            while True:
                command_line = tail_line(commands)
                if command_line is None:
                    break
                handled_any = True
                if not command_line:
                    continue
                if is_make_coffee_command(command_line):
                    logger.info("Dispatching make-coffee command.")
                    send_make_coffee(serial_conn, logger=logger)
                else:
                    logger.info("Ignoring command line: %s", command_line)

            raw_line = serial_conn.readline()
            if raw_line:
                handled_any = True
                code = normalize_serial_line(raw_line)
                if code:
                    skip, skip_remaining = should_skip_code(code, skip_remaining)
                    if skip:
                        continue
                    message = BUTTON_CODE_MESSAGES.get(code)
                    if message:
                        append_button_line(BUTTON_PRESSES_FILE, message)
                        logger.info("Button press detected: %s", message)
            if not handled_any:
                time.sleep(poll_interval_s)


def main(
    *,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    reconnect_interval_s: float = DEFAULT_RECONNECT_INTERVAL_S,
) -> None:
    ensure_runtime_files()
    for path in (BUTTON_PRESSES_FILE, COMMANDS_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    logger = _setup_logging(LOG_PATH)
    lock = acquire_process_lock(LOCK_FILE)
    if not lock:
        logger.warning("COFFEE_MACHINE Arduino bridge already running (lock: %s).", LOCK_FILE)
        return

    logger.info("COFFEE_MACHINE Arduino bridge started.")
    logger.info("Button output: %s", BUTTON_PRESSES_FILE)
    logger.info("Command input: %s", COMMANDS_FILE)

    try:
        while True:
            serial_conn = connect_to_arduino(logger=logger)
            if not serial_conn:
                logger.info("Retrying Arduino connection in %.1fs.", reconnect_interval_s)
                time.sleep(reconnect_interval_s)
                continue
            try:
                _run_serial_loop(serial_conn, poll_interval_s=poll_interval_s, logger=logger)
            except KeyboardInterrupt:
                logger.info("Arduino bridge stopped by KeyboardInterrupt.")
                break
            except Exception as exc:
                logger.warning("Arduino bridge error: %s", exc)
            finally:
                if hasattr(serial_conn, "close"):
                    try:
                        serial_conn.close()
                    except Exception:
                        pass
            logger.info("Arduino disconnected; reconnecting in %.1fs.", reconnect_interval_s)
            time.sleep(reconnect_interval_s)
    finally:
        lock.release()


if __name__ == "__main__":
    main()

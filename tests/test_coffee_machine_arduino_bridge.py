import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from COFFEE_MACHINE import arduino_bridge


class FakeSerial:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, payload: bytes) -> None:
        self.writes.append(payload)

    def flush(self) -> None:
        return None

    def read(self, _size: int) -> bytes:
        return b""

    @property
    def in_waiting(self) -> int:
        return 0


def test_find_arduino_port_prefers_usb_device() -> None:
    ports = [
        "/dev/tty.Bluetooth-Incoming-Port",
        "/dev/tty.usbmodem1234",
        "/dev/tty.usbserial999",
    ]
    assert arduino_bridge.find_arduino_port(ports) == "/dev/tty.usbmodem1234"


@pytest.mark.parametrize(
    ("raw_line", "expected"),
    [
        (b"2\r\n", "2"),
        ("7\n", "7"),
        (b"", None),
        ("   ", None),
    ],
)
def test_normalize_serial_line(raw_line: object, expected: str | None) -> None:
    assert arduino_bridge.normalize_serial_line(raw_line) == expected


def test_should_skip_code_tracks_sequence() -> None:
    skip, remaining = arduino_bridge.should_skip_code("0", 0)
    assert skip is True
    assert remaining == 3

    skip, remaining = arduino_bridge.should_skip_code("7", remaining)
    assert skip is True
    assert remaining == 2

    skip, remaining = arduino_bridge.should_skip_code("5", remaining)
    assert skip is True
    assert remaining == 1

    skip, remaining = arduino_bridge.should_skip_code("*", remaining)
    assert skip is True
    assert remaining == 0

    skip, remaining = arduino_bridge.should_skip_code("2", remaining)
    assert skip is False
    assert remaining == 0


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("2", "[VISITOR PUSHED COFFEE BUTTON]"),
        ("7", "[VISITOR PUSHED ESPRESSO BUTTON]"),
        ("c", "[VISITOR PUSHED HOT WATER BUTTON]"),
    ],
)
def test_button_code_messages(code: str, expected: str) -> None:
    assert arduino_bridge.BUTTON_CODE_MESSAGES[code] == expected


@pytest.mark.parametrize(
    "line",
    [
        "<Make_Coffee>",
        "Make_Coffee",
        "  <make_coffee> ",
    ],
)
def test_is_make_coffee_command(line: str) -> None:
    assert arduino_bridge.is_make_coffee_command(line)


def test_is_make_coffee_command_rejects_other() -> None:
    assert not arduino_bridge.is_make_coffee_command("<Other_Action>")


def test_send_make_coffee_writes_sequence() -> None:
    serial_conn = FakeSerial()
    arduino_bridge.send_make_coffee(serial_conn, delay_s=0, sleep_fn=lambda _: None)
    assert serial_conn.writes == list(arduino_bridge.MAKE_COFFEE_SEQUENCE)

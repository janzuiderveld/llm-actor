import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from app.config import ConfigManager
from app.devices import (
    AudioDevicePreferences,
    DeviceInfo,
    apply_audio_device_preferences,
    ensure_audio_device_preferences,
)


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "audio": {
                    "input_device_index": None,
                    "output_device_index": None,
                    "output_sample_rate": 48000,
                    "auto_select_devices": False,
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_apply_audio_device_preferences_resolves_by_name(tmp_path):
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    manager = ConfigManager(path=config_path)

    devices = [
        DeviceInfo(index=10, name="Test Mic", max_input_channels=1, max_output_channels=0),
        DeviceInfo(index=20, name="Test Speaker", max_input_channels=0, max_output_channels=2),
    ]
    prefs = AudioDevicePreferences(input_device_name="Test Mic", output_device_name="Test Speaker")

    apply_audio_device_preferences(manager, prefs, devices=devices)

    assert manager.config.audio.input_device_index == 10
    assert manager.config.audio.output_device_index == 20
    assert manager.config.audio.auto_select_devices is False


def test_ensure_audio_device_preferences_prompts_and_persists(tmp_path):
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    manager = ConfigManager(path=config_path)

    devices = [
        DeviceInfo(index=3, name="Chosen Mic", max_input_channels=1, max_output_channels=0),
        DeviceInfo(index=4, name="Chosen Speaker", max_input_channels=0, max_output_channels=2),
    ]
    preferences_path = tmp_path / "audio_device_preferences.json"
    inputs = iter(["3", "4"])

    def fake_input(_prompt: str) -> str:
        return next(inputs)

    captured: list[str] = []

    def fake_print(message: str) -> None:
        captured.append(message)

    prefs = ensure_audio_device_preferences(
        manager,
        preferences_path=preferences_path,
        devices=devices,
        interactive=True,
        input_fn=fake_input,
        print_fn=fake_print,
    )

    assert prefs.input_device_name == "Chosen Mic"
    assert prefs.output_device_name == "Chosen Speaker"
    assert preferences_path.exists()

    persisted = json.loads(preferences_path.read_text(encoding="utf-8"))
    assert persisted["input_device_name"] == "Chosen Mic"
    assert persisted["output_device_name"] == "Chosen Speaker"
    assert manager.config.audio.input_device_index == 3
    assert manager.config.audio.output_device_index == 4


def test_ensure_audio_device_preferences_reuses_existing_file(tmp_path):
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    manager = ConfigManager(path=config_path)

    preferences_path = tmp_path / "audio_device_preferences.json"
    preferences_path.write_text(
        json.dumps(
            {"input_device_name": "Saved Mic", "output_device_name": "Saved Speaker"},
            indent=2,
        ),
        encoding="utf-8",
    )

    devices = [
        DeviceInfo(index=99, name="Saved Mic", max_input_channels=1, max_output_channels=0),
        DeviceInfo(index=100, name="Saved Speaker", max_input_channels=0, max_output_channels=2),
    ]

    prefs = ensure_audio_device_preferences(
        manager,
        preferences_path=preferences_path,
        devices=devices,
        interactive=False,
    )

    assert prefs.input_device_name == "Saved Mic"
    assert prefs.output_device_name == "Saved Speaker"
    assert manager.config.audio.input_device_index == 99
    assert manager.config.audio.output_device_index == 100


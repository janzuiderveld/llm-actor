from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from .config import ConfigManager, detect_default_audio_device_indices

PREFERENCES_FILENAME = "audio_device_preferences.json"


@dataclass
class DeviceInfo:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "maxInputChannels": self.max_input_channels,
            "maxOutputChannels": self.max_output_channels,
        }


def _from_sounddevice() -> List[DeviceInfo]:
    try:
        import sounddevice as sd  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        return []

    devices = []
    for idx, data in enumerate(sd.query_devices()):
        devices.append(
            DeviceInfo(
                index=idx,
                name=data.get("name", f"Device {idx}"),
                max_input_channels=int(data.get("max_input_channels", 0)),
                max_output_channels=int(data.get("max_output_channels", 0)),
            )
        )
    return devices


def _from_pipecat() -> List[DeviceInfo]:
    try:  # pragma: no cover - depends on optional install
        from pipecat.transports.local.devices import list_audio_devices  # type: ignore
    except Exception:  # noqa: BLE001
        return []

    devices: List[DeviceInfo] = []
    for device in list_audio_devices():
        devices.append(
            DeviceInfo(
                index=device.index,
                name=getattr(device, "name", f"Device {device.index}"),
                max_input_channels=getattr(device, "max_input_channels", 0),
                max_output_channels=getattr(device, "max_output_channels", 0),
            )
        )
    return devices


def list_devices() -> List[DeviceInfo]:
    devices = _from_pipecat()
    if not devices:
        devices = _from_sounddevice()
    return devices


@dataclass(frozen=True, slots=True)
class AudioDevicePreferences:
    input_device_name: str
    output_device_name: str

    def as_dict(self) -> dict:
        return {
            "input_device_name": self.input_device_name,
            "output_device_name": self.output_device_name,
        }


def _default_preferences_path(config_manager: ConfigManager) -> Path:
    return config_manager.path.parent / PREFERENCES_FILENAME


def load_audio_device_preferences(path: Path) -> Optional[AudioDevicePreferences]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    input_name = raw.get("input_device_name")
    output_name = raw.get("output_device_name")
    if not isinstance(input_name, str) or not input_name.strip():
        return None
    if not isinstance(output_name, str) or not output_name.strip():
        return None
    return AudioDevicePreferences(input_device_name=input_name, output_device_name=output_name)


def save_audio_device_preferences(preferences: AudioDevicePreferences, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(preferences.as_dict(), indent=2), encoding="utf-8")


def _resolve_device_index_by_name(
    devices: List[DeviceInfo],
    device_name: str,
    *,
    require_input: bool,
) -> Optional[int]:
    def is_valid(device: DeviceInfo) -> bool:
        if require_input:
            return device.max_input_channels > 0
        return device.max_output_channels > 0

    exact = [device for device in devices if device.name == device_name and is_valid(device)]
    if exact:
        return exact[0].index

    lowered = device_name.casefold()
    casefold_matches = [device for device in devices if device.name.casefold() == lowered and is_valid(device)]
    if casefold_matches:
        return casefold_matches[0].index

    return None


def apply_audio_device_preferences(
    config_manager: ConfigManager,
    preferences: AudioDevicePreferences,
    *,
    devices: Optional[List[DeviceInfo]] = None,
) -> None:
    resolved_devices = list_devices() if devices is None else devices
    input_index = _resolve_device_index_by_name(
        resolved_devices,
        preferences.input_device_name,
        require_input=True,
    )
    output_index = _resolve_device_index_by_name(
        resolved_devices,
        preferences.output_device_name,
        require_input=False,
    )
    if input_index is None or output_index is None:
        missing = []
        if input_index is None:
            missing.append(f"input '{preferences.input_device_name}'")
        if output_index is None:
            missing.append(f"output '{preferences.output_device_name}'")
        raise RuntimeError("Could not match saved audio device preferences: " + ", ".join(missing))

    config_manager.apply_updates(
        audio={
            "input_device_index": input_index,
            "output_device_index": output_index,
            "auto_select_devices": False,
        }
    )


def prompt_for_audio_device_preferences(
    *,
    devices: Optional[List[DeviceInfo]] = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> AudioDevicePreferences:
    resolved_devices = list_devices() if devices is None else devices
    if not resolved_devices:
        raise RuntimeError("No audio devices found. Ensure PortAudio is installed and devices are available.")

    print_fn("Available audio devices:")
    for device in resolved_devices:
        print_fn(
            f"  {device.index}: {device.name} (in:{device.max_input_channels} out:{device.max_output_channels})"
        )
    print_fn("")

    input_devices = {device.index: device for device in resolved_devices if device.max_input_channels > 0}
    output_devices = {device.index: device for device in resolved_devices if device.max_output_channels > 0}
    if not input_devices:
        raise RuntimeError("No input-capable audio devices found.")
    if not output_devices:
        raise RuntimeError("No output-capable audio devices found.")

    def prompt_index(prompt: str, choices: dict[int, DeviceInfo]) -> DeviceInfo:
        while True:
            raw = input_fn(prompt).strip()
            try:
                idx = int(raw)
            except ValueError:
                print_fn("Please enter a valid integer device index.")
                continue
            selected = choices.get(idx)
            if selected is None:
                print_fn("Please choose one of the listed device indices.")
                continue
            return selected

    input_device = prompt_index("Select input device index (microphone): ", input_devices)
    output_device = prompt_index("Select output device index (speaker): ", output_devices)

    return AudioDevicePreferences(
        input_device_name=input_device.name,
        output_device_name=output_device.name,
    )


def ensure_audio_device_preferences(
    config_manager: ConfigManager,
    *,
    preferences_path: Optional[Path] = None,
    devices: Optional[List[DeviceInfo]] = None,
    interactive: Optional[bool] = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> Optional[AudioDevicePreferences]:
    """Ensure audio device preferences exist and are applied to runtime config.

    - If preferences file exists, resolve device indices by name and apply them.
    - If missing and interactive, prompt the user to select devices and persist by name.
    - If missing and not interactive, leave selection to other mechanisms (auto-select or manual indices).
    """
    path = _default_preferences_path(config_manager) if preferences_path is None else preferences_path
    if interactive is None:
        try:
            interactive = sys.stdin.isatty()
        except Exception:  # pragma: no cover - defensive
            interactive = False

    preferences = load_audio_device_preferences(path)
    resolved_devices = list_devices() if devices is None else devices

    if preferences is not None:
        try:
            apply_audio_device_preferences(config_manager, preferences, devices=resolved_devices)
            return preferences
        except RuntimeError:
            if not interactive:
                raise
            print_fn(f"Saved audio device preferences at {path} could not be resolved. Re-selecting devices.")

    if not interactive:
        return None

    preferences = prompt_for_audio_device_preferences(
        devices=resolved_devices,
        input_fn=input_fn,
        print_fn=print_fn,
    )
    save_audio_device_preferences(preferences, path)
    apply_audio_device_preferences(config_manager, preferences, devices=resolved_devices)
    return preferences


def ensure_devices_selected(
    config_manager: ConfigManager,
    *,
    require_input: bool = True,
    require_output: bool = True,
) -> None:
    cfg = config_manager.config
    has_input = cfg.audio.input_device_index is not None
    has_output = cfg.audio.output_device_index is not None
    if (not require_input or has_input) and (not require_output or has_output):
        return

    input_index, output_index = detect_default_audio_device_indices()

    updates: dict[str, int] = {}
    if require_input and cfg.audio.input_device_index is None and input_index is not None:
        updates["input_device_index"] = input_index
    if require_output and cfg.audio.output_device_index is None and output_index is not None:
        updates["output_device_index"] = output_index

    if updates:
        config_manager.apply_updates(audio=updates)

    if (
        (require_input and config_manager.config.audio.input_device_index is None)
        or (require_output and config_manager.config.audio.output_device_index is None)
    ):
        try:
            import sounddevice  # type: ignore  # noqa: F401
        except ImportError:
            hint = (
                "Install PortAudio and the `sounddevice` Python package, then rerun "
                "`pip install -e .` inside your virtualenv."
            )
        else:
            hint = "Verify PortAudio can enumerate devices via `python -m sounddevice`."
        raise RuntimeError(
            "Unable to detect system default audio devices automatically. "
            "Set the missing `audio.*_device_index` values manually in runtime/config.json. "
            f"{hint}"
        )

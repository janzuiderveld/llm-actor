from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Optional

from .config import ConfigManager


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


def _prompt_user_choice(devices: Iterable[DeviceInfo], kind: str) -> Optional[int]:
    print(f"Available {kind} devices:")
    for device in devices:
        print(json.dumps(device.as_dict()))
    while True:
        raw = input(f"Select {kind} device index (or leave blank to skip): ")
        if not raw:
            return None
        try:
            idx = int(raw)
        except ValueError:
            print("Please enter a valid integer index.")
            continue
        return idx


def ensure_devices_selected(config_manager: ConfigManager) -> None:
    cfg = config_manager.config
    if cfg.audio.input_device_index is not None and cfg.audio.output_device_index is not None:
        return

    devices = list_devices()
    if not devices:
        raise RuntimeError(
            "No audio devices detected. Install PortAudio or run on a system with audio hardware."
        )

    input_index = cfg.audio.input_device_index
    output_index = cfg.audio.output_device_index

    if input_index is None:
        input_index = _prompt_user_choice(devices, "input")
    if output_index is None:
        output_index = _prompt_user_choice(devices, "output")

    if input_index is None or output_index is None:
        raise RuntimeError("Input/output device selection is required for the voice pipeline.")

    config_manager.set_audio_devices(input_index, output_index)

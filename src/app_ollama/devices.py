from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .config import ConfigManager, detect_default_audio_device_indices


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


def ensure_devices_selected(config_manager: ConfigManager) -> None:
    cfg = config_manager.config
    if cfg.audio.input_device_index is not None and cfg.audio.output_device_index is not None:
        return

    input_index, output_index = detect_default_audio_device_indices()

    updates: dict[str, int] = {}
    if cfg.audio.input_device_index is None and input_index is not None:
        updates["input_device_index"] = input_index
    if cfg.audio.output_device_index is None and output_index is not None:
        updates["output_device_index"] = output_index

    if updates:
        config_manager.apply_updates(audio=updates)

    if (
        config_manager.config.audio.input_device_index is None
        or config_manager.config.audio.output_device_index is None
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
            "Set `audio.input_device_index` and `audio.output_device_index` manually in runtime/config.json. "
            f"{hint}"
        )

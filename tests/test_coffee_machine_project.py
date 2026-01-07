import json
import sys
from dataclasses import fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from COFFEE_MACHINE import action_watcher, boot
from app.config import AudioConfig, LLMConfig, PipelineConfig, STTConfig, TTSConfig


def test_coffee_machine_action_tag_matches_spec() -> None:
    assert boot.ACTION_MAKE_COFFEE == "Make_Coffee"


def test_coffee_machine_prompt_references_action_tag() -> None:
    assert f"<{boot.ACTION_MAKE_COFFEE}>" in boot.SYSTEM_PROMPT


def test_coffee_machine_llm_is_gemini_flash() -> None:
    assert boot.RUNTIME_CONFIG["llm"]["model"] == "gemini-3-flash-preview"


def test_coffee_machine_tts_is_macos_say() -> None:
    assert boot.RUNTIME_CONFIG["tts"]["provider"] == "macos_say"


def test_coffee_machine_runtime_config_fully_specified() -> None:
    assert set(boot.RUNTIME_CONFIG.keys()) == {"audio", "pipeline", "stt", "llm", "tts"}

    assert set(boot.RUNTIME_CONFIG["audio"].keys()) == {field.name for field in fields(AudioConfig)}
    assert set(boot.RUNTIME_CONFIG["pipeline"].keys()) == {field.name for field in fields(PipelineConfig)}
    assert set(boot.RUNTIME_CONFIG["stt"].keys()) == {field.name for field in fields(STTConfig)}
    assert set(boot.RUNTIME_CONFIG["llm"].keys()) == {field.name for field in fields(LLMConfig)}
    assert set(boot.RUNTIME_CONFIG["tts"].keys()) == {field.name for field in fields(TTSConfig)}


def test_coffee_machine_resets_history_on_idle() -> None:
    assert boot.RUNTIME_CONFIG["pipeline"]["history_on_idle"] == "reset"


def test_coffee_machine_thinking_level_is_minimal() -> None:
    assert boot.RUNTIME_CONFIG["llm"]["thinking_level"] == "MINIMAL"


def test_coffee_machine_io_files_live_in_runtime() -> None:
    assert str(boot.BUTTON_PRESSES_FILE).endswith("runtime/coffee_machine_buttons.txt")
    assert str(boot.COMMANDS_FILE).endswith("runtime/coffee_machine_commands.txt")


def test_coffee_machine_watcher_imports_boot_constants() -> None:
    assert action_watcher.ACTION_MAKE_COFFEE == boot.ACTION_MAKE_COFFEE
    assert action_watcher.BUTTON_PRESSES_FILE == boot.BUTTON_PRESSES_FILE
    assert action_watcher.COMMANDS_FILE == boot.COMMANDS_FILE


def test_coffee_machine_watcher_formats_button_presses() -> None:
    assert (
        action_watcher.format_button_press_inbox_line("BREW\n")
        == "P: [ButtonPress] BREW"
    )


def test_coffee_machine_watcher_detects_make_coffee_action() -> None:
    assert action_watcher.is_make_coffee_action("Make_Coffee")
    assert action_watcher.is_make_coffee_action("make_coffee")
    assert not action_watcher.is_make_coffee_action("OTHER_ACTION")


def test_coffee_machine_watcher_appends_commands(tmp_path) -> None:
    commands_path = tmp_path / "commands.txt"
    action_watcher.append_command_line(commands_path, "<Make_Coffee>")
    assert commands_path.read_text(encoding="utf-8") == "<Make_Coffee>\n"


def test_boot_resolves_say_audio_device_from_preferences(tmp_path) -> None:
    prefs_path = tmp_path / "audio_device_preferences.json"
    prefs_path.write_text(
        json.dumps(
            {
                "input_device_name": "krisp microphone",
                "output_device_name": "krisp speaker",
            }
        ),
        encoding="utf-8",
    )

    device_id = boot._resolve_say_audio_device(
        prefs_path,
        list_devices_fn=lambda: [
            ("173", "Krisp Speaker"),
            ("153", "MacBook Pro Speakers"),
        ],
    )

    assert device_id == "173"


def test_boot_resolves_say_audio_device_returns_none_on_miss(tmp_path) -> None:
    prefs_path = tmp_path / "audio_device_preferences.json"
    prefs_path.write_text(
        json.dumps(
            {
                "input_device_name": "krisp microphone",
                "output_device_name": "krisp speaker",
            }
        ),
        encoding="utf-8",
    )

    device_id = boot._resolve_say_audio_device(
        prefs_path,
        list_devices_fn=lambda: [("153", "MacBook Pro Speakers")],
    )

    assert device_id is None

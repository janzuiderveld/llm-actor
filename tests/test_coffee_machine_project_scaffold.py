import importlib.util
from pathlib import Path


def _load_boot_module():
    repo_root = Path(__file__).resolve().parents[1]
    boot_path = repo_root / "COFFEE_MACHINE" / "boot.py"
    spec = importlib.util.spec_from_file_location("coffee_machine_boot", boot_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


boot = _load_boot_module()


def test_coffee_machine_boot_module_exposes_required_constants() -> None:
    assert boot.PROJECT_SLUG == "COFFEE_MACHINE"
    assert boot.ACTION_MAKE_COFFEE == "Make_Coffee"
    assert "<Make_Coffee>" in boot.SYSTEM_PROMPT
    assert str(boot.BUTTON_PRESSES_FILE).endswith("runtime/coffee_machine_buttons.txt")
    assert str(boot.COMMANDS_FILE).endswith("runtime/coffee_machine_commands.txt")


def test_coffee_machine_boot_defines_main_entrypoint() -> None:
    assert callable(getattr(boot, "main", None))

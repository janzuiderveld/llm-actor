import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from COFFEE_MACHINE import boot


def test_prompt_profile_renders_template_and_append(tmp_path: Path) -> None:
    template_path = tmp_path / "template.txt"
    template_path.write_text(
        "Event {event_name} at {location_name}. "
        "Organizer {organizer_name}. "
        "Drink {drink}. "
        "Command <{make_cmd}>. "
        "Context {context}. "
        "Time {clean_time}.",
        encoding="utf-8",
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "event_name": "Test Expo",
                "location_name": "Test Hall",
                "organizer_name": "Test Org",
                "drink": "espresso",
                "context": "Test context.",
                "prompt_append": "Extra instructions.",
            }
        ),
        encoding="utf-8",
    )

    profile = boot._load_prompt_profile(profile_path)
    prompt = boot._render_system_prompt(
        profile,
        template_path=template_path,
        clean_time="2024-01-01 12:00:00",
    )

    assert "Test Expo" in prompt
    assert "Test Hall" in prompt
    assert "Test Org" in prompt
    assert "espresso" in prompt
    assert "<Make_Coffee>" in prompt
    assert "Test context." in prompt
    assert "2024-01-01 12:00:00" in prompt
    assert "Extra instructions." in prompt


def test_prompt_template_override_resolves_relative_path(tmp_path: Path) -> None:
    template_path = tmp_path / "alt_template.txt"
    template_path.write_text("Hello {event_name}.", encoding="utf-8")

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "event_name": "Test Expo",
                "location_name": "Test Hall",
                "organizer_name": "Test Org",
                "drink": "espresso",
                "context": "Test context.",
                "prompt_template": "alt_template.txt",
            }
        ),
        encoding="utf-8",
    )

    profile = boot._load_prompt_profile(profile_path)
    resolved = boot._resolve_prompt_template_path(profile, profile_path)

    assert resolved == template_path


def test_prompt_template_name_resolves_from_templates_dir(tmp_path: Path, monkeypatch) -> None:
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    template_path = templates_dir / "v2.txt"
    template_path.write_text("Hello {event_name}.", encoding="utf-8")

    monkeypatch.setattr(boot, "PROMPT_TEMPLATES_DIR", templates_dir)

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "event_name": "Test Expo",
                "location_name": "Test Hall",
                "organizer_name": "Test Org",
                "drink": "espresso",
                "context": "Test context.",
                "template_name": "v2",
            }
        ),
        encoding="utf-8",
    )

    profile = boot._load_prompt_profile(profile_path)
    resolved = boot._resolve_prompt_template_path(profile, profile_path)

    assert resolved == template_path


def test_template_override_prefers_explicit_file(tmp_path: Path, monkeypatch) -> None:
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    named_template = templates_dir / "v3.txt"
    named_template.write_text("Named {event_name}.", encoding="utf-8")
    file_template = tmp_path / "override.txt"
    file_template.write_text("Override {event_name}.", encoding="utf-8")

    monkeypatch.setattr(boot, "PROMPT_TEMPLATES_DIR", templates_dir)

    resolved = boot._resolve_template_override_path(
        template_name="v3",
        template_file=str(file_template),
    )

    assert resolved == file_template


def test_load_system_prompt_uses_template_override(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "event_name": "Test Expo",
                "location_name": "Test Hall",
                "organizer_name": "Test Org",
                "drink": "espresso",
                "context": "Test context.",
            }
        ),
        encoding="utf-8",
    )

    override_template = tmp_path / "override.txt"
    override_template.write_text("Override {event_name}.", encoding="utf-8")

    prompt = boot._load_system_prompt(profile_path, template_path=override_template)

    assert prompt == "Override Test Expo."


def test_load_system_prompt_preserves_clean_time_token(tmp_path: Path) -> None:
    template_path = tmp_path / "template.txt"
    template_path.write_text(
        "Event {event_name}. Time {clean_time}.",
        encoding="utf-8",
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "event_name": "Test Expo",
                "location_name": "Test Hall",
                "organizer_name": "Test Org",
                "drink": "espresso",
                "context": "Test context.",
            }
        ),
        encoding="utf-8",
    )

    prompt = boot._load_system_prompt(profile_path, template_path=template_path)

    assert "{clean_time}" in prompt

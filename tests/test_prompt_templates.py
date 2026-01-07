import datetime

from app.prompt_templates import CLEAN_TIME_TOKEN, render_clean_time


def test_render_clean_time_replaces_token() -> None:
    template = f"Time {CLEAN_TIME_TOKEN}."
    now = datetime.datetime(2024, 1, 1, 12, 0, 0)

    assert render_clean_time(template, now=now) == "Time 2024-01-01 12:00:00."


def test_render_clean_time_no_token_returns_template() -> None:
    template = "No time here."

    assert render_clean_time(template) == template


def test_render_clean_time_none_returns_none() -> None:
    assert render_clean_time(None) is None

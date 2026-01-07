from __future__ import annotations

import datetime
from typing import Optional

CLEAN_TIME_TOKEN = "{clean_time}"


def current_time_string(now: Optional[datetime.datetime] = None) -> str:
    timestamp = now or datetime.datetime.now()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def render_clean_time(template: Optional[str], *, now: Optional[datetime.datetime] = None) -> Optional[str]:
    if not isinstance(template, str):
        return None
    if CLEAN_TIME_TOKEN not in template:
        return template
    return template.replace(CLEAN_TIME_TOKEN, current_time_string(now))

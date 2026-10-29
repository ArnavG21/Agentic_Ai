"""Reads OpenStreetMap opening_hours strings well enough to answer one question:
"is this place open at any time during the meal the traveller wants?"

Handles the common forms, e.g.
    "Mo-Fr 08:00-16:00; Sa,Su 10:00-23:30"
    "Tu-Su 13:00-16:00,20:00-00:00; Mo off"
    "24/7"
Anything more exotic returns None (unknown) rather than a wrong answer.
"""
from __future__ import annotations

import re
from typing import Optional

MEAL_WINDOWS = {  # minutes after midnight - broad enough for meal times around the world
    "breakfast": (7 * 60 + 30, 10 * 60 + 30),   # 7:30-10:30
    "brunch": (10 * 60, 13 * 60),               # 10:00-13:00
    "lunch": (12 * 60, 15 * 60),                # 12:00-15:00
    "dinner": (18 * 60 + 30, 22 * 60),          # 18:30-22:00
}
RANGE_RE = re.compile(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\+?")


def intervals(opening_hours: str) -> Optional[list[tuple[int, int]]]:
    """All opening intervals (minutes) mentioned for any day; None if unreadable."""
    if not opening_hours:
        return None
    oh = opening_hours.strip()
    if oh == "24/7":
        return [(0, 24 * 60)]
    found = []
    for h1, m1, h2, m2 in RANGE_RE.findall(oh):
        start, end = int(h1) * 60 + int(m1), int(h2) * 60 + int(m2)
        if end <= start:          # closes after midnight, e.g. 20:00-01:00
            end += 24 * 60
        found.append((start, end))
    return found or None


def open_for_meal(opening_hours: str, meal: str) -> Optional[bool]:
    if meal not in MEAL_WINDOWS:
        return None
    ivs = intervals(opening_hours)
    if ivs is None:
        return None
    lo, hi = MEAL_WINDOWS[meal]
    return any(s < hi and e > lo for s, e in ivs)

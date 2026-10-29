"""Step 3 - shortlist. Plain yes/no checks, no AI and no points: every decision has a written reason.

1. Checks. A restaurant is left out only when the evidence clearly says it doesn't fit:
   - the map says it doesn't serve vegetarian/vegan food (for veg/vegan requests);
   - its opening hours show it's closed at the meal time;
   - a guide gives it a rating below 3.8 out of 5;
   - a guide quotes a price per person well above the budget.
   When a fact is unknown, the restaurant is NOT left out - it's marked "?" and
   checked later on the menu and in reviews.
2. Which ones to look at in depth: of those that pass, the ones recommended by the
   most food guides (ties: found on the map first, then has a website, so its menu
   can be read).
"""
from __future__ import annotations

from typing import Optional

from .hours import open_for_meal
from .models import Candidate, ShortlistRow, UserQuery
from .money import fmt_price
from .osm import diet_flag

OVER_BUDGET = 1.25   # leave out when a guide's price is more than 25% above the budget limit
MIN_RATING = 3.8
DIET_WORDS = {"veg": "vegetarian", "vegan": "vegan"}


def guide_count(c: Candidate) -> int:
    return len({m.url for m in c.mentions})


def check_candidate(c: Candidate, q: UserQuery, min_rating: float = MIN_RATING) -> ShortlistRow:
    reasons: list[str] = []
    failed = False

    diet: Optional[bool] = diet_flag(c.tags, q.diet) if q.diet in DIET_WORDS else None
    if q.diet in DIET_WORDS:
        if diet is False:
            reasons.append(f"✗ Map says no {DIET_WORDS[q.diet]} food")
            failed = True
        elif diet is True:
            reasons.append(f"✓ Map confirms {DIET_WORDS[q.diet]} food")
        else:
            reasons.append(f"? {DIET_WORDS[q.diet].capitalize()} food not recorded - will check the menu")

    meal = open_for_meal(c.opening_hours, q.meal_type)
    if q.meal_type != "any":
        if meal is False:
            reasons.append(f"✗ Closed at {q.meal_type} time ({c.opening_hours})")
            failed = True
        elif meal is True:
            reasons.append(f"✓ Open at {q.meal_type} time")
        else:
            reasons.append("? Opening hours unknown")

    if c.rating_hint is not None:
        if c.rating_hint < min_rating:
            reasons.append(f"✗ Guide rating {c.rating_hint:g}/5 is below {min_rating:g}")
            failed = True
        else:
            reasons.append(f"✓ Guide rating {c.rating_hint:g}/5")

    if q.budget_max is not None and c.guide_price:
        price, cap = fmt_price(c.guide_price, q.currency), fmt_price(q.budget_max, q.currency)
        if c.guide_price > q.budget_max * OVER_BUDGET:
            reasons.append(f"✗ A guide quotes about {price} per person, above your {cap} limit")
            failed = True
        elif c.guide_price <= q.budget_max:
            reasons.append(f"✓ A guide quotes about {price} per person, within your {cap} limit")
        else:
            reasons.append(f"? A guide quotes about {price} per person, a little above your {cap} limit")
    elif q.budget_max is not None:
        reasons.append("? Price not known yet - worked out from the menu later")

    n = guide_count(c)
    reasons.append(f"✓ Recommended by {n} food guide{'s' if n != 1 else ''}" if n else "? No guide mentions kept")
    if not c.on_map:
        reasons.append("? Not found on OpenStreetMap - address unconfirmed")

    return ShortlistRow(place_id=c.place_id, name=c.name, passed=not failed, guides=n, reasons=reasons)


def shortlist(cands: list[Candidate], q: UserQuery, size: int = 4,
              min_rating: float = MIN_RATING) -> tuple[list[ShortlistRow], list[Candidate]]:
    """Check every candidate; return all verdicts, and the ones to look at in depth."""
    by_id = {c.place_id: c for c in cands}
    rows = [check_candidate(c, q, min_rating) for c in cands]
    # passed first; then most guides; ties: on the map, then has a website (its menu can be read)
    rows.sort(key=lambda r: (r.passed, r.guides, by_id[r.place_id].on_map, bool(by_id[r.place_id].website)),
              reverse=True)
    chosen = [r for r in rows if r.passed][:size]
    for r in chosen:
        r.chosen = True
    return rows, [by_id[r.place_id] for r in chosen]

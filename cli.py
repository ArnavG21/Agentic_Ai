"""Run the agent from the terminal - handy for debugging.

    python cli.py "vegetarian dinner in Barcelona near Sagrada Familia, under 35 euros"
"""
from __future__ import annotations

import sys

from scout.config import Settings
from scout.links import all_links
from scout.pipeline import STAGES, Agent, save_run
from scout.money import budget_text, fmt_price, fmt_range

TITLES = {k: f"{i}. {t}" for i, (k, t, _) in enumerate(STAGES, 1)}


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python cli.py "vegetarian dinner in Barcelona under 35 euros"', file=sys.stderr)
        return 1
    s = Settings.from_env()
    if not s.has_ai:
        print("Add GEMINI_API_KEY (and optionally GROQ_API_KEY) to .env - see .env.example", file=sys.stderr)
        return 1
    state = None
    for ev in Agent.from_settings(s).run(" ".join(sys.argv[1:])):
        if ev.kind == "start":
            print(f"\n=== {TITLES[ev.stage]}")
        elif ev.kind == "log":
            print(f"  - {ev.message}")
        elif ev.kind == "error":
            print(f"  !! {ev.message}")
        elif ev.kind == "final":
            state = ev.payload

    if state and state.recommendation:
        print(f"\n=== WHERE TO EAT  (budget: {budget_text(state.query)})\n" + state.recommendation.summary)
        by_id = {d.candidate.place_id: d for d in state.dossiers}
        for p in state.recommendation.picks:
            d = by_id[p.place_id]
            print(f"\n* {p.name}: {p.headline}")
            if d.cost:
                print(f"   Cost per person:    {fmt_range(d.cost.low, d.cost.high, d.cost.currency)} - {d.cost.verdict}")
            for ln in all_links(d, state.query.city):
                print(f"   {ln.label + ':':<20} {ln.url}")
            for o in p.what_to_order:
                print(f"   * {o.item} .... {fmt_price(o.price, o.currency) or 'price not listed'}")
            if d.menu.items:
                print(f"   Menu ({len(d.menu.items)} dishes): "
                      + "; ".join(f"{it.english_name} {fmt_price(it.price, it.currency)}" for it in d.menu.items[:8]))
    if state:
        print(f"\nSaved to {save_run(state)}  |  {state.usage}")
    return 0 if state and not state.error else 2


if __name__ == "__main__":
    sys.exit(main())

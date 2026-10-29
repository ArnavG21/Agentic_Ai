"""Checks each free service on its own, so a problem can be pinned to one service.

Used by the "Check services" button in the app and by `python check.py`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterator

from .config import Settings
from .llm import LLM, build_routes
from .osm import OSM
from .places import GooglePlaces
from .scrape import Scraper
from .search import GroqSearch, WebSearch

TEST_PAGE = "https://en.wikipedia.org/wiki/Paella"


@dataclass
class Check:
    service: str
    ok: bool
    seconds: float
    detail: str


def _run(service: str, fn: Callable[[], str]) -> Check:
    t = time.monotonic()
    try:
        detail = fn()
    except Exception as e:  # report the problem in words and keep checking the others
        return Check(service, False, round(time.monotonic() - t, 1), f"{type(e).__name__}: {str(e)[:220]}")
    return Check(service, True, round(time.monotonic() - t, 1), detail)


def _nonzero(n: int, what: str) -> str:
    if n == 0:
        raise RuntimeError(f"0 {what}: the service answered but returned nothing (it may be blocking you)")
    return f"{n} {what}"


def iter_checks(s: Settings, ai_timeout: float = 25.0) -> Iterator[tuple[str, Callable[[], Check]]]:
    """Yields (service name, run) pairs, so callers can show each result as soon as it's ready."""
    smart, fast = build_routes(s, timeout=ai_timeout)
    if not smart:
        yield "AI", lambda: Check("AI", False, 0.0, "No AI key found. Add a free Gemini key (and/or Groq key).")
    seen = set()
    for r in smart + fast:   # every model the app may use - each has its own free allowance
        if r.label in seen:
            continue
        seen.add(r.label)
        llm = LLM([r], [r])
        name = f"AI: {r.label}"
        yield name, (lambda llm=llm, name=name: _run(name, lambda: "replied: " + llm.chat(
            system="Answer in one word.", messages=[{"role": "user", "content": "Capital of Spain?"}],
            max_tokens=600)[:30]))
    yield "Web search: DuckDuckGo", lambda: _run("Web search: DuckDuckGo", lambda: _nonzero(
        len(WebSearch(min_gap=0).search("best vegetarian restaurants Barcelona")), "results"))
    if s.groq_key:
        yield "Web search backup: Groq", lambda: _run("Web search backup: Groq", lambda: _nonzero(
            len(GroqSearch(s.groq_key, per_minute=60)("best vegetarian restaurants Barcelona", "es-es", 5)), "results"))
    osm = OSM(s.contact_email, timeout=15)
    yield "Map: OpenStreetMap", lambda: _run("Map: OpenStreetMap", lambda: (
        osm.locate_city("Barcelona") or _fail("Barcelona not found")).name)
    if s.google_maps_key:
        yield "Google Maps (open/closed check)", lambda: _run("Google Maps (open/closed check)", lambda: (
            lambda g: f"Sagrada Família: {g.status_words}" if g else _fail("no result"))(
            GooglePlaces(s.google_maps_key, timeout=15).find("Basílica de la Sagrada Família", "Barcelona")))
    yield "Page reading: direct", lambda: _run("Page reading: direct", lambda: _nonzero(
        len(Scraper(use_jina=False, timeout=15).read(TEST_PAGE).text), "characters"))
    yield "Page reading: Jina Reader", lambda: _run("Page reading: Jina Reader", lambda: _nonzero(
        len(Scraper(jina_key=s.jina_key, timeout=15).jina(TEST_PAGE).text), "characters"))


def check_all(s: Settings) -> list[Check]:
    return [run() for _name, run in iter_checks(s)]


def _fail(msg: str):
    raise RuntimeError(msg)

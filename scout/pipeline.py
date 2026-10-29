"""The agent: runs the nine stages in order and streams what it is doing.

run() is a generator. It yields Event objects (stage started / log line /
stage finished) so the Streamlit UI can show each step live and the CLI can
print them. The final event carries the complete RunState, which can be saved
to JSON and replayed later without any API calls.

Design choices worth knowing:
- A fixed pipeline, not a free-roaming agent loop: the brief asks to *see* each
  step, so each step is explicit, ordered and inspectable. The AI is used where
  language or judgement is needed (understanding the request, reading guides,
  translating menus, summarising reviews, writing suggestions). Filtering, checks, menu-link
  finding, opening-hours checks and fact-checking are plain code.
- Free services only: Gemini / Groq free AI plans, DuckDuckGo search (Groq search
  as backup), OpenStreetMap, and Jina Reader for pages that need a real browser.
  All are rate limited, so calls are spaced out.
- Per-restaurant work (stages 4-7) runs in parallel threads; one restaurant
  failing (site down, no menu) degrades that dossier but never stops the run.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from pydantic import BaseModel, Field

from . import discover, menu, recommend, reviews, shortlist, status, understand
from .config import Settings
from .llm import LLM
from .money import budget_text, estimate_cost, fmt_range
from .models import Candidate, Dossier, Recommendation, ReviewInsights, ShortlistRow, UserQuery
from .osm import OSM
from .places import GooglePlaces, PlacesError
from .scrape import Scraper
from .web import fetch_bytes
from .search import Evidence, GroqSearch, WebSearch, region_for
from .europe import EXAMPLE_CITIES, is_european
from .understand import NeedMoreInfo

STAGES: list[tuple[str, str, str]] = [
    ("understand", "Understand the request", "The AI turns your message into structured search parameters."),
    ("discover", "Identify restaurants", "Web search for food guides, then each place is checked on OpenStreetMap."),
    ("shortlist", "Shortlist", "Simple yes/no checks, then the places most food guides recommend, checked for closure news."),
    ("menu", "Read menus", "Open each website and find its menu page or PDF. Jina Reader steps in for pages that need a real browser."),
    ("translate", "Translate menus", "The AI translates each menu into your language (any language in, English by default out): dish, price, diet."),
    ("reviews", "Positive and negative reviews", "Search for reviews, including complaints, and sort praise from problems."),
    ("context", "Reviews about your needs", "Keyword scan plus searches about exactly what you asked for."),
    ("dishes", "Regular menu items and prices", "Dishes that fit your diet, with prices, and the cost per person worked out from the menu and compared with your budget."),
    ("recommend", "Suggestions", "The AI writes up each shortlisted place from the evidence, then code fact-checks it."),
]


class RunState(BaseModel):
    started_at: str = ""
    request: str = ""
    query: Optional[UserQuery] = None
    guide_sources: list[dict] = Field(default_factory=list)   # pages read during discovery
    candidates: list[Candidate] = Field(default_factory=list)
    shortlist_rows: list[ShortlistRow] = Field(default_factory=list)
    dossiers: list[Dossier] = Field(default_factory=list)
    stage_logs: dict[str, list[str]] = Field(default_factory=dict)
    recommendation: Optional[Recommendation] = None
    checks: list[str] = Field(default_factory=list)
    usage: dict = Field(default_factory=dict)
    provider: str = ""
    error: str = ""
    needs_info: bool = False       # True when the request was missing the city - not a real failure


@dataclass
class Event:
    stage: str
    kind: str                      # start | log | done | error | final
    message: str = ""
    payload: Any = None


@dataclass
class Agent:
    settings: Settings
    llm: LLM
    web: WebSearch
    osm: OSM
    scraper: Scraper
    places: Optional[GooglePlaces] = None      # Google Maps open/closed check, only with a key
    probe: Callable[[str], Any] = status.probe_website   # checks the restaurant's own website
    fetch_bytes: Optional[Callable[[str], tuple]] = fetch_bytes   # downloads menu photos and sitemaps
    state: RunState = field(default_factory=RunState)

    def __post_init__(self) -> None:
        self._gm = self.places          # kept for the usage count even if the check is switched off

    @classmethod
    def from_settings(cls, s: Settings) -> "Agent":
        scraper = Scraper(jina_key=s.jina_key, timeout=s.http_timeout, use_jina=s.use_jina)
        groq = GroqSearch(s.groq_key) if s.groq_key else None
        if s.search_mode == "groq" and groq:
            web = WebSearch(backend=groq, min_gap=0, fetch=scraper.read)
        else:
            web = WebSearch(fallback=groq if s.search_mode != "duckduckgo" else None, fetch=scraper.read)
        places = GooglePlaces(s.google_maps_key, timeout=s.http_timeout) if s.google_maps_key else None
        return cls(settings=s, llm=LLM.from_settings(s), web=web, osm=OSM(s.contact_email), scraper=scraper,
                   places=places)

    # ------------------------------------------------------------------ utils
    def _log(self, stage: str, msg: str) -> Event:
        self.state.stage_logs.setdefault(stage, []).append(msg)
        return Event(stage, "log", msg)

    def _parallel(self, stage: str, fn: Callable[[Dossier], str]) -> Iterator[Event]:
        """Run fn on every dossier in threads; yield one log line per restaurant as it finishes."""
        with ThreadPoolExecutor(max_workers=self.settings.max_workers) as pool:
            futures = {pool.submit(fn, d): d for d in self.state.dossiers}
            for fut in as_completed(futures):
                d = futures[fut]
                try:
                    yield self._log(stage, f"{d.candidate.name}: {fut.result()}")
                except Exception as e:  # keep going - one bad site must not sink the run
                    yield self._log(stage, f"⚠ {d.candidate.name}: {e}")

    def _fill_open(self, stage: str, rows: list[ShortlistRow], by_cand: dict, q: UserQuery, today: str,
                   top: list[Candidate], ok: Callable[[ShortlistRow], bool],
                   limit: int | None = None) -> Iterator[Event]:
        """Go down the rows in order; keep each one that is still open until `top` is full."""
        limit = limit or self.settings.shortlist_size
        st = self.state
        for r in rows:
            c = by_cand[r.place_id]
            if not ok(r) or r.closed or c in top or len(top) >= limit:
                continue
            try:
                stat = status.check_open(c, q, self.web, self.scraper.read, self.llm, today, places=self.places,
                                         probe=self.probe)
            except PlacesError as e:          # bad key or daily limit: say so, carry on without Google
                yield self._log(stage, f"⚠ {e} Continuing with the free checks.")
                st.usage["google_maps_problem"] = str(e)
                self.places = None
                try:
                    stat = status.check_open(c, q, self.web, self.scraper.read, self.llm, today, probe=self.probe)
                except Exception as e2:  # noqa: BLE001
                    stat = status.OpenStatus(False, f"Couldn't check whether it's still open ({e2}).")
            except Exception as e:  # noqa: BLE001 - a failed check must not block the search
                stat = status.OpenStatus(False, f"Couldn't check whether it's still open ({e}).")
            r.open_checks = stat.checked or ["attempted"]
            if stat.maps_url:
                c.maps_url = stat.maps_url        # link straight to the place on Google Maps
            if stat.closed:
                r.passed, r.closed, r.closed_source = False, True, stat.source_url
                r.closed_temporarily = stat.temporary
                word = "temporarily closed" if stat.temporary else "closed for good"
                r.reasons.insert(0, f"✗ {word[0].upper()}{word[1:]}: {stat.note}")
                yield self._log(stage, f"✗ {c.name}: {word}. {stat.note} Skipped; checking the next place.")
                continue
            r.open_confirmed = stat.confirmed_open
            r.reasons.append(("✓ Open on Google Maps: " if stat.confirmed_open else "✓ No sign of a closure: ")
                             + stat.note + f" (checked: {', '.join(stat.checked)})")
            r.chosen = True
            top.append(c)
            yield self._log(stage, f"✓ {c.name}: {stat.note}")

    # ------------------------------------------------------------------- run
    def run(self, request: str, menu_language: str = "English", fixed: dict | None = None) -> Iterator[Event]:
        s = self.settings
        st = self.state = RunState(started_at=dt.datetime.now().isoformat(timespec="seconds"), request=request)
        stage = "understand"
        try:
            # 1 ------------------------------------------------------------
            yield Event(stage, "start")
            q = st.query = understand.understand(self.llm, request, menu_language, fixed)
            yield self._log(stage, f"{q.city} ({q.country_code or '?'}), meal: {q.meal_type}, diet: {q.diet}")
            yield self._log(stage, f"Budget: {budget_text(q)}")
            if q.requirements:
                yield self._log(stage, "Other needs: " + "; ".join(q.requirements))
            yield self._log(stage, "Review keywords: " + ", ".join(q.review_keywords))
            if q.local_menu_words:
                yield self._log(stage, f"Local words for 'menu' ({q.local_language}): " + ", ".join(q.local_menu_words))
            yield Event(stage, "done")

            # 2 ------------------------------------------------------------
            stage = "discover"
            yield Event(stage, "start")
            city = self.osm.locate_city(q.city)
            if city and city.country_code and not is_european(city.country_code):
                raise NeedMoreInfo(f"{q.city} looks like it's outside Europe on the map, and Restaurant Scout "
                                   f"covers European cities only. Try, for example, {EXAMPLE_CITIES}.")
            if city:
                q.country_code = q.country_code or city.country_code
                yield self._log(stage, f"Found {city.name} on OpenStreetMap")
            else:
                yield self._log(stage, f"⚠ Could not find {q.city} on OpenStreetMap, so addresses can't be confirmed")
            self.web.region = region_for(q.country_code)
            for qq in q.search_queries:
                yield self._log(stage, f'Searching: "{qq}"')
            cands, ev = discover.find_in_guides(q, self.llm, self.web, pages=5)
            st.guide_sources = [{"n": x["n"], "url": x["url"], "title": x["title"], "read": bool(x["text"]),
                                 "via": x.get("via", "")} for x in ev.sources]
            if self.web.fallback_count:
                yield self._log(stage, f"DuckDuckGo returned nothing for {self.web.fallback_count} search(es), "
                                       "so Groq's web search was used instead")
            jina = sum(x["via"] == "jina" for x in st.guide_sources)
            yield self._log(stage, f"Read {sum(x['read'] for x in st.guide_sources)} guide pages"
                                   + (f" ({jina} with the Jina Reader scraper)" if jina else "") + " and "
                                   f"{len(ev.sources)} search results; {len(cands)} restaurants recommended")
            if not cands:                                   # backup 1: broader guide searches
                why = ("the web searches returned nothing (DuckDuckGo sometimes blocks for a few minutes)"
                       if not ev.sources else "the pages found didn't name any restaurants")
                yield self._log(stage, f"⚠ No restaurants yet: {why}. Trying broader searches.")
                food = {"veg": "vegetarian ", "vegan": "vegan "}.get(q.diet, "")
                broad = list(dict.fromkeys([f"best {food}restaurants {q.city}", f"where to eat in {q.city}",
                                            f"best restaurants {q.city} guide"]))
                for qq in broad:
                    yield self._log(stage, f'Searching: "{qq}"')
                cands, ev2 = discover.find_in_guides(q.model_copy(update={"search_queries": broad}),
                                                     self.llm, self.web, pages=5)
                st.guide_sources += [{"n": len(st.guide_sources) + i + 1, "url": x["url"], "title": x["title"],
                                      "read": bool(x["text"]), "via": x.get("via", "")} for i, x in enumerate(ev2.sources)]
                yield self._log(stage, f"{len(cands)} restaurants recommended by the broader searches")
            if not cands and self.places:                   # backup 2: Google Maps' own list
                try:
                    cands = discover.from_google(q, city, self.places)
                    yield self._log(stage, f"Used Google Maps instead: {len(cands)} open restaurants listed")
                except PlacesError as e:
                    yield self._log(stage, f"⚠ Google Maps backup failed: {e}")
            if not cands and city:                          # backup 3: restaurants on OpenStreetMap
                cands = discover.from_map(q, city, self.osm)
                yield self._log(stage, f"Used OpenStreetMap instead: {len(cands)} restaurants mapped in {city.name}")
            if not cands:
                raise RuntimeError("Couldn't find any restaurants: web search returned nothing and the map had no "
                                   "matches. This is usually a short block by DuckDuckGo - wait a few minutes and try "
                                   "again, or click Check services in the sidebar.")
            if city:
                for c in cands[: s.candidates_to_verify]:
                    if c.on_map:
                        continue
                    found = discover.verify_on_map(c, city, self.osm)
                    yield self._log(stage, f"{'📍' if found else '·'} {c.name}: "
                                           + (f"on the map at {c.address}" if found else "not found on the map"))
            merged: dict[str, Candidate] = {}   # two spellings can resolve to the same map entry
            for c in cands:
                if c.place_id in merged:
                    known = {m.url for m in merged[c.place_id].mentions}
                    merged[c.place_id].mentions += [m for m in c.mentions if m.url not in known]
                else:
                    merged[c.place_id] = c
            st.candidates = cands = list(merged.values())
            yield Event(stage, "done")

            # 3 ------------------------------------------------------------
            stage = "shortlist"
            yield Event(stage, "start")
            rows, _ = shortlist.shortlist(cands, q, s.shortlist_size)
            st.shortlist_rows = rows
            passed = sum(r.passed for r in rows)
            how = ("Google Maps, their websites and closure news" if self.places
                   else "their websites, review listings and closure news")
            yield self._log(stage, f"{passed} of {len(rows)} passed the checks. Now making sure the most "
                                   f"recommended ones are still open ({how}).")
            by_cand = {c.place_id: c for c in cands}
            today = dt.date.today().isoformat()
            top: list[Candidate] = []
            for r in rows:
                r.chosen = False
            yield from self._fill_open(stage, rows, by_cand, q, today, top, lambda r: r.passed)

            want = min(2, s.shortlist_size)
            if len(top) < want:               # too few: allow places that only missed a soft check
                soft = [r for r in rows if not r.passed and not r.closed and not r.open_checks
                        and all(not x.startswith("✗") or ("rating" in x or "per person" in x) for x in r.reasons)]
                if soft:
                    yield self._log(stage, f"Only {len(top)} place(s) passed every check, so places that only missed "
                                           "the rating or guide-price check are considered too.")
                    for r in soft:
                        r.reasons.insert(0, "↻ Added because too few places passed every check")
                    yield from self._fill_open(stage, soft, by_cand, q, today, top, lambda r: True, limit=want)

            if len(top) < want:               # still too few: more places from Google Maps or the map
                extra = []
                try:
                    if self.places:
                        extra = discover.from_google(q, city, self.places)
                        src = "Google Maps"
                except PlacesError as e:
                    yield self._log(stage, f"⚠ Google Maps: {e}")
                if not extra and city:
                    extra, src = discover.from_map(q, city, self.osm), "OpenStreetMap"
                known = {c.name.lower() for c in cands}
                extra = [c for c in extra if c.name.lower() not in known]
                if extra:
                    yield self._log(stage, f"Not enough open places from the guides, so {len(extra)} more came "
                                           f"from {src}.")
                    more_rows, _ = shortlist.shortlist(extra, q, s.shortlist_size)
                    for r in more_rows:
                        r.chosen = False
                    rows += more_rows
                    cands += extra
                    st.candidates = cands
                    by_cand.update({c.place_id: c for c in extra})
                    yield from self._fill_open(stage, more_rows, by_cand, q, today, top, lambda r: r.passed,
                                               limit=want)
            st.shortlist_rows = rows
            by_id = {r.place_id: r for r in rows}
            st.dossiers = [Dossier(candidate=c, shortlist=by_id[c.place_id]) for c in top]
            if top:
                yield self._log(stage, f"Looking in depth at the {len(top)} most recommended places that are "
                                       "still open: " + ", ".join(c.name for c in top))
            yield Event(stage, "done")
            if not top:
                raise RuntimeError("Every place found was closed or didn't fit your diet or meal time. Try another "
                                   "meal time, a wider budget, or fewer extra requirements.")

            # 4 + 5 ---------------------------------------------------------
            raw_menus: dict[str, str] = {}

            def do_read(d: Dossier) -> str:
                vision = self.llm.read_images if self.llm.can_see else None
                res, text = menu.read_menu(d.candidate, q, self.web, self.scraper.read, s.menu_char_limit,
                                           vision=vision, fetch=self.fetch_bytes)
                d.menu, raw_menus[d.candidate.place_id] = res, text
                if res.how_found == "none":
                    return f"no readable menu found ({len(res.pages_tried)} pages checked)"
                how = res.how_found.replace("+jina", ", read with Jina Reader")
                return f"menu found on the {how} ({len(res.pages_tried)} pages checked)"

            stage = "menu"
            yield Event(stage, "start")
            yield from self._parallel(stage, do_read)
            yield Event(stage, "done")

            def do_translate(d: Dossier) -> str:
                if not raw_menus.get(d.candidate.place_id):
                    return "skipped (no menu text)"
                menu.translate_menu(raw_menus[d.candidate.place_id], d.candidate, q, self.llm, d.menu)
                priced = sum(1 for it in d.menu.items if it.price is not None)
                return (f"translated from {d.menu.detected_language or '?'} into {d.menu.translated_to}: "
                        f"{len(d.menu.items)} dishes ({priced} with prices)")

            stage = "translate"
            yield Event(stage, "start")
            yield from self._parallel(stage, do_translate)
            yield Event(stage, "done")

            # 6 + 7 ---------------------------------------------------------
            general: dict[str, Evidence] = {}

            def do_reviews(d: Dossier) -> str:
                c = d.candidate
                d.reviews = ReviewInsights()
                ev = general[c.place_id] = reviews.gather_general(c, q, self.web, s.pages_per_search)
                reviews.summarise_balance(c, ev, self.llm, d.reviews)
                return (f"{len(ev.sources)} review sources, {sum(bool(x['text']) for x in ev.sources)} read in full: "
                        f"{len(d.reviews.positives)} positives, {len(d.reviews.negatives)} negatives")

            stage = "reviews"
            yield Event(stage, "start")
            yield from self._parallel(stage, do_reviews)
            yield Event(stage, "done")

            def do_context(d: Dossier) -> str:
                c = d.candidate
                ev = reviews.gather_context(c, q, self.web)
                reviews.check_context(c, q, ev, general.get(c.place_id, Evidence()), self.llm, d.reviews)
                verdicts = ", ".join(f"{f.requirement}: {f.verdict.replace('_', ' ')}" for f in d.reviews.context)
                return f"{len(d.reviews.keyword_hits)} keyword matches. {verdicts}"

            stage = "context"
            yield Event(stage, "start")
            yield from self._parallel(stage, do_context)
            yield Event(stage, "done")

            # 8 ------------------------------------------------------------
            stage = "dishes"
            yield Event(stage, "start")
            for d in st.dossiers:
                items = reviews.regular_items(d.menu, d.reviews, q.diet)
                pop = sum(it.popular for it in items)
                d.cost = estimate_cost(d.menu.items, q, d.candidate.guide_price)
                cost = (f"Estimated {fmt_range(d.cost.low, d.cost.high, d.cost.currency)} per person. {d.cost.verdict}"
                        if d.cost else "Cost per person unknown (no prices found).")
                yield self._log(stage, f"{d.candidate.name}: {len(items)} dishes fit your diet "
                                       f"({pop} mentioned in reviews). {cost}")
            yield Event(stage, "done")

            # 9 ------------------------------------------------------------
            stage = "recommend"
            yield Event(stage, "start")
            rec, checks = recommend.recommend(q, st.dossiers, self.llm)
            st.recommendation, st.checks = rec, checks
            for c in checks:
                yield self._log(stage, "Fact-check: " + c)
            yield Event(stage, "done")
        except NeedMoreInfo as e:
            st.error, st.needs_info = str(e), True
            yield Event(stage, "error", st.error)
        except Exception as e:
            st.error = str(e) or type(e).__name__
            self._log(stage, "Stopped: " + st.error)
            yield Event(stage, "error", st.error, payload=traceback.format_exc())
        finally:
            st.provider = self.llm.main_label
            st.usage = {**st.usage, **self.llm.usage, "web_searches": self.web.count,
                        "groq_backup_searches": self.web.fallback_count, "pages": dict(self.scraper.stats),
                        "ai_notes": list(self.llm.notes), "google_maps_lookups": self._gm.lookups if self._gm else 0}
        yield Event("final", "final", payload=st)


def save_run(state: RunState, folder: str | Path = "runs") -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    city = re.sub(r"[^a-z0-9]+", "-", (state.query.city if state.query else "run").lower()).strip("-")
    path = folder / f"{state.started_at.replace(':', '').replace('-', '')}_{city}.json"
    path.write_text(state.model_dump_json(indent=1), encoding="utf-8")
    return path


def load_run(path: str | Path) -> RunState:
    return RunState(**json.loads(Path(path).read_text(encoding="utf-8")))

"""Is the restaurant still open? Food guides and maps go out of date, so every place the agent is
about to recommend is checked before any time is spent on it. A place that fails is dropped and the
next one on the list takes its spot.

check_open() runs these checks for one restaurant, stopping as soon as one is decisive:
  1. OpenStreetMap: closed places carry tags like "disused:amenity".
  2. Google Maps (only with GOOGLE_MAPS_API_KEY): the same "Permanently closed" / "Temporarily
     closed" label people see on Google Maps. Decisive either way.
  3. The restaurant's own website: a domain that no longer exists, a removed page (404/410), or a
     "domain for sale" page are strong signs it has closed. A farewell message ("thank you for all
     these years", "último servicio") is passed to the AI below.
  4. Web search, three searches: "<name> <city> permanently closed", the same in the local language
     ("cierra", "fermé définitivement"...), and the restaurant's review-site listing (Tripadvisor and
     similar sites mark closed places "CLOSED" in the page title).
  5. If any of that looks suspicious, the AI reads those pages and decides - only from them -
     whether THIS restaurant has closed. Weekly closing days and similar names don't count.
"""
from __future__ import annotations

import re
import socket
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import urlparse

import httpx

from .europe import closed_words
from .llm import LLM
from .models import Candidate, UserQuery
from .places import GooglePlaces, PlacesError
from .scrape import readable
from .search import WebSearch
from .web import USER_AGENT, FetchError, Page

STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["open", "permanently_closed", "temporarily_closed", "unknown"]},
        "evidence": {"type": "string", "description": "One sentence: what the source says, with any date"},
        "source_number": {"type": ["integer", "null"], "description": "The source that says so, or null"},
    },
    "required": ["status", "evidence", "source_number"],
    "additionalProperties": False,
}

SYSTEM = """You check whether ONE restaurant has closed, using ONLY the numbered sources.
- "permanently_closed": a source clearly says THIS restaurant (same name, same city) has closed for good,
  is closing for good on a date that has passed, has served its last service, or a listing for it is
  marked "CLOSED" / "Permanently closed".
- "temporarily_closed": closed right now for holidays, renovation or a short period.
- "open": sources show it operating normally, or the closure news is about a different place.
- "unknown": the sources don't say.
Weekly closing days ("closed on Mondays") are NOT closures. Be careful with similar names."""

# Review and listing sites put "CLOSED" in the page title of a closed restaurant.
CLOSED_MARKERS = ["- closed", "closed -", "(closed)", "[closed]", "| closed", "· closed", "permanently closed",
                  "closed permanently", "cerrado permanentemente", "tancat definitivament", "définitivement fermé",
                  "chiuso definitivamente", "dauerhaft geschlossen", "encerrado permanentemente"]
PARKED_WORDS = ["domain is for sale", "domain may be for sale", "buy this domain", "this domain is parked",
                "domain has expired", "domain expired", "parked free", "hugedomains", "sedo domain parking",
                "dan.com", "afternic", "account has been suspended", "account suspended"]
FAREWELL_WORDS = ["thank you for all these years", "thanks for all these years", "we have closed", "we are closed for good",
                  "gracias por estos años", "gracias por todos estos años", "gràcies per aquests anys",
                  "merci pour toutes ces années", "grazie per questi anni", "danke für die schönen jahre",
                  "obrigado por estes anos", "bedankt voor al die jaren"]
SOCIAL = ("instagram.com", "facebook.com", "tiktok.com", "twitter.com", "x.com", "linktr.ee", "google.com",
          "tripadvisor.", "thefork.", "yelp.")


@dataclass
class OpenStatus:
    closed: bool
    note: str                          # what the evidence says, in one line
    source_url: str = ""
    confirmed_open: bool = False       # Google Maps says it's operating
    temporary: bool = False            # closed for now, not for good
    maps_url: str = ""                 # Google Maps page of the place, if Google found it
    checked: list[str] = field(default_factory=list)   # which checks ran


def _name_tokens(name: str) -> list[str]:
    return [t for t in re.findall(r"\w+", name.lower()) if len(t) >= 4] or [name.lower()]


def suspicious(text: str, name: str, words: list[str]) -> bool:
    """True when the text mentions the restaurant and a closure word or a "CLOSED" listing marker."""
    low = text.lower()
    return any(t in low for t in _name_tokens(name)) and any(w in low for w in words + CLOSED_MARKERS)


def osm_says_closed(c: Candidate) -> bool:
    tags = c.tags or {}
    return any(k.split(":")[0] in ("disused", "was", "abandoned", "demolished", "removed") for k in tags) \
        or tags.get("opening_hours", "").strip().lower() == "closed"


# ------------------------------------------------------------------ website
@dataclass
class SiteCheck:
    state: str          # "gone" (strong sign of closure) | "ok" | "unknown" (couldn't tell)
    note: str = ""
    text: str = ""


def _resolves(host: str, resolve: Callable = socket.getaddrinfo) -> bool:
    try:
        resolve(host, 443)
        return True
    except socket.gaierror:
        return False


def probe_website(url: str, resolve: Callable = socket.getaddrinfo,
                  transport: Optional[httpx.BaseTransport] = None, timeout: float = 12.0) -> SiteCheck:
    host = (urlparse(url).hostname or "").lower()
    if not host or any(s in host for s in SOCIAL):
        return SiteCheck("unknown")
    if not _resolves(host, resolve) and not _resolves(host, resolve):
        if _resolves("www.wikipedia.org", resolve):     # we're online, so the domain really is gone
            return SiteCheck("gone", f"its website domain ({host}) no longer exists")
        return SiteCheck("unknown")

    def get(u: str) -> Optional[httpx.Response]:
        try:
            with httpx.Client(follow_redirects=True, timeout=timeout, transport=transport,
                              headers={"User-Agent": USER_AGENT, "Accept-Language": "en,*;q=0.5"}) as c:
                return c.get(u)
        except httpx.HTTPError:
            return None

    r = get(url)
    if r is not None and r.status_code in (404, 410):
        root = f"{urlparse(url).scheme or 'https'}://{host}/"
        if url.rstrip("/") != root.rstrip("/"):
            r = get(root)                                  # a moved page isn't a closed restaurant
        if r is not None and r.status_code in (404, 410):
            return SiteCheck("gone", f"its website ({host}) has been taken down (HTTP {r.status_code})")
    if r is None or r.status_code >= 400:
        return SiteCheck("unknown")
    text = re.sub(r"<[^>]+>", " ", r.text[:200_000])
    text = re.sub(r"\s+", " ", text)
    low = text.lower()
    if any(w in low for w in PARKED_WORDS) and len(low) < 20_000:
        return SiteCheck("gone", f"its website ({host}) is now a parked or for-sale domain")
    return SiteCheck("ok", text=text)


# --------------------------------------------------------------------- main
def check_open(c: Candidate, q: UserQuery, web: WebSearch, read: Callable[..., Page], llm: LLM,
               today: str = "", places: Optional[GooglePlaces] = None,
               probe: Callable[[str], SiteCheck] = probe_website) -> OpenStatus:
    checked: list[str] = ["OpenStreetMap"]
    if osm_says_closed(c):
        return OpenStatus(True, "OpenStreetMap marks this place as closed.", c.osm_url, checked=checked)

    # Google Maps - decisive when it finds the place
    maps_url, g_note = "", ""
    if places is not None:
        checked.append("Google Maps")
        g = places.find(c.name, q.city, c.lat, c.lon)       # PlacesError is handled by the caller
        if g is not None:
            maps_url = g.maps_url
            if g.status == "CLOSED_PERMANENTLY":
                return OpenStatus(True, f"Google Maps lists \"{g.name}\" as permanently closed.", g.maps_url,
                                  maps_url=g.maps_url, checked=checked)
            if g.status == "CLOSED_TEMPORARILY":
                return OpenStatus(True, f"Google Maps lists \"{g.name}\" as temporarily closed.", g.maps_url,
                                  temporary=True, maps_url=g.maps_url, checked=checked)
            if g.status == "OPERATIONAL":
                return OpenStatus(False, "Google Maps lists it as open.", g.maps_url, confirmed_open=True,
                                  maps_url=g.maps_url, checked=checked)
        g_note = "Not found on Google Maps under this name. "

    # The restaurant's own website
    site = SiteCheck("unknown")
    if c.website:
        checked.append("its website")
        try:
            site = probe(c.website)
        except Exception:  # noqa: BLE001 - a broken check must never block the search
            site = SiteCheck("unknown")

    # Web search for closure news and "CLOSED" listings
    checked.append("web search")
    words = closed_words(q.country_code)
    local = next((w for w in words[len(closed_words("")):] if " " not in w), "")
    queries = [f'"{c.name}" {q.city} permanently closed', f'"{c.name}" {q.city} tripadvisor']
    if local:
        queries.insert(1, f'"{c.name}" {q.city} {local}')
    hits, seen = [], set()
    for query in queries:
        try:
            results = web.search(query, max_results=8)
        except Exception:  # noqa: BLE001
            results = []
        for h in results:
            if h.url not in seen and suspicious(f"{h.title} {h.snippet}", c.name, words):
                seen.add(h.url)
                hits.append(h)

    farewell = site.state == "ok" and any(w in site.text.lower() for w in FAREWELL_WORDS + words[:5])
    if hits or farewell:
        sources: list[tuple[int, str, str, str]] = []      # (number, title, url, text)
        if farewell:
            sources.append((1, f"{c.name} - own website", c.website, site.text[:2500]))
        for h in hits[:3]:
            text = ""
            if readable(h.url):
                try:
                    text = read(h.url).text[:2500]
                except FetchError:
                    text = ""
            sources.append((len(sources) + 1, h.title, h.url, text or h.snippet))
        prompt = (f"Restaurant: {c.name}, {q.city}. Today is {today or 'unknown'}.\n\nSources:\n"
                  + "\n\n".join(f"[{i}] {t}\nURL: {u}\n{x}" for i, t, u, x in sources))
        data = llm.structured(system=SYSTEM, prompt=prompt, schema=STATUS_SCHEMA, name="open_status", max_tokens=2000)
        url = {i: u for i, _, u, _ in sources}.get(data["source_number"] or 0, "")
        if data["status"] == "permanently_closed":
            return OpenStatus(True, data["evidence"] or "A source says it has closed permanently.", url,
                              maps_url=maps_url, checked=checked)
        if data["status"] == "temporarily_closed":
            return OpenStatus(True, data["evidence"] or "A source says it is temporarily closed.", url,
                              temporary=True, maps_url=maps_url, checked=checked)
        if site.state == "gone":
            return OpenStatus(True, f"{g_note}{site.note[0].upper()}{site.note[1:]}, a strong sign it has closed.",
                              c.website, maps_url=maps_url, checked=checked)
        note = {"open": "Checked closure news: ", "unknown": "Closure news was unclear: "}[data["status"]] \
            + (data["evidence"] or "")
        return OpenStatus(False, g_note + note, url, maps_url=maps_url, checked=checked)

    if site.state == "gone":
        return OpenStatus(True, f"{g_note}{site.note[0].upper()}{site.note[1:]}, a strong sign it has closed.",
                          c.website, maps_url=maps_url, checked=checked)
    extra = " Its website is working." if site.state == "ok" else ""
    return OpenStatus(False, f"{g_note}No sign of a closure on the web.{extra}", maps_url=maps_url, checked=checked)


__all__ = ["OpenStatus", "SiteCheck", "check_open", "probe_website", "suspicious", "osm_says_closed", "PlacesError"]

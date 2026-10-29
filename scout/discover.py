"""Stage 2 - identify candidate restaurants, using only free sources.

1. Web search for restaurant guides and lists matching the request
   (e.g. "best vegetarian restaurants Barcelona dinner") and open the top pages.
2. The AI reads those pages and lists every restaurant they recommend, with the
   numbers of the sources that mention it. A place recommended by several
   independent guides is a strong signal - that count is our popularity measure.
3. Each restaurant is looked up on OpenStreetMap inside the city. This confirms
   it exists where the guides say, and adds facts volunteers mapped: address,
   website, cuisine, vegetarian/vegan tags and opening hours.
"""
from __future__ import annotations

import re
from urllib.parse import quote_plus

from .llm import LLM
from .models import Candidate, Mention, UserQuery
from .money import budget_text
from .osm import OSM, City, diet_flag
from .schemas import GUIDE_EXTRACT
from .search import Evidence, WebSearch

EXTRACT_SYSTEM = """You read restaurant guides and lists from the web and extract the restaurants they
recommend. Rules:
- Only restaurants (or cafes/bars serving food) located in the traveller's city.
- Use only what the sources say. Never add restaurants from memory.
- Merge duplicates (same place written differently) into one entry listing all its source numbers.
- Skip chains and places a source says are closed.
- Return up to 25 restaurants, most-mentioned first."""


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def find_in_guides(q: UserQuery, llm: LLM, web: WebSearch, pages: int = 5) -> tuple[list[Candidate], Evidence]:
    ev = web.research(q.search_queries, pages=pages, max_sources=12)
    if not ev.sources:                       # nothing found: no point asking the AI
        return [], ev
    needs = ", ".join(filter(None, [
        f"diet: {q.diet}" if q.diet != "any" else "", f"meal: {q.meal_type}" if q.meal_type != "any" else "",
        f"budget: {budget_text(q)}" if q.budget_max or q.budget_min else "", *q.cuisines, *q.requirements]))
    data = llm.structured(
        system=EXTRACT_SYSTEM, name="guide_restaurants",
        prompt=f"City: {q.city}\nTraveller wants: {needs or 'anything good'}\n\nSources:\n{ev.as_prompt(3000)}",
        schema=GUIDE_EXTRACT,
    )
    by_n = {s["n"]: s for s in ev.sources}
    out: dict[str, Candidate] = {}
    for r in data["restaurants"]:
        key = _norm(r["name"])
        if not key:
            continue
        mentions = [Mention(url=by_n[n]["url"], title=by_n[n]["title"], why=r["why"])
                    for n in dict.fromkeys(r["source_numbers"]) if n in by_n]   # drop invented source numbers
        if key in out:
            out[key].mentions.extend(m for m in mentions if m.url not in {x.url for x in out[key].mentions})
            continue
        out[key] = Candidate(
            place_id="web:" + key.replace(" ", "-"), name=r["name"], area_hint=r["area"],
            rating_hint=r["rating"] if r["rating"] and 0 < r["rating"] <= 5 else None,
            guide_price=r["price_per_person"] if r["price_per_person"] and r["price_per_person"] > 0 else None,
            mentions=mentions,
            maps_url="https://www.google.com/maps/search/?api=1&query=" + quote_plus(f"{r['name']} {q.city}"),
        )
    cands = sorted(out.values(), key=lambda c: len(c.mentions), reverse=True)
    return cands, ev


def verify_on_map(c: Candidate, city: City, osm: OSM) -> bool:
    p = osm.find_place(c.name, city, c.area_hint)
    if p is None:
        return False
    c.on_map = True
    c.place_id = p.osm_url
    c.osm_url = p.osm_url
    c.lat, c.lon = p.lat, p.lon
    c.address = p.address
    c.tags = p.tags
    c.website = c.website or p.website
    c.cuisine = (p.tags.get("cuisine") or "").replace(";", ", ").replace("_", " ")
    c.opening_hours = p.tags.get("opening_hours", "")
    c.maps_url = "https://www.google.com/maps/search/?api=1&query=" + quote_plus(f"{c.name} {p.address}")
    return True


# ------------------------------------------------------------- backups
def _center(city: City | None) -> tuple[float | None, float | None]:
    return ((city.south + city.north) / 2, (city.west + city.east) / 2) if city else (None, None)


def from_google(q: UserQuery, city: City | None, places) -> list[Candidate]:
    """Backup when guides name nobody: what Google Maps lists for the request (open places only)."""
    food = {"veg": "vegetarian ", "vegan": "vegan "}.get(q.diet, "")
    meal = f" for {q.meal_type}" if q.meal_type not in ("any", "") else ""
    cuisine = f"{q.cuisines[0]} " if q.cuisines else ""
    lat, lon = _center(city)
    out = []
    for g in places.search(f"{food}{cuisine}restaurant{meal} in {q.city}", lat, lon)[:14]:
        key = _norm(g.name)
        out.append(Candidate(place_id="gmaps:" + key.replace(" ", "-"), name=g.name, address=g.address,
                             lat=g.lat, lon=g.lon, maps_url=g.maps_url,
                             mentions=[Mention(url=g.maps_url, title="Google Maps listing",
                                               why="Listed on Google Maps for this request")]))
    return out


def from_map(q: UserQuery, city: City, osm: OSM, limit: int = 14) -> list[Candidate]:
    """Last backup: restaurants volunteers mapped in the city, diet-tagged ones first."""
    food = {"veg": "vegetarian", "vegan": "vegan"}.get(q.diet, "")
    words = [w for w in [f"{food} restaurant".strip() if food else "", *(f"{c} restaurant" for c in q.cuisines[:1]),
                         "restaurant"] if w]
    found = osm.food_places(city, list(dict.fromkeys(words)))
    found.sort(key=lambda p: (diet_flag(p.tags, q.diet) is not True, not p.website))
    out = []
    for p in found[:limit]:
        out.append(Candidate(
            place_id=p.osm_url, name=p.name, address=p.address, lat=p.lat, lon=p.lon, on_map=True,
            osm_url=p.osm_url, tags=p.tags, website=p.website,
            cuisine=(p.tags.get("cuisine") or "").replace(";", ", ").replace("_", " "),
            opening_hours=p.tags.get("opening_hours", ""),
            maps_url="https://www.google.com/maps/search/?api=1&query=" + quote_plus(f"{p.name} {p.address}"),
            mentions=[Mention(url=p.osm_url, title="OpenStreetMap", why="Mapped as a restaurant in the city")]))
    return out

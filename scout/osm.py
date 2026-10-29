"""OpenStreetMap lookups through the free Nominatim API (no key).

- locate_city():  the city's bounding box, used to keep lookups inside the city
- find_place():   look a restaurant up by name inside that box and return the
                  map facts volunteers have recorded: address, website, cuisine,
                  diet tags (diet:vegetarian=yes/only), opening hours, phone.

Nominatim's usage policy: at most 1 request per second and a User-Agent that
identifies the app - both handled here.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

NOMINATIM = "https://nominatim.openstreetmap.org/search"
FOOD_TYPES = {"restaurant", "cafe", "fast_food", "bar", "pub", "food_court", "ice_cream", "bakery", "biergarten"}


@dataclass
class City:
    name: str
    country_code: str
    south: float
    north: float
    west: float
    east: float


@dataclass
class MapPlace:
    osm_url: str
    name: str
    lat: float
    lon: float
    kind: str
    address: str
    tags: dict = field(default_factory=dict)

    @property
    def website(self) -> str:
        return self.tags.get("website") or self.tags.get("contact:website") or ""


class OSM:
    def __init__(self, contact_email: str = "", timeout: float = 15.0,
                 transport: Optional[httpx.BaseTransport] = None, min_gap: float = 1.1):
        ua = "RestaurantScout/1.0 (github.com; student project)"
        if contact_email:
            ua += f" contact: {contact_email}"
        self.headers = {"User-Agent": ua, "Accept-Language": "en"}
        self.timeout = timeout
        self.transport = transport
        self.min_gap = min_gap
        self._lock = threading.Lock()
        self._last = 0.0

    def _get(self, params: dict) -> list[dict]:
        with self._lock:
            gap = self.min_gap - (time.monotonic() - self._last)
            if gap > 0:
                time.sleep(gap)
            try:
                with httpx.Client(timeout=self.timeout, transport=self.transport, headers=self.headers) as c:
                    r = c.get(NOMINATIM, params={"format": "jsonv2", **params})
            except httpx.HTTPError:
                return []
            finally:
                self._last = time.monotonic()
        if r.status_code != 200:
            return []
        try:
            return r.json() or []
        except ValueError:
            return []

    def locate_city(self, city: str) -> Optional[City]:
        rows = self._get({"q": city, "limit": 1, "addressdetails": 1, "featureType": "city"}) or \
               self._get({"q": city, "limit": 1, "addressdetails": 1})
        if not rows:
            return None
        r = rows[0]
        s, n, w, e = (float(x) for x in r["boundingbox"])
        return City(name=r.get("name") or city, country_code=(r.get("address") or {}).get("country_code", "").upper(),
                    south=s, north=n, west=w, east=e)

    def find_place(self, name: str, city: City, hint: str = "") -> Optional[MapPlace]:
        params = {"q": f"{name} {hint}".strip(), "limit": 5, "addressdetails": 1, "extratags": 1,
                  "viewbox": f"{city.west},{city.north},{city.east},{city.south}", "bounded": 1}
        rows = self._get(params)
        if not rows and hint:
            rows = self._get({**params, "q": name})
        food = [r for r in rows if r.get("type") in FOOD_TYPES or r.get("category") == "amenity"]
        if not food:
            return None
        r = food[0]
        return MapPlace(
            osm_url=f"https://www.openstreetmap.org/{r.get('osm_type', 'node')}/{r.get('osm_id', '')}",
            name=r.get("name") or name, lat=float(r["lat"]), lon=float(r["lon"]), kind=r.get("type", ""),
            address=_short_address(r.get("address") or {}, r.get("display_name", "")),
            tags=r.get("extratags") or {},
        )

    def food_places(self, city: City, words: list[str], limit: int = 40) -> list[MapPlace]:
        """Restaurants mapped inside the city, for when web guides give nothing (e.g. "vegan restaurant")."""
        out: dict[str, MapPlace] = {}
        for w in words:
            rows = self._get({"q": f"{w}", "limit": limit, "addressdetails": 1, "extratags": 1,
                              "viewbox": f"{city.west},{city.north},{city.east},{city.south}", "bounded": 1})
            for r in rows:
                if r.get("type") not in FOOD_TYPES or not r.get("name"):
                    continue
                url = f"https://www.openstreetmap.org/{r.get('osm_type', 'node')}/{r.get('osm_id', '')}"
                out.setdefault(url, MapPlace(osm_url=url, name=r["name"], lat=float(r["lat"]), lon=float(r["lon"]),
                                             kind=r.get("type", ""), tags=r.get("extratags") or {},
                                             address=_short_address(r.get("address") or {}, r.get("display_name", ""))))
        return list(out.values())


def _short_address(a: dict, fallback: str) -> str:
    street = " ".join(x for x in [a.get("road", ""), a.get("house_number", "")] if x)
    parts = [street, a.get("suburb") or a.get("neighbourhood") or a.get("quarter", ""), a.get("city") or a.get("town", "")]
    out = ", ".join(p for p in parts if p)
    return out or ", ".join(fallback.split(", ")[:3])


def diet_flag(tags: dict, diet: str) -> Optional[bool]:
    """True/False if the map says the place does/doesn't suit the diet, None if unknown."""
    cuisine = (tags.get("cuisine") or "").lower()
    if diet == "vegan":
        v = tags.get("diet:vegan")
        if v in ("yes", "only") or "vegan" in cuisine:
            return True
        return False if v == "no" else None
    if diet == "veg":
        v = tags.get("diet:vegetarian")
        if v in ("yes", "only") or tags.get("diet:vegan") in ("yes", "only") or "vegetarian" in cuisine or "vegan" in cuisine:
            return True
        return False if v == "no" else None
    return None

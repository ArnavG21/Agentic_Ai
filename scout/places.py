"""Optional: ask Google Maps itself whether a restaurant is open (Places API, "Text Search").

Google Maps is where people check if a restaurant is "Permanently closed", so this is the most
reliable source. It needs a Google Maps Platform key (GOOGLE_MAPS_API_KEY) on a Google Cloud project
with billing turned on; Google gives a free monthly allowance, and this app makes about 4-8 lookups
per search. Without the key the app uses the free checks in status.py instead.

One request per restaurant: search "<name>, <city>" near the restaurant's map position, then keep
the result only if the name matches and it's within a few hundred metres - so a different
restaurant with a similar name is never mistaken for this one.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

import httpx

PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
FIELDS = ("places.id,places.displayName,places.businessStatus,places.formattedAddress,"
          "places.googleMapsUri,places.location")
STATUS_WORDS = {"OPERATIONAL": "open", "CLOSED_PERMANENTLY": "permanently closed",
                "CLOSED_TEMPORARILY": "temporarily closed"}
GENERIC = {"restaurant", "restaurante", "ristorante", "restaurang", "cafe", "café", "bar", "the", "el", "la",
           "le", "il", "da", "de", "del", "di", "and", "y", "i", "et", "bistro", "trattoria", "osteria", "taberna"}


class PlacesError(RuntimeError):
    pass


@dataclass
class GooglePlace:
    name: str
    status: str            # OPERATIONAL | CLOSED_PERMANENTLY | CLOSED_TEMPORARILY | "" (unknown)
    address: str
    maps_url: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    distance_m: Optional[float] = None

    @property
    def status_words(self) -> str:
        return STATUS_WORDS.get(self.status, "status not given")


def _norm(name: str) -> str:
    name = unicodedata.normalize("NFKD", name.lower())
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    return " ".join(t for t in re.findall(r"[a-z0-9]+", name) if t not in GENERIC)


def same_name(a: str, b: str) -> bool:
    """'Rasoterra' ~ 'Rasoterra Bistró Vegetarià', 'Cal Pep' ~ 'Restaurant Cal Pep', not 'Can Pep'."""
    x, y = _norm(a), _norm(b)
    if not x or not y:
        return False
    if x == y or (len(x) >= 4 and x in y) or (len(y) >= 4 and y in x):
        return True
    need = 0.9 if min(len(x), len(y)) < 10 else 0.82     # one letter matters more in a short name
    return SequenceMatcher(None, x, y).ratio() >= need


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class GooglePlaces:
    def __init__(self, key: str, timeout: float = 15.0, transport: Optional[httpx.BaseTransport] = None,
                 max_distance_m: float = 600):
        self.key = key
        self.timeout = timeout
        self.transport = transport
        self.max_distance_m = max_distance_m
        self.lookups = 0

    def search(self, text: str, lat: Optional[float] = None, lon: Optional[float] = None,
               radius_m: float = 8000) -> list[GooglePlace]:
        """Restaurants Google Maps lists for a text such as "vegetarian restaurant in Lisbon" (open ones only)."""
        body: dict = {"textQuery": text, "pageSize": 20, "languageCode": "en", "includedType": "restaurant"}
        if lat is not None and lon is not None:
            body["locationBias"] = {"circle": {"center": {"latitude": lat, "longitude": lon}, "radius": radius_m}}
        out = []
        for p in self._post(body):
            loc = p.get("location") or {}
            if p.get("businessStatus", "OPERATIONAL") != "OPERATIONAL":
                continue
            out.append(GooglePlace(name=(p.get("displayName") or {}).get("text", ""), status=p.get("businessStatus", ""),
                                   address=p.get("formattedAddress", ""), maps_url=p.get("googleMapsUri", ""),
                                   lat=loc.get("latitude"), lon=loc.get("longitude")))
        return [g for g in out if g.name]

    def _post(self, body: dict) -> list[dict]:
        headers = {"X-Goog-Api-Key": self.key, "X-Goog-FieldMask": FIELDS, "Content-Type": "application/json"}
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as c:
                r = c.post(PLACES_URL, json=body, headers=headers)
        except httpx.HTTPError as e:
            raise PlacesError(f"Google Maps couldn't be reached ({type(e).__name__})") from e
        self.lookups += 1
        if r.status_code in (400, 401, 403):
            try:
                msg = r.json().get("error", {}).get("message", "")
            except ValueError:
                msg = r.text[:200]
            raise PlacesError(f"Google Maps refused the key (HTTP {r.status_code}): {msg[:200]} "
                              "Check that the Places API (New) is enabled and billing is on for the project.")
        if r.status_code == 429:
            raise PlacesError("Google Maps: too many requests right now (the daily limit may be used up)")
        if r.status_code >= 400:
            raise PlacesError(f"Google Maps: HTTP {r.status_code}")
        return r.json().get("places", []) or []

    def find(self, name: str, city: str, lat: Optional[float] = None, lon: Optional[float] = None
             ) -> Optional[GooglePlace]:
        body: dict = {"textQuery": f"{name}, {city}", "pageSize": 5, "languageCode": "en"}
        if lat is not None and lon is not None:
            body["locationBias"] = {"circle": {"center": {"latitude": lat, "longitude": lon}, "radius": 1500.0}}
        places = self._post(body)
        best, best_score = None, -1.0
        for p in places:
            pname = (p.get("displayName") or {}).get("text", "")
            if not same_name(name, pname):
                continue
            loc = p.get("location") or {}
            plat, plon = loc.get("latitude"), loc.get("longitude")
            dist = None
            if lat is not None and lon is not None and plat is not None and plon is not None:
                dist = distance_m(lat, lon, plat, plon)
                if dist > self.max_distance_m:
                    continue
            score = SequenceMatcher(None, _norm(name), _norm(pname)).ratio() - (dist or 0) / 10_000
            if score > best_score:
                best_score = score
                best = GooglePlace(name=pname, status=p.get("businessStatus", "") or "",
                                   address=p.get("formattedAddress", ""), maps_url=p.get("googleMapsUri", ""),
                                   lat=plat, lon=plon, distance_m=dist)
        return best

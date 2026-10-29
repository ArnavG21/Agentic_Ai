"""Links shown on every suggestion: only real pages, never a search page dressed up as one.

- Website: the restaurant's own site, when the agent found it.
- Menu: the page (or PDF) the menu was actually read from. When the menu was read from several pages
  (lunch, dinner, desserts), the best one is the button and the rest are listed under the menu.
- Google Maps: always - the place itself when Google Maps found it, otherwise its name and address on
  Google Maps, which is where travellers check opening hours and directions.
If there is no website or no readable menu online, the button is simply left out and the card says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote_plus

from .models import Dossier


@dataclass
class Link:
    label: str
    url: str
    found: bool = True


def website_link(d: Dossier, city: str) -> Link | None:
    return Link("Restaurant website", d.candidate.website) if d.candidate.website else None


def menu_link(d: Dossier, city: str) -> Link | None:
    return Link("Original menu", d.menu.menu_url) if d.menu.menu_url else None


def maps_link(d: Dossier, city: str) -> Link:
    c = d.candidate
    return Link("Google Maps", c.maps_url or ("https://www.google.com/maps/search/?api=1&query="
                                               + quote_plus(f"{c.name} {c.address or city}")))


def all_links(d: Dossier, city: str) -> list[Link]:
    return [ln for ln in (website_link(d, city), menu_link(d, city), maps_link(d, city)) if ln]

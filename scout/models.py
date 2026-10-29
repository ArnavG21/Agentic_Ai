"""Data models passed between pipeline stages.

Every stage reads and writes these typed objects, so each step's input and
output can be shown in the UI and saved to a file for replay.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

MealType = Literal["breakfast", "brunch", "lunch", "dinner", "any"]
Diet = Literal["veg", "vegan", "non_veg", "any"]


class UserQuery(BaseModel):
    """What the user wants, normalised by the 'understand' stage."""

    city: str
    country_code: str = ""             # ISO-2, e.g. "ES"
    meal_type: MealType = "any"
    diet: Diet = "any"
    budget_min: Optional[float] = None    # per person, local currency
    budget_max: Optional[float] = None    # per person, local currency
    budget_estimated: bool = False        # True when the traveller used words ("cheap"), not numbers
    budget_note: str = ""                 # the words they used, e.g. "cheap"
    currency: str = ""                    # local currency, e.g. "EUR"
    cuisines: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)   # "kid-friendly", "near Sagrada Familia"...
    local_language: str = ""
    local_menu_words: list[str] = Field(default_factory=list)   # "carta", "メニュー"... for finding menu pages
    menu_language: str = "English"        # language menus are translated into
    review_keywords: list[str] = Field(default_factory=list)  # English + local-language words to scan reviews for
    search_queries: list[str] = Field(default_factory=list)   # web searches that find restaurant guides
    raw_request: str = ""


class Mention(BaseModel):
    url: str
    title: str = ""
    why: str = ""                      # what the guide says about the place


class Candidate(BaseModel):
    """A restaurant found in the discovery stage."""

    place_id: str
    name: str
    address: str = ""
    area_hint: str = ""                # neighbourhood mentioned by guides
    lat: Optional[float] = None
    lon: Optional[float] = None
    on_map: bool = False               # found on OpenStreetMap inside the city
    website: str = ""
    maps_url: str = ""                 # Google Maps search link (free, no API)
    osm_url: str = ""
    cuisine: str = ""
    opening_hours: str = ""
    tags: dict = Field(default_factory=dict)   # raw OpenStreetMap tags
    rating_hint: Optional[float] = None        # a rating out of 5 quoted by a guide, if any
    guide_price: Optional[float] = None        # price per person a guide quotes, if any (local currency)
    mentions: list[Mention] = Field(default_factory=list)


class ShortlistRow(BaseModel):
    """One candidate's checks in the shortlist step - kept for every candidate, not just the chosen ones."""

    place_id: str
    name: str
    passed: bool                       # passed every yes/no check
    guides: int = 0                    # how many food guides recommend it
    chosen: bool = False               # looked at in depth (menu, reviews)
    closed: bool = False               # found to have closed for good
    closed_source: str = ""            # where that was found
    closed_temporarily: bool = False   # closed for now (holidays, works), not for good
    open_confirmed: bool = False       # Google Maps lists it as open
    open_checks: list[str] = Field(default_factory=list)   # which "still open?" checks ran
    reasons: list[str] = Field(default_factory=list)


class MenuItem(BaseModel):
    original_name: str
    english_name: str
    description: str = ""
    section: str = ""
    price: Optional[float] = None
    currency: str = ""
    diet: str = "unknown"              # veg | vegan | non_veg | unknown
    popular: bool = False              # mentioned in reviews


class MenuResult(BaseModel):
    menu_url: str = ""
    menu_urls: list[str] = Field(default_factory=list)   # every page the menu was read from (lunch, dinner, PDF...)
    translated_to: str = ""            # language the dishes were translated into
    how_found: str = ""                # website_link | website_home | pdf | web_search | none
    pages_tried: list[str] = Field(default_factory=list)
    detected_language: str = ""
    raw_excerpt: str = ""
    items: list[MenuItem] = Field(default_factory=list)
    notes: str = ""


class Finding(BaseModel):
    point: str
    source_url: str = ""


class ContextFinding(BaseModel):
    requirement: str
    verdict: str                       # supported | contradicted | mixed | no_evidence
    evidence: str = ""
    source_url: str = ""


class ReviewInsights(BaseModel):
    closed_note: str = ""                                   # set if reviews say the place closed for good
    closed_source: str = ""
    queries: list[str] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)       # {n, url, title, snippet}
    keyword_hits: list[dict] = Field(default_factory=list)  # {keyword, snippet, url}
    positives: list[Finding] = Field(default_factory=list)
    negatives: list[Finding] = Field(default_factory=list)
    context: list[ContextFinding] = Field(default_factory=list)
    popular_dishes: list[str] = Field(default_factory=list)


class CostEstimate(BaseModel):
    """Estimated spend per person, worked out in code from the menu (see money.py)."""

    low: float
    high: float
    currency: str = ""
    source: str = "menu"               # menu | guide
    basis: str = ""                    # the calculation, in words
    fit: str = "no_budget"             # fits | partly | over | under | no_budget
    verdict: str = ""                  # the comparison with the budget, in words


class Dossier(BaseModel):
    """Everything the agent learned about one shortlisted restaurant."""

    candidate: Candidate
    shortlist: ShortlistRow
    menu: MenuResult = Field(default_factory=MenuResult)
    reviews: ReviewInsights = Field(default_factory=ReviewInsights)
    cost: Optional[CostEstimate] = None


class OrderSuggestion(BaseModel):
    item: str
    price: Optional[float] = None
    currency: str = ""
    why: str = ""


class NeedNote(BaseModel):
    need: str                          # one of the traveller's needs, e.g. "Quiet", "Vegetarian", "Under 35 €"
    insight: str                       # what we found about it, in plain words, with a tip when unsure


class Pick(BaseModel):
    """The write-up of one suggested restaurant. No score and no rank: just the evidence, summarised."""

    place_id: str
    name: str
    headline: str
    why_it_fits: list[str]
    watch_outs: list[str]
    what_to_order: list[OrderSuggestion]
    for_you: list[NeedNote] = Field(default_factory=list)
    good_to_know: list[str] = Field(default_factory=list)
    estimated_spend: str = ""


class Recommendation(BaseModel):
    summary: str
    picks: list[Pick]

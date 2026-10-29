"""JSON schemas for Claude's structured outputs.

Structured outputs require every object to set additionalProperties: false.
We also mark every property as required (nullable where a value may be
missing) so the output shape is always identical and easy to validate.
"""
from __future__ import annotations


def obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def arr(items: dict) -> dict:
    return {"type": "array", "items": items}


STR = {"type": "string"}
STRS = arr(STR)
NUM_OR_NULL = {"type": ["number", "null"]}

USER_QUERY = obj({
    "on_topic": {"type": "boolean", "description": "true only if the message asks for help finding or choosing "
                 "somewhere to eat or drink (restaurant, café, bar, bakery, food market) or about food/menus for that"},
    "city": STR,
    "country_code": {"type": "string", "description": "ISO 3166-1 alpha-2 code, e.g. ES"},
    "meal_type": {"type": "string", "enum": ["breakfast", "brunch", "lunch", "dinner", "any"]},
    "diet": {"type": "string", "enum": ["veg", "vegan", "non_veg", "any"]},
    "budget_min": {**NUM_OR_NULL, "description": "Lowest spend per person in local currency, or null"},
    "budget_max": {**NUM_OR_NULL, "description": "Highest spend per person in local currency, or null"},
    "budget_estimated": {"type": "boolean", "description": "true if you converted words like 'cheap' into numbers"},
    "budget_note": {"type": "string", "description": "The traveller's own budget words, e.g. 'cheap'; else empty"},
    "currency": {"type": "string", "description": "Local currency code of the city, e.g. EUR"},
    "cuisines": STRS,
    "requirements": {**STRS, "description": "Other needs, each as a short phrase"},
    "local_language": STR,
    "local_menu_words": {**STRS, "description": "How 'menu' is written locally, e.g. carta / メニュー; empty if English"},
    "review_keywords": {**STRS, "description": "Words to scan reviews for, in English AND the local language"},
    "search_queries": {**STRS, "description": "3 web searches that find restaurant guides and lists"},
})

GUIDE_EXTRACT = obj({
    "restaurants": arr(obj({
        "name": {"type": "string", "description": "Restaurant name exactly as written"},
        "area": {"type": "string", "description": "Neighbourhood or street if the source says; else empty"},
        "rating": {**NUM_OR_NULL, "description": "Rating out of 5 if a source states one; else null"},
        "price_per_person": {**NUM_OR_NULL, "description": "Typical spend per person in local currency if a "
                             "source states one (e.g. 'about €25 a head', 'set menu €14'); else null"},
        "why": {"type": "string", "description": "One sentence: what the source says about it"},
        "source_numbers": {"type": "array", "items": {"type": "integer"},
                           "description": "Numbers of the sources that mention it"},
    }))
})

MENU = obj({
    "detected_language": STR,
    "currency": STR,
    "items": arr(obj({
        "original_name": STR,
        "english_name": {"type": "string", "description": "The dish name translated into the requested language"},
        "description": {"type": "string", "description": "Short description in the requested language; empty if none"},
        "section": {"type": "string", "description": "Menu section in the requested language, e.g. Starters, Mains"},
        "price": NUM_OR_NULL,
        "diet": {"type": "string", "enum": ["veg", "vegan", "non_veg", "unknown"]},
    })),
    "notes": STR,
})

_FINDING = obj({"point": STR, "source_url": STR})

REVIEW_BALANCE = obj({
    "positives": arr(_FINDING),
    "negatives": arr(_FINDING),
    "popular_dishes": STRS,
    "closed_for_good": {"type": "boolean", "description": "true only if a source clearly says this restaurant has "
                                                         "permanently closed (not holidays or weekly closing days)"},
    "closure_source_url": STR,
})

CONTEXT_CHECK = obj({
    "context": arr(obj({
        "requirement": STR,
        "verdict": {"type": "string", "enum": ["supported", "contradicted", "mixed", "no_evidence"]},
        "evidence": STR,
        "source_url": STR,
    })),
})

RECOMMENDATION = obj({
    "summary": STR,
    "picks": arr(obj({
        "place_id": STR,
        "name": STR,
        "headline": STR,
        "why_it_fits": STRS,
        "for_you": arr(obj({"need": STR, "insight": STR})),
        "good_to_know": STRS,
        "watch_outs": STRS,
        "what_to_order": arr(obj({"item": STR, "why": STR})),
        "estimated_spend": STR,
    })),
})

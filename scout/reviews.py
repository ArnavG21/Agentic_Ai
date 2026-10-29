"""Stages 6, 7 & 8 - reviews and the dishes worth ordering.

Stage 6 (positive & negative): web search for reviews of the restaurant,
  including a search aimed at complaints (most top results are positive),
  open the readable pages, and have the AI separate praise from complaints -
  each point tied to the page it came from.
Stage 7 (your context): a keyword scan of all review text for the traveller's
  words in English and the local language (no AI), plus searches about their
  exact needs ("vegetarian", "kid-friendly", "near Sagrada Familia"...). The AI
  gives each need a verdict: supported / contradicted / mixed / no_evidence.
Stage 8 (regular menu items): pure Python - filters the translated menu to the
  traveller's diet and marks dishes reviewers mention as popular.

Any source URL the AI returns that the agent never actually opened is blanked.
"""
from __future__ import annotations

import re
import statistics
from difflib import SequenceMatcher

from .llm import LLM
from .models import Candidate, ContextFinding, Finding, MenuItem, MenuResult, ReviewInsights, UserQuery
from .schemas import CONTEXT_CHECK, REVIEW_BALANCE
from .money import budget_text, fmt_price  # noqa: F401  (fmt_price re-exported)
from .search import Evidence, WebSearch

DIET_LABEL = {"veg": "vegetarian options", "vegan": "vegan options", "non_veg": "meat and fish dishes"}


def context_topics(q: UserQuery) -> list[str]:
    topics = []
    if q.diet in DIET_LABEL:
        topics.append(DIET_LABEL[q.diet])
    if q.meal_type != "any":
        topics.append(f"{q.meal_type} experience")
    if q.budget_max is not None or q.budget_min is not None:
        topics.append(f"prices and value for money ({budget_text(q)})")
    topics.extend(q.cuisines)
    topics.extend(q.requirements)
    return topics or ["overall experience"]


def _record(ins: ReviewInsights, ev: Evidence) -> None:
    ins.queries += ev.queries
    known = {s["url"] for s in ins.sources}
    ins.sources += [{"n": s["n"], "url": s["url"], "title": s["title"], "snippet": s["snippet"]}
                    for s in ev.sources if s["url"] not in known]


# ---------------------------------------------------------------- stage 6
BALANCE_SYSTEM = """You summarise what diners say about ONE restaurant, from the numbered sources given.
Rules:
- Use only the sources. Ignore sources about a different restaurant with a similar name.
- positives / negatives: up to 5 distinct points each, in your own words, most common first.
  If the sources contain no complaints, return an empty negatives list - never invent one.
- source_url: copy the URL of the source that supports the point exactly; "" if unsure.
- popular_dishes: dishes reviewers recommend by name, as written.
- closed_for_good: true only if a source clearly says THIS restaurant has closed permanently; give that
  source's URL in closure_source_url. Otherwise false and ""."""


def gather_general(c: Candidate, q: UserQuery, web: WebSearch, pages: int = 3) -> Evidence:
    return web.research([f"{c.name} {q.city} reviews", f"{c.name} {q.city} review disappointing overpriced closed"],
                        pages=pages, max_sources=10)


def summarise_balance(c: Candidate, ev: Evidence, llm: LLM, ins: ReviewInsights) -> ReviewInsights:
    _record(ins, ev)
    if not ev.sources:
        return ins
    data = llm.structured(system=BALANCE_SYSTEM, name="review_balance", schema=REVIEW_BALANCE,
                          prompt=f"Restaurant: {c.name}, {c.address or c.area_hint}\n\nSources:\n{ev.as_prompt()}")
    ok = ev.urls | {m.url for m in c.mentions}
    ins.positives = [Finding(point=f["point"], source_url=f["source_url"] if f["source_url"] in ok else "")
                     for f in data["positives"]]
    ins.negatives = [Finding(point=f["point"], source_url=f["source_url"] if f["source_url"] in ok else "")
                     for f in data["negatives"]]
    ins.popular_dishes = data["popular_dishes"]
    if data.get("closed_for_good") and data.get("closure_source_url") in ok:
        ins.closed_note = "A review source says it has closed for good."
        ins.closed_source = data["closure_source_url"]
    return ins


# ---------------------------------------------------------------- stage 7
CONTEXT_SYSTEM = """You check ONE restaurant against a traveller's needs using the numbered sources.
For each need, give a verdict: "supported" (sources say it fits), "contradicted" (sources say it
doesn't), "mixed", or "no_evidence" (sources don't say). evidence: one sentence in your own words.
source_url: the exact URL of the supporting source, or "". Use only the sources."""


def keyword_hits(ev: Evidence, keywords: list[str], window: int = 90) -> list[dict]:
    out, seen = [], set()
    for s in ev.sources:
        text = s["text"] or s["snippet"]
        low = text.lower()
        for k in keywords:
            i = low.find(k.lower()) if k else -1
            if i < 0 or (k.lower(), s["url"]) in seen:
                continue
            seen.add((k.lower(), s["url"]))
            a, b = max(0, i - window), min(len(text), i + len(k) + window)
            out.append({"keyword": k, "url": s["url"],
                        "snippet": ("…" if a else "") + text[a:b].replace("\n", " ") + ("…" if b < len(text) else "")})
    return out


def gather_context(c: Candidate, q: UserQuery, web: WebSearch, pages: int = 2) -> Evidence:
    topics = context_topics(q)[:2]
    queries = [f"{c.name} {q.city} {t}" for t in topics]
    return web.research(queries, pages=pages, max_sources=8)


def check_context(c: Candidate, q: UserQuery, ev_ctx: Evidence, ev_general: Evidence, llm: LLM,
                  ins: ReviewInsights) -> ReviewInsights:
    _record(ins, ev_ctx)
    merged = Evidence(sources=[{**s, "n": i} for i, s in enumerate(ev_general.sources + ev_ctx.sources, 1)])
    ins.keyword_hits = keyword_hits(merged, q.review_keywords)
    needs = context_topics(q)
    if not merged.sources:
        ins.context = [ContextFinding(requirement=t, verdict="no_evidence") for t in needs]
        return ins
    data = llm.structured(
        system=CONTEXT_SYSTEM, name="context_check", schema=CONTEXT_CHECK,
        prompt=f"Restaurant: {c.name}, {c.address or c.area_hint}\nTraveller needs: {'; '.join(needs)}\n\n"
               f"Sources:\n{merged.as_prompt(2000)}",
    )
    ok = merged.urls | {m.url for m in c.mentions}
    ins.context = [ContextFinding(**{**f, "source_url": f["source_url"] if f["source_url"] in ok else ""})
                   for f in data["context"]]
    return ins


# ---------------------------------------------------------------- stage 8
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9à-ÿ ]+", " ", s.lower()).strip()


def is_mentioned(item: MenuItem, dishes: list[str], texts: list[str]) -> bool:
    names = [_norm(item.original_name), _norm(item.english_name)]
    for d in dishes:
        nd = _norm(d)
        for n in names:
            if n and nd and (n in nd or nd in n or SequenceMatcher(None, n, nd).ratio() >= 0.8):
                return True
    blob = _norm(" ".join(texts))
    return any(len(n) >= 6 and n in blob for n in names)


def regular_items(menu: MenuResult, insights: ReviewInsights, diet: str, limit: int = 12) -> list[MenuItem]:
    texts = [s.get("snippet", "") for s in insights.sources] + [f.point for f in insights.positives]
    for it in menu.items:
        it.popular = is_mentioned(it, insights.popular_dishes, texts)
    allowed = {"veg": {"veg", "vegan"}, "vegan": {"vegan"}}.get(diet, {"veg", "vegan", "non_veg", "unknown"})
    pool = [it for it in menu.items if it.diet in allowed] or [it for it in menu.items if it.diet == "unknown"]
    pool.sort(key=lambda it: (not it.popular, it.price is None))
    return pool[:limit]


def price_summary(items: list[MenuItem]) -> str:
    prices = [it.price for it in items if it.price]
    if not prices:
        return "No prices found"
    cur = next((it.currency for it in items if it.currency), "")
    return (f"{len(prices)} priced dishes, {fmt_price(min(prices), cur)} to {fmt_price(max(prices), cur)}, "
            f"median {fmt_price(statistics.median(prices), cur)}")

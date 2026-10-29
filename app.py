"""Restaurant Scout - Streamlit chat app.

Run:  python -m streamlit run app.py      (or deploy free on Streamlit Community Cloud)

A search form (city, diet, meal, budget bracket and one box for anything else) starts
a search and the nine agent steps stream live; the suggestions then appear as cards with the
restaurant's website, its menu (link + translated dishes) and a map link. Later
messages are follow-up questions answered from the evidence the agent collected.
"""
from __future__ import annotations

import datetime as dt
import html
import os
import re
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

from scout.config import Settings
from scout.health import check_all
from scout.links import all_links
from scout.models import Dossier, Pick
from scout.pipeline import STAGES, Agent, RunState, load_run, save_run
from scout.recommend import NEED_WORDS, answer_followup
from scout.pdf_report import build_history_pdf, build_pdf
from scout.sources import source_rows, summarise, to_dot
from scout.money import budget_text, fmt_price, fmt_range
from scout.reviews import price_summary, regular_items

st.set_page_config(page_title="Restaurant Scout · Europe", page_icon="🍽️", layout="wide")

CITIES = {  # city -> flag, for the city picker (any other European city can be typed in)
    "Amsterdam": "🇳🇱", "Athens": "🇬🇷", "Barcelona": "🇪🇸", "Berlin": "🇩🇪", "Bologna": "🇮🇹", "Brussels": "🇧🇪",
    "Budapest": "🇭🇺", "Copenhagen": "🇩🇰", "Dublin": "🇮🇪", "Dubrovnik": "🇭🇷", "Edinburgh": "🇬🇧", "Florence": "🇮🇹",
    "Helsinki": "🇫🇮", "Istanbul": "🇹🇷", "Kraków": "🇵🇱", "Lisbon": "🇵🇹", "Ljubljana": "🇸🇮", "London": "🇬🇧",
    "Lyon": "🇫🇷", "Madrid": "🇪🇸", "Milan": "🇮🇹", "Munich": "🇩🇪", "Naples": "🇮🇹", "Nice": "🇫🇷", "Oslo": "🇳🇴",
    "Paris": "🇫🇷", "Porto": "🇵🇹", "Prague": "🇨🇿", "Reykjavík": "🇮🇸", "Rome": "🇮🇹", "Salzburg": "🇦🇹",
    "San Sebastián": "🇪🇸", "Seville": "🇪🇸", "Stockholm": "🇸🇪", "Tallinn": "🇪🇪", "Valencia": "🇪🇸", "Venice": "🇮🇹",
    "Vienna": "🇦🇹", "Warsaw": "🇵🇱", "Zürich": "🇨🇭",
}
OTHER_CITY = "✏️ Another European city…"
DIETS = {"Any": ("any", ""), "Vegetarian": ("veg", "vegetarian"), "Vegan": ("vegan", "vegan"),
         "Non-veg": ("non_veg", "non-vegetarian (meat or fish)")}
MEALS = {"Breakfast": "breakfast", "Brunch": "brunch", "Lunch": "lunch", "Dinner": "dinner", "Any": "any"}
BUDGET_BANDS = [("Any", None, None), ("Budget", None, 15), ("Mid-range", 15, 30), ("Comfortable", 30, 50),
                ("Treat", 50, 80), ("Splurge", 80, None)]          # per person, in euros
CITY_CURRENCY = {"London": "GBP", "Edinburgh": "GBP", "Prague": "CZK", "Budapest": "HUF", "Kraków": "PLN",
                 "Warsaw": "PLN", "Copenhagen": "DKK", "Stockholm": "SEK", "Oslo": "NOK", "Reykjavík": "ISK",
                 "Zürich": "CHF", "Istanbul": "TRY"}
EUR_RATE = {"EUR": 1, "GBP": 0.85, "CZK": 25, "HUF": 400, "PLN": 4.3, "DKK": 7.45, "SEK": 11.5, "NOK": 11.7,
            "ISK": 145, "CHF": 0.95, "TRY": 45}    # rough rates, only to label the budget buttons sensibly


def _nice(x: float) -> int:
    """Round a converted amount to a number people actually use: 12 -> 12, 374 -> 375, 6000 -> 6,000."""
    step = 1 if x < 30 else 5 if x < 300 else 50 if x < 3000 else 500
    return int(round(x / step) * step)


def budget_options(currency: str) -> dict[str, tuple]:
    """Label -> (min, max, currency) for the city's own currency, e.g. 'Mid-range · 375–750 Kč'."""
    rate = EUR_RATE.get(currency, 1)
    out = {}
    for name, lo, hi in BUDGET_BANDS:
        lo_c = _nice(lo * rate) if lo else None
        hi_c = _nice(hi * rate) if hi else None
        if name == "Any":
            out["Any budget"] = (None, None, currency)
            continue
        amount = (f"under {fmt_price(hi_c, currency)}" if lo_c is None else
                  f"{fmt_price(lo_c, currency)}+" if hi_c is None else
                  f"{fmt_range(lo_c, hi_c, currency)}")
        out[f"{name} · {amount}"] = (lo_c, hi_c, currency)
    return out
MENU_LANGUAGES = ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Dutch", "Polish",
                  "Greek", "Russian"]
VERDICT = {"supported": ("reviews confirm it", "✓"), "contradicted": ("reviews say otherwise", "✗"),
           "mixed": ("mixed reports", "≈"), "no_evidence": ("not mentioned in reviews", "·")}
DIET_LABEL = {"veg": "veg", "vegan": "vegan", "non_veg": "meat/fish", "unknown": "not sure"}
STAGE_INFO = {k: (i, t, d) for i, (k, t, d) in enumerate(STAGES, 1)}
FIT_CLASS = {"fits": "", "under": "", "partly": "mixed", "over": "no", "no_budget": "unknown"}
FIT_WORD = {"fits": "within budget", "under": "under budget", "partly": "partly within budget", "over": "over budget",
            "no_budget": ""}

esc = html.escape

# ---------------------------------------------------------------------- styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Figtree:wght@400;500;600&display=swap');
:root {
  --olive:#2E5339; --olive-soft:#E3EBE1; --saffron:#D99A1E; --saffron-soft:#F8EBCB;
  --ink:#1D2320; --stone:#6C736D; --tomato:#B5432F; --tomato-soft:#F5DFDA;
}
html, body, [class*="css"], .stMarkdown, .stTextInput, .stSelectbox, button { font-family:'Figtree', sans-serif; }
h1, h2, h3, h4, .rs-name, .rs-hero h1 { font-family:'Bricolage Grotesque', sans-serif !important; letter-spacing:-0.01em; }
.block-container { max-width: 980px; padding-top: 2.2rem; }
.rs-hero h1 { font-size: clamp(2.1rem, 4.6vw, 3.3rem); line-height:1.05; font-weight:700; margin:0 0 .6rem; color:var(--ink); }
.rs-hero p { font-size:1.08rem; color:var(--stone); max-width:62ch; margin:0 0 1.2rem; line-height:1.55; }
.rs-line { display:flex; align-items:baseline; gap:.5rem; max-width:34rem; margin:.5rem 0 .2rem;
  font-family:'Bricolage Grotesque', sans-serif; font-size:1.25rem; color:var(--olive); }
.rs-leader { flex:1; border-bottom:2px dotted var(--saffron); transform:translateY(-4px); min-width:1.5rem; }
.rs-card-head { display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; }
.rs-name { font-size:1.65rem; font-weight:700; line-height:1.15; margin:.1rem 0 .3rem; color:var(--ink); }
.rs-headline { font-size:1.05rem; color:var(--ink); margin:.2rem 0 .7rem; max-width:70ch; }
.rs-chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.3rem 0 .8rem; }
.rs-chip { background:var(--olive-soft); color:var(--olive); border-radius:999px; padding:.2rem .65rem; font-size:.85rem; font-weight:500; }
.rs-chip.no { background:var(--tomato-soft); color:var(--tomato); }
.rs-chip.mixed { background:var(--saffron-soft); color:#8A5D05; }
.rs-chip.unknown { background:#ECEDEA; color:var(--stone); }
.rs-links { font-size:.88rem; color:var(--stone); margin:.45rem 0 0; line-height:1.6; word-break:break-all; }
.rs-links a { color:var(--olive); }
.rs-sub { font-family:'Bricolage Grotesque', sans-serif; font-weight:700; font-size:1.02rem; margin:1rem 0 .35rem; color:var(--ink); }
.rs-list { margin:0; padding-left:1.1rem; line-height:1.55; }
.rs-list li { margin-bottom:.2rem; }
.rs-dish { display:flex; align-items:baseline; gap:.5rem; margin-top:.55rem; }
.rs-dish .n { font-weight:600; }
.rs-dish .p { font-weight:600; color:var(--olive); white-space:nowrap; }
.rs-why { color:var(--stone); font-size:.9rem; margin:.05rem 0 0; }
.rs-spend { color:var(--stone); font-size:.92rem; margin-top:.6rem; }
.rs-summary { border-left:4px solid var(--saffron); background:#fff; padding:.8rem 1rem; border-radius:0 6px 6px 0; margin:.4rem 0 1.2rem; line-height:1.55; }
.rs-step { color:var(--stone); font-size:.92rem; margin:-.2rem 0 .5rem; }
.rs-need { display:flex; gap:.8rem; padding:.45rem 0; border-bottom:1px solid #E6E9E3; line-height:1.5; }
.rs-need:last-child { border-bottom:none; }
.rs-need-k { flex:0 0 8.5rem; font-weight:600; color:var(--olive); font-size:.92rem; }
.rs-need-v { flex:1; font-size:.95rem; }
@media (max-width:640px) { .rs-need { flex-direction:column; gap:.1rem; } .rs-need-k { flex:none; } }
.rs-menucard { background:#FFFDF8; border:1px solid #E4DCC8; border-radius:6px; padding:1.4rem 1.6rem 1rem;
  max-width:44rem; margin:.3rem auto .6rem; box-shadow:0 6px 20px -16px rgba(29,35,32,.5); }
.rs-menucard-head { text-align:center; border-bottom:2px solid var(--saffron); padding-bottom:.7rem; margin-bottom:.4rem; }
.rs-menucard-name { font-family:'Bricolage Grotesque', sans-serif; font-weight:700; font-size:1.5rem; color:var(--ink); }
.rs-menucard-sub { color:var(--stone); font-size:.82rem; margin-top:.15rem; }
.rs-menucard-sec { font-family:'Bricolage Grotesque', sans-serif; font-weight:700; letter-spacing:.12em;
  text-transform:uppercase; font-size:.85rem; color:var(--olive); text-align:center; margin:1.1rem 0 .35rem; }
.rs-mi { margin:.45rem 0; }
.rs-mi-row { display:flex; align-items:baseline; gap:.45rem; }
.rs-mi-name { font-weight:600; color:var(--ink); }
.rs-mi-price { font-weight:600; color:var(--olive); white-space:nowrap; }
.rs-mi-orig { font-style:italic; color:var(--stone); font-size:.85rem; }
.rs-mi-desc { color:#4A514C; font-size:.88rem; line-height:1.45; }
.rs-mi-diet { display:inline-block; font-size:.68rem; font-weight:700; color:var(--olive); border:1px solid var(--olive);
  border-radius:3px; padding:0 .25rem; margin-left:.2rem; vertical-align:2px; }
.rs-mi-pop { font-size:.72rem; color:#8A5D05; background:var(--saffron-soft); border-radius:999px; padding:.05rem .45rem;
  margin-left:.25rem; vertical-align:2px; font-weight:500; }
.rs-hero { border-top:10px solid var(--olive); padding-top:1.3rem; margin-top:2.6rem;
  border-image:repeating-linear-gradient(90deg, var(--olive) 0 18px, var(--olive-soft) 18px 36px, var(--saffron) 36px 54px, var(--olive-soft) 54px 72px) 10; }
.rs-eyebrow { font-size:.8rem; font-weight:600; letter-spacing:.14em; text-transform:uppercase; color:var(--saffron); margin:0 0 .5rem; }
.rs-menu { background:#fff; border:1px solid #D5DBD2; border-radius:6px; padding:1rem 1.3rem .8rem; max-width:36rem; margin:.4rem 0 1.2rem;
  box-shadow:0 1px 0 #D5DBD2, 0 6px 18px -12px rgba(29,35,32,.35); }
.rs-menu .rs-menu-title { font-family:'Bricolage Grotesque', sans-serif; font-weight:700; font-size:.95rem; color:var(--stone);
  letter-spacing:.08em; text-transform:uppercase; margin:0 0 .2rem; }
.rs-menu .rs-line { margin:.45rem 0; font-size:1.15rem; }
.rs-menu .rs-line small { color:var(--stone); font-family:'Figtree', sans-serif; font-size:.85rem; }
.rs-cities { color:var(--stone); font-size:.92rem; line-height:1.7; margin:.2rem 0 1.4rem; max-width:62ch; }
.rs-cities b { color:var(--olive); font-weight:600; }
[class*="st-key-ex"] button { justify-content:flex-start; text-align:left; min-height:3.6rem; background:#fff; }
[class*="st-key-ex"] button > div, [class*="st-key-ex"] button div { justify-content:flex-start !important; width:100%; }
[class*="st-key-ex"] button p { text-align:left; line-height:1.4; white-space:normal !important; overflow:visible !important;
  text-overflow:clip !important; }
[class*="st-key-ex"] button:hover { border-color:var(--saffron); color:var(--ink); }
@media (prefers-reduced-motion: reduce) { * { transition:none !important; animation:none !important; } }
</style>
""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("messages", [])     # [{role, content}] ; "__RUN__" marks where the results render
ss.setdefault("run", None)        # RunState of the finished search
ss.setdefault("pending", None)    # (request, fixed choices) submitted from the search form
ss.setdefault("notice", "")       # a message for the form, e.g. "that city isn't in Europe"
ss.setdefault("health", None)     # results of "Check services"


# ===================================================================== helpers
def money(v, cur) -> str:
    return fmt_price(v, cur)


def chip(text: str, kind: str = "") -> str:
    return f'<span class="rs-chip {kind}">{esc(text)}</span>'


def src(url: str) -> str:
    return f' <a href="{esc(url)}" target="_blank">source</a>' if url else ""


def table(rows: list[dict], **kw) -> None:
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", **kw)


def _findings(items, empty: str) -> str:
    if not items:
        return f'<p class="rs-why">{esc(empty)}</p>'
    return '<ul class="rs-list">' + "".join(f"<li>{esc(f.point)}{src(f.source_url)}</li>" for f in items) + "</ul>"


# ================================================================ step details
def stage_details(stage: str, s: RunState) -> None:
    q = s.query
    if stage == "understand" and q:
        st.json(q.model_dump(exclude={"raw_request"}), expanded=False)

    elif stage == "discover":
        if s.guide_sources:
            st.markdown("**Pages searched**")
            table([{"Read": ("in full" + (" (Jina)" if x.get("via") == "jina" else "")) if x["read"] else "preview only",
                    "Page": x["title"], "Link": x["url"]} for x in s.guide_sources],
                  column_config={"Link": st.column_config.LinkColumn(display_text="open")})
        st.markdown("**Restaurants the guides recommend**")
        table([{"Name": c.name, "Guides": len(c.mentions), "On map": "yes" if c.on_map else "no",
                "Address": c.address or c.area_hint, "Cuisine": c.cuisine, "Hours": c.opening_hours,
                "Website": c.website or None, "Map": c.maps_url or None} for c in s.candidates],
              column_config={"Website": st.column_config.LinkColumn(),
                             "Map": st.column_config.LinkColumn(display_text="map")})

    elif stage == "shortlist":
        st.caption("Each restaurant gets simple yes/no checks. ✗ means the evidence rules it out, ? means not known "
                   "yet (checked later), ✓ means confirmed. Of those that pass, the ones recommended by the most "
                   "food guides are checked to make sure they're still open (Google Maps if you added a key, their website, "
                   "review listings and closure news) and then looked at in depth.")
        table([{"Result": "looked at in depth" if r.chosen else (("temporarily closed" if r.closed_temporarily
                                                                  else "closed for good") if r.closed else
                                                                 ("passed" if r.passed else "left out")),
                "Name": r.name, "Food guides": r.guides, "Checks": "; ".join(r.reasons)} for r in s.shortlist_rows])

    elif stage == "menu":
        for d in s.dossiers:
            m = d.menu
            with st.container(border=True):
                found = m.how_found != "none"
                st.markdown(f"**{esc(d.candidate.name)}**: "
                            + (f"menu found on the {m.how_found.replace('+jina', ', read with Jina Reader')}"
                               if found else "no readable menu"))
                if m.menu_url:
                    st.markdown(f"Menu: [{m.menu_url}]({m.menu_url})")
                if m.notes:
                    st.caption(m.notes)
                if m.raw_excerpt:
                    with st.popover("What the agent read"):
                        st.code(m.raw_excerpt, language=None)

    elif stage == "translate":
        for d in s.dossiers:
            m = d.menu
            if not m.items:
                st.markdown(f"**{esc(d.candidate.name)}**: nothing to translate")
                continue
            st.markdown(f"**{esc(d.candidate.name)}**: {esc(m.detected_language or 'unknown')} to English. "
                        f"{price_summary(m.items)}")
            table([{"Original": it.original_name, "English": it.english_name, "Section": it.section,
                    "Price": money(it.price, it.currency), "Diet": DIET_LABEL.get(it.diet, it.diet),
                    "Description": it.description} for it in m.items],
                  height=min(38 + 35 * len(m.items), 320))

    elif stage == "reviews":
        for d in s.dossiers:
            rv = d.reviews
            with st.container(border=True):
                st.markdown(f"**{esc(d.candidate.name)}**")
                a, b = st.columns(2)
                a.markdown('<div class="rs-sub">What people like</div>' + _findings(rv.positives, "Nothing found."),
                           unsafe_allow_html=True)
                b.markdown('<div class="rs-sub">What people complain about</div>'
                           + _findings(rv.negatives, "No recurring complaints found."), unsafe_allow_html=True)
                if rv.queries:
                    st.caption("Searches: " + "; ".join(f'"{x}"' for x in rv.queries))

    elif stage == "context":
        for d in s.dossiers:
            rv = d.reviews
            with st.container(border=True):
                st.markdown(f"**{esc(d.candidate.name)}**")
                table([{"Your need": f.requirement, "What reviews say": VERDICT.get(f.verdict, (f.verdict, ""))[0],
                        "Details": f.evidence, "Source": f.source_url or None} for f in rv.context],
                      column_config={"Source": st.column_config.LinkColumn(display_text="open")})
                if rv.keyword_hits:
                    with st.popover(f"{len(rv.keyword_hits)} keyword matches in review text"):
                        for h in rv.keyword_hits[:15]:
                            st.markdown(f"`{h['keyword']}`: {h['snippet']} ([page]({h['url']}))")

    elif stage == "dishes" and q:
        for d in s.dossiers:
            items = regular_items(d.menu, d.reviews, q.diet)
            st.markdown(f"**{esc(d.candidate.name)}**: {price_summary(d.menu.items)}")
            if d.cost:
                st.markdown(f"Cost per person: **{fmt_range(d.cost.low, d.cost.high, d.cost.currency)}**. "
                            f"{d.cost.verdict}")
                st.caption("How: " + d.cost.basis)
            else:
                st.caption("Cost per person unknown: no prices were found.")
            if items:
                table([{"Popular": "★" if it.popular else "", "Dish": it.english_name, "Original": it.original_name,
                        "Price": money(it.price, it.currency), "Diet": DIET_LABEL.get(it.diet, it.diet)} for it in items])
            else:
                st.caption("No dishes matching your diet were found on this menu.")

    elif stage == "recommend":
        for c in s.checks:
            st.caption(c)


# ===================================================================== results
DIET_MARK = {"vegan": ("VG", "vegan"), "veg": ("V", "vegetarian")}


def menu_card_html(d: Dossier) -> str:
    """The full translated menu, laid out like a printed menu: sections, dish, price, original name."""
    m = d.menu
    if not m.items:
        return ""
    sections: dict[str, list] = {}
    for it in m.items:
        sections.setdefault(it.section.strip() or "Menu", []).append(it)
    lang_from = m.detected_language or ""
    out = ['<div class="rs-menucard">',
           f'<div class="rs-menucard-head"><div class="rs-menucard-name">{esc(d.candidate.name)}</div>'
           f'<div class="rs-menucard-sub">{len(m.items)} dishes'
           + (f' · translated from {esc(lang_from)}' if lang_from and lang_from.lower() != (m.translated_to or "English").lower() else "")
           + ' · V vegetarian · VG vegan</div></div>']
    for sec, items in sections.items():
        out.append(f'<div class="rs-menucard-sec">{esc(sec)}</div>')
        for it in items:
            mark = DIET_MARK.get(it.diet)
            orig = it.original_name if it.original_name.strip().lower() != it.english_name.strip().lower() else ""
            out.append(
                '<div class="rs-mi"><div class="rs-mi-row">'
                f'<span class="rs-mi-name">{esc(it.english_name)}'
                + (f' <span class="rs-mi-diet" title="{mark[1]}">{mark[0]}</span>' if mark else "")
                + (' <span class="rs-mi-pop">reviewers\' pick</span>' if it.popular else "")
                + '</span><span class="rs-leader"></span>'
                f'<span class="rs-mi-price">{esc(money(it.price, it.currency)) if it.price is not None else ""}</span></div>'
                + (f'<div class="rs-mi-orig">{esc(orig)}</div>' if orig else "")
                + (f'<div class="rs-mi-desc">{esc(it.description)}</div>' if it.description else "")
                + '</div>')
    out.append("</div>")
    return "".join(out)


def render_pick(p: Pick, d: Dossier, city: str) -> None:
    c = d.candidate
    links = all_links(d, city)
    with st.container(border=True):
        facts = [chip(x) for x in [c.cuisine, c.address or c.area_hint] if x]
        sl = d.shortlist
        if sl and sl.open_confirmed:
            facts.insert(0, chip("✓ Open on Google Maps"))
        elif sl and sl.open_checks:
            facts.insert(0, chip("No closure found, check Maps before going", "mixed"))
        if d.cost:
            word = FIT_WORD.get(d.cost.fit, "")
            facts.insert(0, chip(f"{fmt_range(d.cost.low, d.cost.high, d.cost.currency)} per person"
                                 + (f", {word}" if word else ""), FIT_CLASS.get(d.cost.fit, "")))
        if len(c.mentions) > 1:
            facts.append(chip(f"Recommended by {len(c.mentions)} food guides"))
        st.markdown(
            f'<div class="rs-name">{esc(c.name)}</div>'
            f'<div class="rs-headline">{esc(p.headline)}</div>'
            f'<div class="rs-chips">{"".join(facts)}</div>', unsafe_allow_html=True)

        cols = st.columns(len(links))
        for col, ln in zip(cols, links):
            col.link_button(ln.label, ln.url, width="stretch", type="primary" if ln is links[0] else "secondary")
        missing = [x for x, ok in (("a website", bool(c.website)), ("a menu online", bool(d.menu.menu_url))) if not ok]
        if missing:
            st.caption(f"We couldn't find {' or '.join(missing)} for this restaurant.")

        if p.for_you:
            st.markdown('<div class="rs-sub">For what you asked</div>' + "".join(
                f'<div class="rs-need"><span class="rs-need-k">{esc(n.need)}</span>'
                f'<span class="rs-need-v">{esc(n.insight)}</span></div>' for n in p.for_you), unsafe_allow_html=True)

        left, right = st.columns([1.1, 1])
        with left:
            if p.what_to_order:
                lines = "".join(
                    f'<div class="rs-dish"><span class="n">{esc(o.item)}</span><span class="rs-leader"></span>'
                    f'<span class="p">{esc(money(o.price, o.currency)) or "ask"}</span></div>'
                    f'<p class="rs-why">{esc(o.why)}</p>' for o in p.what_to_order)
                st.markdown('<div class="rs-sub">What to order</div>' + lines
                            + (f'<div class="rs-spend">Expect about {esc(p.estimated_spend)}</div>' if p.estimated_spend else ""),
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="rs-sub">What to order</div><p class="rs-why">This restaurant doesn\'t publish '
                            'a menu we could read online, so ask for today\'s specials.</p>', unsafe_allow_html=True)
        with right:
            blocks = []
            if p.why_it_fits:
                blocks.append('<div class="rs-sub">Why go</div><ul class="rs-list">'
                              + "".join(f"<li>{esc(x)}</li>" for x in p.why_it_fits) + "</ul>")
            if p.good_to_know:
                blocks.append('<div class="rs-sub">Good to know</div><ul class="rs-list">'
                              + "".join(f"<li>{esc(x)}</li>" for x in p.good_to_know) + "</ul>")
            if p.watch_outs:
                blocks.append('<div class="rs-sub">Heads-up</div><ul class="rs-list">'
                              + "".join(f"<li>{esc(x)}</li>" for x in p.watch_outs) + "</ul>")
            st.markdown("".join(blocks), unsafe_allow_html=True)

        m = d.menu
        if m.items:
            with st.expander(f"See the full menu ({len(m.items)} dishes)", expanded=False):
                st.markdown(menu_card_html(d), unsafe_allow_html=True)
                srcs = m.menu_urls or [m.menu_url]
                st.caption("Read from " + " · ".join(f"[{u.split('//')[-1][:60]}]({u})" for u in srcs if u)
                           + f". {price_summary(m.items)}")
        elif m.menu_url:
            st.caption(f"The menu is online but couldn't be read automatically: [open it]({m.menu_url}).")

        with st.expander("How this was chosen"):
            render_how(p, d)


def render_how(p: Pick, d: Dossier) -> None:
    """Everything that led to this suggestion, in the order the agent did it."""
    c, row = d.candidate, d.shortlist
    st.markdown(f"**1. Found in {len(c.mentions)} food guide{'s' if len(c.mentions) != 1 else ''}**")
    for m in c.mentions[:6]:
        st.markdown(f"- [{m.title or m.url}]({m.url}): {m.why}")
    if not c.mentions:
        st.caption("No guide mentions were kept.")

    st.markdown("**2. Checks in the shortlist**")
    for r in row.reasons:
        st.markdown(f"- {r}")

    st.markdown("**3. Cost per person**")
    if d.cost:
        st.markdown(f"- {d.cost.basis}\n- {d.cost.verdict}")
    else:
        st.markdown("- No prices were found on the menu or in the guides, so the cost is unknown.")

    st.markdown("**4. What reviews say about your needs**")
    for f in d.reviews.context or []:
        st.markdown(f"- {VERDICT.get(f.verdict, ('', '?'))[1]} **{f.requirement}**: {f.evidence or NEED_WORDS.get(f.verdict, '')}"
                    + (f" ([source]({f.source_url}))" if f.source_url else ""))
    if not d.reviews.context:
        st.caption("Reviews didn't discuss your needs for this place.")

    st.markdown("**5. The write-up**")
    st.markdown("- The AI summarised the evidence above into this card. It doesn't score or rank the places.\n"
                "- Code then checked it: the dishes it suggests are on the real menu, and every price and cost "
                "shown comes from the menu, not from the AI.")


def render_run(s: RunState) -> None:
    st.markdown("#### How the agent worked")
    for key, title, desc in STAGES:
        logs = s.stage_logs.get(key)
        if not logs and key != "understand":
            continue
        with st.expander(f"{STAGE_INFO[key][0]}. {title}", expanded=False):
            st.markdown(f'<p class="rs-step">{esc(desc)}</p>', unsafe_allow_html=True)
            for line in logs or []:
                st.markdown(f"- {esc(line)}")
            stage_details(key, s)
    if s.error:
        st.error(s.error)
    rec = s.recommendation
    city = s.query.city if s.query else ""
    if rec:
        st.markdown("#### Where to eat")
        if s.query:
            st.markdown(f"Your budget: **{budget_text(s.query)}**")
        passed = sum(r.passed for r in s.shortlist_rows)
        read = sum(1 for x in s.guide_sources if x.get("read"))
        st.caption(f"How these were chosen: {len(s.candidates)} restaurants found in {read} food guide pages → "
                   f"{passed} passed the checks → the {len(s.dossiers)} recommended by the most guides were looked "
                   f"at in depth (menu, prices, reviews) → shown below. Open “How this was chosen” on any card, "
                   f"or see “Where the data came from” at the bottom.")
        st.markdown(f'<div class="rs-summary">{esc(rec.summary)}</div>', unsafe_allow_html=True)
        by_id = {d.candidate.place_id: d for d in s.dossiers}
        for p in rec.picks:
            if p.place_id in by_id:
                render_pick(p, by_id[p.place_id], city)
        render_sources(s)
        st.download_button("Download these suggestions as a PDF", pdf_for_run(s.started_at + s.request, s),
                           file_name=pdf_name(s), mime="application/pdf", type="primary", key=f"doc_{s.started_at}",
                           help="A clean document with every website, menu and Google Maps link.")
    u = s.usage
    if u:
        pages = u.get("pages", {})
        st.caption(f"{u.get('calls', 0)} AI calls ({s.provider}), {u.get('web_searches', 0)} web searches"
                   + (f" ({u['groq_backup_searches']} by Groq)" if u.get("groq_backup_searches") else "")
                   + (f", {pages.get('direct', 0) + pages.get('jina', 0)} pages read ({pages.get('jina', 0)} with Jina Reader)"
                      if pages else "") + ". Cost: free.")
        for note in u.get("ai_notes", [])[:3]:
            st.caption(note)


def render_sources(s: RunState) -> None:
    """The visual of where every piece of data came from: a diagram, a map and the list of websites."""
    st.markdown("#### Where the data came from")
    diagram, where, sites = st.tabs(["Diagram", "Map of the restaurants", "Websites used"])
    with diagram:
        st.caption("Green boxes are the free sources. Arrows show which step used them. The yellow box is the AI, "
                   "which only reads and writes: every fact comes from a source on the left.")
        try:
            st.graphviz_chart(to_dot(summarise(s)), width="stretch")
        except Exception:  # never let the picture break the page
            st.info("The diagram couldn't be drawn here; the table under “Websites used” has the same information.")
    with where:
        render_map(s)
    with sites:
        rows = source_rows(s)
        if rows:
            st.caption(f"{len(rows)} websites, {sum(r['Pages'] for r in rows)} pages in total.")
            table(rows)
        else:
            st.caption("No websites were read.")


def render_map(s: RunState) -> None:
    picks = {p.place_id for p in (s.recommendation.picks if s.recommendation else [])}
    pts = [{"name": c.name, "lat": c.lat, "lon": c.lon,
            "label": c.name if c.place_id in picks else "",
            "kind": "Suggested" if c.place_id in picks else "Also found",
            "color": [46, 83, 57, 230] if c.place_id in picks else [150, 158, 152, 160],
            "radius": 70 if c.place_id in picks else 40}
           for c in s.candidates if c.lat is not None and c.lon is not None]
    if not pts:
        st.caption("None of the restaurants could be placed on the map (OpenStreetMap didn't have them).")
        return
    df = pd.DataFrame(pts)
    view = pdk.ViewState(latitude=float(df.lat.mean()), longitude=float(df.lon.mean()), zoom=13)
    layers = [pdk.Layer("ScatterplotLayer", df, get_position="[lon, lat]", get_fill_color="color",
                        get_radius="radius", radius_min_pixels=6, radius_max_pixels=18, pickable=True),
              pdk.Layer("TextLayer", df[df.label != ""], get_position="[lon, lat]", get_text="label",
                        get_color=[29, 35, 32], get_size=13, get_pixel_offset=[0, -18],
                        get_alignment_baseline="'bottom'")]
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_provider="carto", map_style="light",
                             tooltip={"text": "{name}\n{kind}"}), height=420)
    st.caption(f"Green dots are your suggestions; grey dots are other restaurants the guides mentioned. "
               f"Locations from OpenStreetMap ({len(pts)} of {len(s.candidates)} restaurants placed).")


def live_run(agent: Agent, request: str, menu_language: str, fixed: dict | None = None) -> RunState:
    boxes, state = {}, None
    for ev in agent.run(request, menu_language, fixed):
        if ev.kind == "start":
            i, title, desc = STAGE_INFO[ev.stage]
            boxes[ev.stage] = st.status(f"{i}. {title}", expanded=True, state="running")
            boxes[ev.stage].caption(desc)
        elif ev.kind == "log":
            boxes[ev.stage].write(ev.message)
        elif ev.kind == "done":
            with boxes[ev.stage]:
                stage_details(ev.stage, agent.state)
            boxes[ev.stage].update(state="complete", expanded=False)
        elif ev.kind == "error":
            if ev.stage in boxes:
                boxes[ev.stage].update(state="error", expanded=not agent.state.needs_info)
        elif ev.kind == "final":
            state = ev.payload
    return state


# ===================================================================== settings
def secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, ""))
    except Exception:  # no secrets file
        return ""


settings = Settings.from_env(get=lambda k: os.getenv(k) or secret(k))   # keys: .env file or Streamlit Secrets
ss.setdefault("depth", settings.shortlist_size)
ss.setdefault("menu_language", "English")


def pdf_name(r: RunState) -> str:
    city = re.sub(r"[^a-z0-9]+", "-", (r.query.city if r.query else "search").lower()).strip("-") or "search"
    return f"restaurants-{city}-{r.started_at[:16].replace(':', '').replace('T', '-')}.pdf"


@st.cache_data(show_spinner=False, max_entries=40)
def pdf_for_run(key: str, _run: RunState) -> bytes:     # built once per search, then reused
    return build_pdf(_run)


@st.cache_data(show_spinner=False, max_entries=5)
def history_pdf(key: tuple, _runs: list) -> bytes:
    return build_history_pdf(_runs)


def saved_runs(limit: int = 12) -> list[tuple[Path, RunState]]:
    out = []
    for f in sorted(Path("runs").glob("*.json"), reverse=True)[:limit]:
        try:
            out.append((f, load_run(f)))
        except Exception:  # skip files from an older version
            continue
    return out


with st.sidebar:
    st.markdown("### Options")
    st.selectbox("Translate menus into", MENU_LANGUAGES, key="menu_language")
    st.slider("Restaurants to analyse in depth", 2, 6, key="depth", help="More means a longer wait on free plans.")
    settings.shortlist_size = ss.depth
    if st.button("Check services", width="stretch", help="Tests each free service on its own (about a minute)."):
        with st.spinner("Checking each service"):
            ss.health = check_all(settings)
    if ss.health:
        for c in ss.health:
            (st.success if c.ok else st.error)(f"**{c.service}**: {'OK' if c.ok else 'failed'} in {c.seconds}s. {c.detail}")

    st.divider()
    if ss.run and st.button("Start a new search", width="stretch", type="primary"):
        ss.messages, ss.run = [], None
        st.rerun()
    st.markdown("### Search history")
    runs = saved_runs()
    if not runs:
        st.caption("Your searches appear here after they finish, ready to download as PDFs.")
    else:
        st.download_button(f"Download all {len(runs)} as one PDF", history_pdf(tuple(str(f) for f, _ in runs),
                                                                              [r for _, r in runs]),
                           file_name=f"restaurant-search-history-{dt.date.today()}.pdf", mime="application/pdf",
                           width="stretch", key="history_pdf")
    for i, (f, r) in enumerate(runs):
        city = r.query.city if r.query else "Search"
        when = r.started_at.replace("T", " ")[:16]
        with st.container(border=True):
            st.markdown(f"**{esc(city)}** · {when}")
            st.caption((r.request[:70] + "…") if len(r.request) > 70 else r.request)
            a, b = st.columns(2)
            a.download_button("PDF", pdf_for_run(str(f), r), file_name=pdf_name(r), mime="application/pdf",
                              key=f"dl{i}", width="stretch", help="This search as a PDF, with every link clickable.")
            if b.button("Open", key=f"open{i}", width="stretch"):
                ss.run = r
                ss.messages = [{"role": "user", "content": r.request}, {"role": "assistant", "content": "__RUN__"}]
                st.rerun()

# ======================================================================== main
def build_request(city: str, diet: str, meal: str, budget: tuple, extra: str) -> tuple[str, dict]:
    """Turn the form choices into one request sentence for the agent, plus the choices it must keep."""
    d_code, d_words = DIETS.get(diet, ("any", ""))
    m_code = MEALS.get(meal, "any")
    meal_words = "food" if m_code == "any" else m_code
    text = f"{d_words + ' ' if d_words else ''}{meal_words} in {city}"
    text = text[0].upper() + text[1:]
    lo, hi, cur = budget
    if lo or hi:
        text += ", " + (f"under {fmt_price(hi, cur)}" if not lo else f"over {fmt_price(lo, cur)}" if not hi
                        else f"{fmt_range(lo, hi, cur)}") + " per person"
    if extra.strip():
        text += f". Also: {extra.strip()}"
    fixed = {"city": city, "diet": d_code, "meal_type": m_code}
    if lo or hi:       # the numbers the traveller picked win over the AI's reading
        fixed.update({"budget_min": lo, "budget_max": hi, "currency": cur, "budget_estimated": False})
    return text, fixed


def search_form() -> None:
    with st.container(border=True):
        st.markdown('<p class="rs-sub" style="margin-top:0">Plan your meal</p>', unsafe_allow_html=True)
        names = list(CITIES)
        city_pick = st.selectbox("City", [f"{CITIES[c]} {c}" for c in names] + [OTHER_CITY],
                                 index=names.index("Barcelona"), key="f_city")
        other = ""
        if city_pick == OTHER_CITY:
            other = st.text_input("Which European city?", placeholder="e.g. Seville, Bruges, Split…", key="f_other")
        a, b = st.columns(2)
        diet = a.segmented_control("Diet", list(DIETS), default="Any", key="f_diet")
        meal = b.segmented_control("Meal", list(MEALS), default="Dinner", key="f_meal")
        chosen_city = city_pick.split(" ", 1)[1] if city_pick != OTHER_CITY else ""
        currency = CITY_CURRENCY.get(chosen_city, "EUR")
        options = budget_options(currency)
        budget = st.pills("Budget per person", list(options), default="Any budget", key=f"f_budget_{currency}",
                          help="In the city's own currency. For a typed-in city outside the euro area, "
                               "it's converted to the local currency.")
        extra = st.text_area("Anything else? (optional)", max_chars=300, height=80, key="f_extra",
                             placeholder="e.g. outdoor seating, quiet, kid-friendly, gluten-free, near the old town")
        if st.button("Find restaurants", type="primary", width="stretch", key="f_go"):
            city = other.strip() if city_pick == OTHER_CITY else city_pick.split(" ", 1)[1]
            if not city:
                ss.notice = "Type the city you're eating in."
            else:
                ss.notice = ""
                ss.pending = build_request(city, diet or "Any", meal or "Any",
                                           options.get(budget or "Any budget", (None, None, currency)), extra)
                if city_pick == OTHER_CITY:     # unknown currency: the AI converts the euro amounts
                    for k in ("budget_min", "budget_max", "currency", "budget_estimated"):
                        ss.pending[1].pop(k, None)
            st.rerun()
        if ss.notice:
            st.warning(ss.notice)


if not ss.run and not ss.pending:
    dishes = [("Pimientos de Padrón", "Barcelona", "7 €"), ("Pastel de nata", "Lisbon", "1.40 €"),
              ("Wiener Schnitzel", "Vienna", "24 €"), ("Svíčková", "Prague", "289 Kč")]
    menu = "".join(f'<div class="rs-line"><span>{esc(d)} <small>· {esc(c)}</small></span>'
                   f'<span class="rs-leader"></span><span>{esc(p)}</span></div>' for d, c, p in dishes)
    st.markdown(
        '<div class="rs-hero">'
        '<p class="rs-eyebrow">Restaurant Scout · Europe</p>'
        '<h1>From tapas in Barcelona to schnitzel in Vienna, find somewhere good to eat.</h1>'
        '<p>Pick the city, your diet, the meal and your budget. I read local food guides, check each place on the map '
        'and make sure it is still open, translate its menu from the local language, and weigh up both praise and '
        'complaints before I suggest anything.</p></div>', unsafe_allow_html=True)
    if not settings.has_ai:
        st.warning("The app isn't connected to an AI yet. Add your free Gemini key to the `.env` file "
                   "(or to Secrets when it's online) as `GEMINI_API_KEY=...`, then restart the app. "
                   "The README explains how.")
    search_form()
    st.markdown(f'<div class="rs-menu"><p class="rs-menu-title">Today across Europe</p>{menu}</div>'
                f'<p class="rs-cities">{len(CITIES)} cities in the list, or type any other city in Europe. '
                'Prices are shown in the local currency.</p>', unsafe_allow_html=True)

for m in ss.messages:
    with st.chat_message(m["role"]):
        if m["content"] == "__RUN__" and ss.run:
            render_run(ss.run)
        else:
            st.markdown(m["content"])

if ss.pending and not ss.run:                  # ---- a new search from the form
    request, fixed = ss.pending
    ss.messages = [{"role": "user", "content": request}]
    with st.chat_message("user"):
        st.markdown(request)
    with st.chat_message("assistant"):
        if not settings.has_ai:
            ss.pending, ss.messages = None, []
            ss.notice = "I'm not connected to an AI yet. Add `GEMINI_API_KEY=...` to the `.env` file, then restart the app."
            st.rerun()
        st.caption("This takes 2 to 4 minutes on free plans. You can watch each step below.")
        state = live_run(Agent.from_settings(settings), request, ss.menu_language, fixed)
    ss.pending = None
    if state and state.needs_info:              # e.g. not a European city, or off-topic extras: back to the form
        ss.messages, ss.notice = [], state.error
        st.rerun()
    elif state:
        try:
            save_run(state)
        except OSError:
            pass
        ss.run = state
        ss.messages.append({"role": "assistant", "content": "__RUN__"})
        st.rerun()

prompt = st.chat_input(max_chars=600, placeholder="Ask a follow-up question about these restaurants") if ss.run else None

if prompt and ss.run:                          # ---- a follow-up question
    ss.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        if not ss.run.query or not ss.run.dossiers:
            answer = "The last search didn't finish, so there's nothing to answer from. Click **Start a new search**."
        elif not settings.has_ai:
            answer = "Add your free Gemini key to the `.env` file to ask follow-up questions."
        else:
            agent = Agent.from_settings(settings)
            history = [m for m in ss.messages[:-1] if m["content"] != "__RUN__"]
            with st.spinner("Checking what I found"):
                try:
                    answer = answer_followup(prompt, history, ss.run.query, ss.run.dossiers,
                                             ss.run.recommendation, agent.llm)
                except Exception as e:  # show the problem instead of a crash
                    answer = f"I couldn't answer that: {e}"
        st.markdown(answer)
    ss.messages.append({"role": "assistant", "content": answer})

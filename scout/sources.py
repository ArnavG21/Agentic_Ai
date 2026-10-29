"""Where the data came from - counts every source a search used, for the diagram and table in the app.

summarise(state) -> SourceSummary   numbers for each kind of source
to_dot(summary)  -> str             a Graphviz diagram: sources -> steps -> your suggestions
source_rows(state) -> list[dict]    one row per website the agent read, and what it was used for
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from .pipeline import RunState


def domain(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


@dataclass
class SourceSummary:
    searches: int = 0
    groq_searches: int = 0
    guide_pages: int = 0
    guide_domains: list[str] = field(default_factory=list)
    restaurants_found: int = 0
    on_map: int = 0
    menus_read: int = 0
    menus_via_jina: int = 0
    menu_domains: list[str] = field(default_factory=list)
    review_pages: int = 0
    review_domains: list[str] = field(default_factory=list)
    jina_pages: int = 0
    ai_models: dict = field(default_factory=dict)
    suggestions: int = 0


def summarise(s: "RunState") -> SourceSummary:
    u = s.usage or {}
    out = SourceSummary(searches=u.get("web_searches", 0), groq_searches=u.get("groq_backup_searches", 0),
                        jina_pages=(u.get("pages") or {}).get("jina", 0), ai_models=dict(u.get("models") or {}))
    read = [g for g in s.guide_sources if g.get("read")]
    out.guide_pages = len(read)
    out.guide_domains = [d for d, _ in Counter(domain(g["url"]) for g in s.guide_sources).most_common(4)]
    out.restaurants_found = len(s.candidates)
    out.on_map = sum(c.on_map for c in s.candidates)
    menus = [d.menu for d in s.dossiers if d.menu.how_found != "none"]
    out.menus_read = len(menus)
    out.menus_via_jina = sum("jina" in m.how_found for m in menus)
    out.menu_domains = [domain(m.menu_url) for m in menus if m.menu_url][:4]
    review_urls = [x["url"] for d in s.dossiers for x in d.reviews.sources]
    out.review_pages = len(review_urls)
    out.review_domains = [d for d, _ in Counter(domain(x) for x in review_urls).most_common(4)]
    out.suggestions = len(s.recommendation.picks) if s.recommendation else 0
    return out


def _esc(t: str) -> str:
    return t.replace("\\", "\\\\").replace('"', '\\"')


def _node(name: str, title: str, lines: list[str], fill: str, border: str) -> str:
    body = "".join(f'<BR/><FONT POINT-SIZE="10" COLOR="#4A524C">{_html(x)}</FONT>' for x in lines if x)
    return (f'  {name} [label=<<B>{_html(title)}</B>{body}>, shape=box, style="rounded,filled", '
            f'fillcolor="{fill}", color="{border}", margin="0.18,0.1"];\n')


def _html(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_dot(m: SourceSummary) -> str:
    """Three columns: the free sources (left) feed the agent's steps (middle), which produce your suggestions (right)."""
    src_fill, src_border = "#E3EBE1", "#2E5339"
    ai_fill, ai_border = "#F8EBCB", "#D99A1E"
    more = lambda xs: ", ".join(xs[:2]) + ("…" if len(xs) > 2 else "")
    ai_lines = [f"{k.split(' (')[0]}: {v} calls" for k, v in list(m.ai_models.items())[:2]] or ["used at 5 steps"]
    g = ['digraph sources {\n  rankdir=LR; nodesep=0.22; ranksep=1.1; bgcolor="transparent"; newrank=true;\n'
         '  node [fontname="Helvetica", fontsize=13, width=2.6]; edge [color="#7C857E", arrowsize=0.7];\n']
    sources = [
        _node("search", "Web search", [f"{m.searches} searches (DuckDuckGo)",
                                        f"{m.groq_searches} by Groq as backup" if m.groq_searches else ""], src_fill, src_border),
        _node("guides", "Food guides", [f"{m.guide_pages} pages read", more(m.guide_domains)], src_fill, src_border),
        _node("osm", "OpenStreetMap", [f"{m.on_map} of {m.restaurants_found} places confirmed",
                                        "address, hours, diet tags"], src_fill, src_border),
        _node("sites", "Restaurant websites", [f"{m.menus_read} menus read",
                                                f"{m.menus_via_jina} with Jina Reader" if m.menus_via_jina else "",
                                                more(m.menu_domains)], src_fill, src_border),
        _node("reviews", "Review pages", [f"{m.review_pages} pages", more(m.review_domains)], src_fill, src_border),
        _node("ai", "AI (reads and writes)", ai_lines, ai_fill, ai_border),
    ]
    g.extend(sources)
    steps = [("s1", "1  Understand your request"), ("s2", "2  Find restaurants"), ("s3", "3  Shortlist"),
             ("s4", "4–5  Read and translate menus"), ("s6", "6–7  Reviews and your needs"),
             ("s8", "8  Dishes and cost per person"), ("s9", "9  Suggest and fact-check")]
    for k, t in steps:
        g.append(f'  {k} [label="{_esc(t)}", shape=box, style="rounded,filled", fillcolor="#FFFFFF", '
                 f'color="#9AA39C", width=3.0];\n')
    g.append(f'  out [label=<<B>Your {m.suggestions} suggestion{"s" if m.suggestions != 1 else ""}</B>'
             f'<BR/><FONT POINT-SIZE="11">website, menu, map, cost</FONT>>, shape=box, style="rounded,filled", '
             f'fillcolor="#2E5339", fontcolor="white", color="#2E5339", width=2.2];\n')
    # three columns: everything in a column shares a rank; order inside a column is kept with invisible edges
    g.append("  { rank=same; search; guides; osm; sites; reviews; ai; }\n")
    g.append("  { rank=same; " + "; ".join(k for k, _ in steps) + "; }\n")
    order = ["search", "guides", "osm", "sites", "reviews", "ai"]
    for a, b in zip(order, order[1:]):
        g.append(f"  {a} -> {b} [style=invis];\n")
    keys = [k for k, _ in steps]
    for a, b in zip(keys, keys[1:]):
        g.append(f'  {a} -> {b} [color="#C2C8C3", arrowsize=0.5];\n')
    g.append('  s9 -> out [color="#2E5339", penwidth=1.8];\n')
    for a, b in [("search", "s2"), ("guides", "s2"), ("osm", "s2"), ("osm", "s3"),
                 ("sites", "s4"), ("reviews", "s6")]:
        g.append(f"  {a} -> {b};\n")
    for b in ("s1", "s2", "s4", "s6", "s9"):
        g.append(f'  ai -> {b} [style=dashed, color="#D99A1E"];\n')
    g.append("}\n")
    return "".join(g)


def source_rows(s: "RunState") -> list[dict]:
    """Every website the agent read, grouped by site, with what it was used for."""
    use: dict[str, dict] = {}

    def add(url: str, what: str, via: str = ""):
        d = domain(url)
        if not d:
            return
        row = use.setdefault(d, {"Website": d, "Pages": 0, "Used for": set(), "Read with": set()})
        row["Pages"] += 1
        row["Used for"].add(what)
        row["Read with"].add("Jina Reader" if via == "jina" else ("search preview" if via == "preview" else "direct download"))

    for g in s.guide_sources:
        add(g["url"], "finding restaurants", g.get("via") or ("" if g.get("read") else "preview"))
    for d in s.dossiers:
        for p in d.menu.pages_tried:
            add(p.replace(" (via Jina Reader)", ""), "menu", "jina" if "Jina" in p else "")
        for x in d.reviews.sources:
            add(x["url"], "reviews")
    rows = sorted(use.values(), key=lambda r: r["Pages"], reverse=True)
    return [{**r, "Used for": ", ".join(sorted(r["Used for"])), "Read with": ", ".join(sorted(r["Read with"]))}
            for r in rows]

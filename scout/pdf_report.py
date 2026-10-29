"""Searches as PDF documents: one search, or the whole search history in one file.

Built with fpdf2 (pure Python, free). The DejaVu Sans font is bundled in scout/fonts so
every European alphabet prints correctly (ç, ø, ł, ő, Greek, Cyrillic...). Every website,
menu and Google Maps link in the PDF is clickable.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from fpdf import FPDF
from fpdf.enums import WrapMode, XPos, YPos
from fpdf.fonts import FontFace

from .links import all_links
from .money import budget_text, fmt_price, fmt_range
from .sources import source_rows, summarise

if TYPE_CHECKING:
    from .pipeline import RunState

FONTS = Path(__file__).parent / "fonts"
OLIVE, SAFFRON, INK, STONE = (46, 83, 57), (217, 154, 30), (29, 35, 32), (95, 103, 97)
OLIVE_SOFT, SAFFRON_SOFT, TOMATO, LINE = (227, 235, 225), (248, 235, 203), (181, 67, 47), (213, 219, 210)
FIT_WORD = {"fits": "within budget", "under": "under budget", "partly": "partly within budget", "over": "over budget"}
FIT_COLOUR = {"fits": OLIVE, "under": OLIVE, "partly": (138, 93, 5), "over": TOMATO}


def clean(text) -> str:
    """Keep characters the bundled font can draw (all European scripts); drop emoji and the like."""
    out = []
    for ch in str(text or ""):
        o = ord(ch)
        if o < 0x2E80 or 0xFB00 <= o <= 0xFFFD:
            out.append(ch)
    return "".join(out)


class ReportPDF(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.add_font("DejaVu", "", str(FONTS / "DejaVuSans.ttf"))
        self.add_font("DejaVu", "B", str(FONTS / "DejaVuSans-Bold.ttf"))
        self.set_margins(18, 18, 18)
        self.set_auto_page_break(True, 20)
        self.set_title("Restaurant Scout")
        self.set_creator("Restaurant Scout")

    def footer(self):
        self.set_y(-13)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(*STONE)
        self.cell(0, 6, f"Restaurant Scout  ·  page {self.page_no()}", align="C")

    # ---------------------------------------------------------------- text
    def style(self, size: float = 10, bold: bool = False, colour=INK):
        self.set_font("DejaVu", "B" if bold else "", size)
        self.set_text_color(*colour)

    def para(self, text, size: float = 10, bold: bool = False, colour=INK, h: float = 5.2, gap: float = 1.2,
             link: str = "", wrap_chars: bool = False):
        self.style(size, bold, colour)
        self.multi_cell(0, h, clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT, link=link or None, align="L",
                        wrapmode=WrapMode.CHAR if wrap_chars else WrapMode.WORD)
        self.ln(gap)

    def heading(self, text, size: float = 12, colour=INK, before: float = 3):
        if self.get_y() > self.h - 45:      # don't leave a heading alone at the bottom of a page
            self.add_page()
        self.ln(before)
        self.para(text, size=size, bold=True, colour=colour, h=size * 0.5, gap=1.5)

    def bullets(self, items: Iterable[str], colour=INK, size: float = 9.5, numbered: bool = False):
        for i, it in enumerate(items, 1):
            self.style(size, False, colour)
            self.set_x(self.l_margin + 1)
            self.cell(6, 4.9, f"{i}." if numbered else "•")
            self.multi_cell(0, 4.9, clean(it), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
            self.ln(0.6)

    def link_row(self, label: str, url: str):
        self.style(9, False, STONE)
        self.cell(34, 5, clean(label))
        self.style(9, False, OLIVE)
        self.multi_cell(0, 5, clean(url), link=url, new_x=XPos.LMARGIN, new_y=YPos.NEXT, wrapmode=WrapMode.CHAR)
        self.ln(0.6)

    def boxed(self, text, fill=SAFFRON_SOFT, size: float = 10):
        self.style(size, False, INK)
        self.set_fill_color(*fill)
        self.multi_cell(0, 5.4, clean(text), fill=True, padding=(3, 4, 3, 4), new_x=XPos.LMARGIN, new_y=YPos.NEXT,
                        align="L")
        self.set_fill_color(255, 255, 255)
        self.ln(3)

    def rule(self):
        self.set_draw_color(*LINE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def grid(self, headings: list[str], rows: list[list], widths: tuple, size: float = 8.5, links: dict | None = None):
        """A simple table. links: {column index: column of URLs} makes those cells clickable."""
        if not rows:
            return
        self.style(size)
        self.set_fill_color(255, 255, 255)
        head = FontFace(emphasis="BOLD", color=STONE, fill_color=OLIVE_SOFT)
        with self.table(col_widths=widths, text_align="LEFT", line_height=4.6, headings_style=head,
                        borders_layout="HORIZONTAL_LINES", width=self.epw, padding=1.4) as t:
            r = t.row()
            for h in headings:
                r.cell(clean(h))
            for i, row in enumerate(rows):
                r = t.row()
                for j, v in enumerate(row):
                    url = (links or {}).get(j, [None] * len(rows))[i]
                    r.cell(clean(v), link=url or None,
                           style=FontFace(color=OLIVE) if url else None)
        self.ln(2)


DIET_MARK = {"vegan": "VG", "veg": "V"}


def _menu(pdf: ReportPDF, d) -> None:
    """The full translated menu, laid out like a printed menu: sections, dishes, prices."""
    m = d.menu
    if not m.items:
        return
    if pdf.get_y() > pdf.h - 70:
        pdf.add_page()
    pdf.ln(3)
    pdf.set_draw_color(*SAFFRON)
    pdf.set_line_width(0.6)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.ln(3)
    pdf.style(13, True, INK)
    pdf.cell(0, 7, clean(f"The menu  ·  {d.candidate.name}"), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    lang = m.detected_language or ""
    sub = f"{len(m.items)} dishes" + (f", translated from {lang}" if lang and lang.lower() != (m.translated_to or 'English').lower() else "")
    pdf.style(8.5, False, STONE)
    pdf.cell(0, 5, clean(sub + "  ·  V vegetarian  ·  VG vegan"), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    sections: dict[str, list] = {}
    for it in m.items:
        sections.setdefault(it.section.strip() or "Menu", []).append(it)
    price_w = 22
    for sec, items in sections.items():
        if pdf.get_y() > pdf.h - 40:
            pdf.add_page()
        pdf.ln(3)
        pdf.style(9, True, OLIVE)
        pdf.cell(0, 6, clean(sec.upper()), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        for it in items:
            if pdf.get_y() > pdf.h - 28:
                pdf.add_page()
            y = pdf.get_y()
            name = it.english_name + (f"  {DIET_MARK[it.diet]}" if it.diet in DIET_MARK else "")
            pdf.style(9.5, True, INK)
            pdf.multi_cell(pdf.epw - price_w, 5, clean(name), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
            after = pdf.get_y()
            pdf.set_xy(pdf.l_margin + pdf.epw - price_w, y)
            pdf.style(9.5, True, OLIVE)
            pdf.cell(price_w, 5, clean(fmt_price(it.price, it.currency) if it.price is not None else ""), align="R")
            pdf.set_y(after)
            if it.original_name.strip().lower() != it.english_name.strip().lower():
                pdf.style(8.3, False, STONE)
                pdf.multi_cell(pdf.epw - price_w, 4.2, clean(it.original_name), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            if it.description:
                pdf.style(8.5, False, (74, 81, 76))
                pdf.multi_cell(pdf.epw - price_w, 4.3, clean(it.description), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1.4)
    pdf.ln(1)
    srcs = m.menu_urls or [m.menu_url]
    for u in srcs[:4]:
        if u:
            pdf.para(f"Read from {u}", size=7.8, colour=STONE, link=u, wrap_chars=True, h=4, gap=0.3)


def _place(pdf: ReportPDF, p, d, city: str) -> None:
    c = d.candidate
    if pdf.get_y() > pdf.h - 90:
        pdf.add_page()
    pdf.rule()
    pdf.para(c.name, size=17, bold=True, colour=OLIVE, h=8, gap=0.5)
    pdf.para(p.headline, size=10.5, gap=1.5)

    facts = []
    sl = d.shortlist
    if sl and sl.open_confirmed:
        facts.append("Open on Google Maps")
    elif sl and sl.open_checks:
        facts.append("No closure found - check Google Maps before going")
    facts += [c.cuisine, c.address or c.area_hint]
    pdf.para("  ·  ".join(x for x in facts if x), size=9, colour=STONE, gap=1)
    if d.cost:
        word = FIT_WORD.get(d.cost.fit, "")
        pdf.para(f"About {fmt_range(d.cost.low, d.cost.high, d.cost.currency)} per person" + (f", {word}" if word else ""),
                 size=10, bold=True, colour=FIT_COLOUR.get(d.cost.fit, INK), gap=1.5)
    for ln in all_links(d, city):
        pdf.link_row(ln.label, ln.url)

    if p.for_you:
        pdf.heading("For what you asked", 11)
        for n in p.for_you:
            if pdf.get_y() > pdf.h - 30:
                pdf.add_page()
            y = pdf.get_y()
            pdf.style(9.5, True, OLIVE)
            pdf.multi_cell(34, 5, clean(n.need), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            left_end = pdf.get_y()
            pdf.set_xy(pdf.l_margin + 36, y)
            pdf.style(9.5, False, INK)
            pdf.multi_cell(pdf.epw - 36, 5, clean(n.insight), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
            pdf.set_y(max(left_end, pdf.get_y()) + 1.5)

    pdf.heading("What to order", 11)
    if p.what_to_order:
        pdf.grid(["Dish", "Price", "Why"],
                 [[o.item, fmt_price(o.price, o.currency) or "ask", o.why] for o in p.what_to_order], (40, 12, 48))
    else:
        pdf.para("This restaurant doesn't publish a menu we could read online, so ask for today's specials.",
                 size=9.5, colour=STONE)
    for title, items in (("Why go", p.why_it_fits), ("Good to know", p.good_to_know), ("Heads-up", p.watch_outs)):
        if items:
            pdf.heading(title, 11)
            pdf.bullets(items)
    _menu(pdf, d)
    if c.mentions:
        pdf.ln(2)
        pdf.para("Recommended by: " + ", ".join(clean(g.title or g.url)[:60] for g in c.mentions[:4]),
                 size=8, colour=STONE)


def _search(pdf: ReportPDF, s: "RunState") -> None:
    q = s.query
    city = q.city if q else ""
    when = s.started_at.replace("T", " ")[:16] if s.started_at else ""
    pdf.para(f"Where to eat in {city}" if city else "Restaurant suggestions", size=22, bold=True, h=10, gap=2)
    pdf.para(f"“{s.request}”", size=11, gap=1)
    pdf.para(f"Budget: {budget_text(q) if q else 'not given'}  ·  Menus in {q.menu_language if q else 'English'}  ·  {when}",
             size=9, colour=STONE, gap=4)
    rec = s.recommendation
    if s.error and not rec:
        pdf.boxed(f"This search didn't finish: {s.error}", fill=(245, 223, 218))
        return
    if rec:
        pdf.boxed(rec.summary)
    by_id = {d.candidate.place_id: d for d in s.dossiers}
    for p in (rec.picks if rec else []):
        if p.place_id in by_id:
            _place(pdf, p, by_id[p.place_id], city)

    closed = [r for r in s.shortlist_rows if r.closed]
    m = summarise(s)
    pdf.add_page()
    pdf.para("How this was put together", size=15, bold=True, h=7, gap=2)
    steps = [
        f"Understood the request: {city}, meal: {q.meal_type if q else ''}, diet: {q.diet if q else ''}, "
        f"budget: {budget_text(q) if q else ''}.",
        f"Found {m.restaurants_found} restaurants in {m.guide_pages} food guide pages ({m.searches} web searches) and "
        f"confirmed {m.on_map} of them on OpenStreetMap.",
        "Shortlisted with simple yes/no checks; the places recommended by the most guides were checked to make sure "
        "they're still open (Google Maps when available, their websites, review listings and closure news)"
        + (f" ({len(closed)} found closed: {', '.join(r.name for r in closed)})" if closed else "") + ".",
        f"Read {m.menus_read} menus from the restaurants' websites"
        + (f" ({m.menus_via_jina} with the Jina Reader scraper)" if m.menus_via_jina else "") + " and translated them.",
        f"Read {m.review_pages} review pages for praise, complaints and the traveller's needs.",
        "Worked out the cost per person from each menu and compared it with the budget.",
        "Wrote up each place from this evidence only, then checked every suggested dish and price against the menus.",
    ]
    pdf.bullets(steps, numbered=True)
    rows = source_rows(s)[:30]
    if rows:
        pdf.heading("Websites used", 11)
        pdf.grid(["Website", "Pages", "Used for"], [[r["Website"], r["Pages"], r["Used for"]] for r in rows],
                 (40, 10, 50), links={0: ["https://" + r["Website"] for r in rows]})
    pdf.ln(3)
    pdf.para("Menus, prices and opening hours come from the restaurants' websites and food guides and may have "
             "changed, so check before you go.", size=8.5, colour=STONE)


def build_pdf(s: "RunState") -> bytes:
    pdf = ReportPDF()
    pdf.add_page()
    _search(pdf, s)
    return bytes(pdf.output())


def build_history_pdf(runs: list["RunState"]) -> bytes:
    """All saved searches in one PDF: a contents page, then each search."""
    pdf = ReportPDF()
    pdf.add_page()
    pdf.para("Your restaurant searches", size=24, bold=True, h=11, gap=2)
    pdf.para(f"Search history from Restaurant Scout  ·  {dt.date.today():%d %B %Y}  ·  {len(runs)} "
             f"search{'es' if len(runs) != 1 else ''}", size=9.5, colour=STONE, gap=5)
    pdf.grid(["City", "Date", "Request", "Suggestions"],
             [[r.query.city if r.query else "-", r.started_at[:10], r.request,
               ", ".join(p.name for p in r.recommendation.picks) if r.recommendation else "didn't finish"]
              for r in runs], (18, 16, 40, 30))
    for r in runs:
        pdf.add_page()
        _search(pdf, r)
    return bytes(pdf.output())

"""Steps 4 & 5 - find and read each restaurant's FULL menu, then translate it.

Reading (step 4). Restaurants publish menus in many ways - a page, several pages (lunch, dinner,
desserts), a PDF, a scanned PDF, photos, or a page built by JavaScript - so the agent tries them all,
most trustworthy source first, and keeps EVERY page that looks like part of the menu:

  a) The restaurant's own website:
     - the home page, and its links that say "menu" in many languages (menu, carta, carte,
       Speisekarte, menukaart, μενού...), plus the local word the AI supplied for the city;
     - menu pages listed in the site's sitemap.xml (for menus hidden behind a JavaScript button);
     - common addresses such as /menu, /carta, /la-carta, /en/menu, /speisekarte when nothing links;
     - one level deeper: a "Menus" page that only links to Lunch / Dinner / Desserts PDFs is
       followed to those PDFs.
     Every page is read directly first, and with Jina Reader (a real browser) when the direct
     download misses JavaScript-built text.
  b) Photos of the menu: if the site only shows the menu as pictures, or as a scanned PDF, those
     pictures are sent to a vision AI (Gemini or Claude) that writes out every dish and price.
  c) Only if the website has no menu: a web search for "<name> <city> menu", using only results
     that are clearly about THIS restaurant - its own site, or a page for this one restaurant on a
     menu or booking site (TheFork, Tripadvisor, delivery apps...). Lists such as "10 best
     restaurants in Rome" and pages about other places are skipped, and each skip is shown.
     If nothing qualifies, the app says no menu was found rather than guess.

A page counts as menu when it has enough prices - with a currency ("12,50 €") or as a plain number
at the end of a dish line ("Croquetas ..... 9"), as many European menus print them.

Translating (step 5): the AI turns all of that text into a structured menu in the traveller's
language - section, original name, translated name, short description, price and a cautious diet
label - in chunks, so long menus are translated in full rather than cut off.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import urljoin, urlparse

from .llm import LLM
from .models import Candidate, MenuItem, MenuResult, UserQuery
from .schemas import MENU
from .scrape import readable
from .search import WebSearch
from .web import (FetchError, Page, find_menu_images, find_menu_links, menu_score, pdf_images, shrink_image,
                  sitemap_menu_urls)

MIN_SCORE = 4            # a page with fewer prices than this isn't treated as menu
ENOUGH = 40              # stop looking once the pages kept hold this many priced dishes
MAX_PAGES = 9            # pages opened per restaurant website
MENU_PATHS = ["menu", "carta", "la-carta", "menus", "en/menu", "carte", "speisekarte", "menu.pdf", "carta.pdf"]
PHOTO_PROMPT = """These are photos of the menu of {name}, a restaurant in {city}. Write out the menu as plain
text exactly as printed, in its original language: section headings on their own line, then one dish per
line as "Dish name - description - price". Copy every dish and every price you can read; never invent
or translate anything. If a photo is not a food menu, write nothing for it."""

Vision = Callable[[str, list[tuple[bytes, str]]], str]
Fetch = Callable[[str], tuple[bytes, str]]


@dataclass
class _Found:
    url: str
    text: str
    score: int
    how: str


@dataclass
class _Menu:
    """Everything found for one restaurant: all menu-like pages, best first."""
    pages: list[_Found] = field(default_factory=list)
    tried: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    seen: set = field(default_factory=set)

    def offer(self, page: Page, how: str) -> int:
        self.tried.append(page.url + (" (read with Jina Reader)" if page.via == "jina" else ""))
        score = menu_score(page.text)
        if score >= MIN_SCORE and page.url not in {p.url for p in self.pages}:
            kind = "PDF menu" if page.content_type == "pdf" else how
            self.pages.append(_Found(page.url, page.text, score, kind + (" (Jina Reader)" if page.via == "jina" else "")))
            self.pages.sort(key=lambda p: p.score, reverse=True)
        return score

    @property
    def total(self) -> int:
        return sum(p.score for p in self.pages)

    @property
    def enough(self) -> bool:
        return self.total >= ENOUGH

    def text(self, limit: int) -> str:
        """All menu pages joined (best first), repeated lines removed, up to `limit` characters."""
        out, seen_lines, size = [], set(), 0
        for p in self.pages:
            lines = [ln for ln in p.text.splitlines() if ln.strip() and ln.strip().lower() not in seen_lines]
            seen_lines.update(ln.strip().lower() for ln in lines)
            block = f"--- {p.url} ---\n" + "\n".join(lines)
            if size + len(block) > limit:
                block = block[: max(0, limit - size)]
            out.append(block)
            size += len(block)
            if size >= limit:
                break
        return "\n\n".join(out)


def _menu_like(url: str, words: tuple[str, ...]) -> bool:
    low = url.lower()
    return any(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", low)
               for w in ("menu", "menus", "carta", "carte", "speisekarte", "karte", "menukaart") + words)


def _read(read: Callable[..., Page], url: str, m: _Menu, jina: bool = True) -> Optional[Page]:
    if url in m.seen or not readable(url):
        return None
    m.seen.add(url)
    try:
        return read(url, want_prices=True, jina=jina)
    except TypeError:                         # readers that don't take the jina switch (tests)
        try:
            return read(url, want_prices=True)
        except FetchError as e:
            m.notes.append(f"Could not open {url}: {e}")
    except FetchError as e:
        m.notes.append(f"Could not open {url}: {e}")
    return None


def _photos(pages: list[Page], m: _Menu, c: Candidate, q: UserQuery, vision: Optional[Vision],
            fetch: Optional[Fetch], words: tuple[str, ...]) -> None:
    """Menus published only as photos or scanned PDFs: let a vision AI write them out."""
    if vision is None or fetch is None:
        return
    images: list[tuple[bytes, str]] = []
    sources: list[str] = []
    for p in pages:
        if p.content_type == "pdf" and p.raw and menu_score(p.text) < MIN_SCORE:
            found = pdf_images(p.raw)
            if found:
                images += found
                sources.append(p.url)
        elif p.html:
            for u in find_menu_images(p.html, p.url, page_is_menu=_menu_like(p.url, words), extra_words=words):
                try:
                    data, mime = fetch(u)
                except FetchError:
                    continue
                if mime == "application/pdf" or u.lower().endswith(".pdf"):
                    images += pdf_images(data)
                elif mime.startswith("image/"):
                    images.append((data, mime))
                sources.append(u)
        if len(images) >= 4:
            break
    images = [x for x in (shrink_image(d, mt) for d, mt in images[:4]) if x[0]]
    if not images:
        return
    m.notes.append(f"The menu is shown as {len(images)} photo(s); a vision AI is reading them.")
    try:
        text = vision(PHOTO_PROMPT.format(name=c.name, city=q.city), images)
    except Exception as e:  # noqa: BLE001 - photos are a bonus, never a reason to fail
        m.notes.append(f"Couldn't read the menu photos: {e}")
        return
    page = Page(url=sources[0] if sources else c.website, content_type="html", text=text)
    if m.offer(page, "menu photos on their website") >= MIN_SCORE:
        m.notes.append("Read the menu from its photos.")


def read_website(site: str, m: _Menu, read: Callable[..., Page], words: tuple[str, ...] = (),
                 fetch: Optional[Fetch] = None) -> list[Page]:
    """Collect every menu-like page on the restaurant's own site. Returns the pages opened."""
    opened: list[Page] = []
    home = _read(read, site, m)
    if home is None:
        return opened
    opened.append(home)
    m.offer(home, "home page of their website")
    root = f"{urlparse(home.url).scheme}://{urlparse(home.url).hostname}/"

    queue = [u for u, _s, _w in (find_menu_links(home.html, home.url, limit=6, extra_words=words) if home.html else [])]
    if queue:
        m.notes.append("Menu links on the website: " + "; ".join(queue[:6]))
    if fetch is not None:                       # sitemap.xml: menu pages a JavaScript button would open
        try:
            data, _ = fetch(urljoin(root, "sitemap.xml"))
            extra = [u for u in sitemap_menu_urls(data.decode("utf-8", "replace"), words) if u not in queue]
            if extra:
                m.notes.append("Menu pages in the sitemap: " + "; ".join(extra))
            queue += extra
        except (FetchError, UnicodeDecodeError):
            pass
    guessed = not queue
    if guessed:                                 # nothing links to a menu: try the usual addresses
        queue = [urljoin(root, p) for p in MENU_PATHS + [w for w in words if w.isascii()]]

    while queue and len(opened) < MAX_PAGES and not m.enough:
        url = queue.pop(0)
        page = _read(read, url, m, jina=not guessed)        # guessed addresses: no Jina, to save time
        if page is None:
            continue
        opened.append(page)
        score = m.offer(page, "menu page of their website")
        if score < MIN_SCORE and page.html:       # a "Menus" hub page: follow its links one level down
            deeper = [u for u, _s, _w in find_menu_links(page.html, page.url, limit=5, extra_words=words)
                      if u not in m.seen and u not in queue]
            if deeper:
                m.notes.append(f"{url} links to more menu pages: " + "; ".join(deeper))
                queue = deeper + queue
                guessed = False
    return opened


# ------------------------------------------------------------ web search
LISTICLE = re.compile(r"\b(best|top\s*\d+|\d+\s+(best|places|restaurants)|where to eat|guide to|mejores|millors|"
                      r"migliori|meilleurs|beste|melhores|ranking)\b", re.I)


def _tokens(name: str) -> list[str]:
    return [t for t in re.findall(r"\w+", name.lower()) if len(t) >= 4] or [name.lower().strip()]


def about_this_restaurant(title: str, url: str, name: str) -> tuple[bool, str]:
    """Is a search result a page for THIS restaurant (not a list, not another place)? Returns (ok, reason)."""
    text = f"{title} {urlparse(url).path}".lower().replace("-", " ").replace("_", " ")
    if LISTICLE.search(title or ""):
        return False, "a list of several restaurants"
    toks = _tokens(name)
    hits = sum(t in text for t in toks)
    if hits < max(1, (len(toks) + 1) // 2):
        return False, "not about this restaurant"
    return True, ""


def _looks_like_own_site(url: str, name: str) -> bool:
    host = (urlparse(url).hostname or "").replace("www.", "")
    tokens = [t for t in re.findall(r"[a-z0-9]+", name.lower()) if len(t) >= 4]
    return bool(tokens) and any(t in host for t in tokens)


def read_menu(c: Candidate, q: UserQuery, web: WebSearch, read: Callable[..., Page],
              char_limit: int = 24000, vision: Optional[Vision] = None,
              fetch: Optional[Fetch] = None) -> tuple[MenuResult, str]:
    m = _Menu()
    words = tuple(w.lower() for w in q.local_menu_words if w)
    opened: list[Page] = []
    if c.website and readable(c.website):
        opened = read_website(c.website, m, read, words, fetch)
        if m.total < MIN_SCORE:
            _photos(opened, m, c, q, vision, fetch, words)
        if m.total < MIN_SCORE:
            m.notes.append("Their website has no readable menu, so searching the web for this restaurant's menu.")
    else:
        m.notes.append("No website on the map" + (" (only a social-media page)" if c.website else "")
                       + ", so searching the web for this restaurant's menu.")

    if m.total < MIN_SCORE:
        word = q.local_menu_words[0] if q.local_menu_words else ""
        query = f"{c.name} {q.city} menu {word}".strip()
        m.notes.append(f'Web search: "{query}"')
        used = 0
        for h in web.search(query, max_results=8):
            if m.enough or used >= 4:
                break
            if not readable(h.url):
                continue
            host = urlparse(h.url).hostname or ""
            own = _looks_like_own_site(h.url, c.name)
            ok, why = (True, "") if own else about_this_restaurant(h.title, h.url, c.name)
            if not ok:
                m.notes.append(f"Skipped {host}: {why} (\"{h.title[:70]}\")")
                continue
            used += 1
            if own and not (c.website and host in c.website):
                c.website = c.website if (c.website and readable(c.website)) else f"{urlparse(h.url).scheme}://{host}"
                m.notes.append(f"Found the restaurant's website: {host}")
                more = read_website(f"{urlparse(h.url).scheme}://{host}", m, read, words, fetch)
                if m.total < MIN_SCORE:
                    _photos(more, m, c, q, vision, fetch, words)
                continue
            page = _read(read, h.url, m)
            if page is None:
                continue
            if not any(t in page.text.lower() for t in _tokens(c.name)):
                m.notes.append(f"Skipped {host}: the page doesn't mention {c.name}")
                continue
            m.offer(page, f"{host.replace('www.', '')} page for this restaurant")

    found = bool(m.pages)
    best = m.pages[0] if found else None
    if found:
        m.notes.append(f"Kept {len(m.pages)} menu page(s): " + "; ".join(f"{p.url} ({p.score} prices)" for p in m.pages))
    res = MenuResult(menu_url=best.url if best else "", how_found=best.how if best else "none",
                     menu_urls=[p.url for p in m.pages], pages_tried=m.tried,
                     raw_excerpt=best.text[:1500] if best else "",
                     notes="\n".join(m.notes + ([] if found else ["No readable menu with prices was found."])))
    return res, (m.text(char_limit) if found else "")


# ------------------------------------------------------------- translate
TRANSLATE_SYSTEM = """You turn raw restaurant menu text (any language and script, scraped from a website, PDF
or photo, possibly messy) into a clean menu for a traveller, translated into {language}.
- Keep each dish's name exactly as written in original_name (original script too, e.g. Χωριάτικη).
- english_name: a natural name for the dish in {language} (keep well-known dish names, e.g. "Paella"),
  and a short helpful description in {language} - what it is, main ingredients - when the menu gives one
  or the name alone isn't clear to a visitor.
- section: the menu's own section in {language} (Starters, Mains, Pizzas, Desserts, Set lunch menu...),
  in the order the menu uses.
- price: the number as printed, as a plain number (12,50 € -> 12.5; 1.200 Ft -> 1200); null if no price is
  printed for that dish. Never guess. currency: the ISO code of the prices (EUR, GBP, CZK, HUF, CHF...).
- diet: "vegan" only if clearly plant-based; "veg" if no meat, fish or seafood (cheese/egg allowed);
  "non_veg" if it contains meat, fish, seafood or meat stock; "unknown" if you cannot tell.
  Be conservative - travellers rely on this.
- Include EVERY food dish in this text (starters, mains, sides, desserts, set menus) in menu order. Skip
  wine, beer and cocktail lists, opening hours, addresses and allergen tables.
- If the text is not a restaurant menu, return no items and say so in notes."""

CHUNK = 7000
MAX_CHUNKS = 4


def _chunks(text: str, size: int = CHUNK) -> list[str]:
    out, cur = [], ""
    for line in text.splitlines():
        if len(cur) + len(line) + 1 > size and cur:
            out.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        out.append(cur)
    return out[:MAX_CHUNKS]


def translate_menu(raw_text: str, c: Candidate, q: UserQuery, llm: LLM, res: MenuResult) -> MenuResult:
    if not raw_text.strip():
        return res
    lang = q.menu_language or "English"
    items: list[MenuItem] = []
    seen: set[tuple[str, Optional[float]]] = set()
    parts = _chunks(raw_text)
    for i, part in enumerate(parts, 1):
        where = f" (part {i} of {len(parts)})" if len(parts) > 1 else ""
        data = llm.structured(
            system=TRANSLATE_SYSTEM.format(language=lang), name="menu", schema=MENU, max_tokens=12000,
            prompt=f"Restaurant: {c.name} ({q.city}). Expected currency: {q.currency or 'local'}.\n\n"
                   f"Menu text{where}:\n{part}",
        )
        res.detected_language = res.detected_language or data["detected_language"]
        cur = data["currency"] or q.currency
        for it in data["items"]:
            key = (it["original_name"].strip().lower(), it.get("price"))
            if key in seen:
                continue
            seen.add(key)
            items.append(MenuItem(**{**it, "currency": cur}))
        if data["notes"]:
            res.notes = (res.notes + "\n" + data["notes"]).strip()
    res.translated_to = lang
    res.items = items
    return res

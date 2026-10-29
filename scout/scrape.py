"""Reading web pages: a direct download first, Jina Reader as the free scraping backup.

Jina Reader (https://r.jina.ai/<url>) opens a page in a real browser on Jina's
servers and returns it as clean text/markdown. It handles what a plain download
can't: menus built by JavaScript, PDF menus, and many sites that block simple
readers. It's free without a key (about 20 pages a minute); a free key raises
that limit.

Scraper.read(url) decides per page:
  1. Sites known to block simple readers (TripAdvisor, TheFork...) -> Jina only.
  2. Otherwise download directly (fast). If that fails, returns almost no text,
     or - for menus - shows too few prices, ask Jina Reader too and keep
     whichever version is better.
Every Page records how it was read (page.via = "direct" or "jina"), and the UI shows it.
"""
from __future__ import annotations

import re
import threading
from typing import Callable, Optional
from urllib.parse import urlparse

import httpx

from .llm import RateLimiter
from .web import FetchError, Page, check_url, html_to_text, menu_score, safe_get

JINA_URL = "https://r.jina.ai/"
NEVER_READ = ("facebook.", "instagram.", "tiktok.", "x.com", "twitter.", "google.", "youtube.",
              "booking.com", "ubereats.", "deliveroo.", "glovo", "just-eat", "wolt.")
JINA_ONLY = ("tripadvisor.", "yelp.", "thefork.", "lafourchette.", "reddit.com", "opentable.", "quandoo.")
MD_LINK = re.compile(r"\[([^\]]{0,200})\]\((https?://[^)\s]+)\)")
MIN_TEXT = 400        # fewer characters than this = the direct download didn't really get the page
MIN_PRICES = 4


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def readable(url: str) -> bool:
    return not any(b in host_of(url) for b in NEVER_READ)


MD_IMG = re.compile(r"!\[([^\]]{0,200})\]\((https?://[^)\s]+)\)")


def markdown_links_as_html(markdown: str) -> str:
    """Jina returns markdown; turn its links and images into HTML so the link and photo finders work on it."""
    imgs = "".join(f'<img src="{u}" alt="{t}">' for t, u in MD_IMG.findall(markdown))
    return imgs + "".join(f'<a href="{u}">{t}</a>' for t, u in MD_LINK.findall(markdown))


class Scraper:
    def __init__(self, jina_key: str = "", timeout: float = 15.0, use_jina: bool = True,
                 direct: Callable[..., Page] = safe_get, transport: Optional[httpx.BaseTransport] = None):
        self.jina_key = jina_key
        self.timeout = timeout
        self.use_jina = use_jina
        self.direct = direct
        self.transport = transport
        self.limiter = RateLimiter(400 if jina_key else 18)
        self._lock = threading.Lock()
        self.stats = {"direct": 0, "jina": 0, "failed": 0}

    def _count(self, key: str) -> None:
        with self._lock:
            self.stats[key] += 1

    def jina(self, url: str) -> Page:
        check_url(url)
        headers = {"Accept": "application/json", "X-Timeout": str(int(self.timeout) + 10)}
        if self.jina_key:
            headers["Authorization"] = f"Bearer {self.jina_key}"
        self.limiter.wait()
        try:
            with httpx.Client(timeout=self.timeout + 20, transport=self.transport) as c:
                r = c.get(JINA_URL + url, headers=headers)
        except httpx.HTTPError as e:
            raise FetchError(f"Jina Reader: {type(e).__name__}") from e
        if r.status_code == 429:
            raise FetchError("Jina Reader: speed limit reached (a free Jina key raises it)")
        if r.status_code >= 400:
            raise FetchError(f"Jina Reader: HTTP {r.status_code}")
        try:
            data = r.json().get("data") or {}
            text = data.get("content") or ""
            final = data.get("url") or url
        except ValueError:
            text, final = r.text, url
        if "<html" in text[:500].lower():   # occasionally raw HTML comes back
            text = html_to_text(text)
        return Page(url=final, content_type="pdf" if final.lower().endswith(".pdf") else "html",
                    text=text.strip(), html=markdown_links_as_html(text), via="jina")

    def read(self, url: str, want_prices: bool = False, jina: bool = True) -> Page:
        if not readable(url):
            raise FetchError(f"{host_of(url)} can't be read automatically")
        if any(b in host_of(url) for b in JINA_ONLY):
            if not self.use_jina:
                raise FetchError(f"{host_of(url)} blocks simple readers")
            return self._jina_counted(url)

        page, err = None, None
        try:
            page = self.direct(url, timeout=self.timeout)
        except FetchError as e:
            err = e
        weak = page is None or len(page.text) < MIN_TEXT or (want_prices and menu_score(page.text) < MIN_PRICES)
        if weak and self.use_jina and jina:
            try:
                alt = self.jina(url)
                if page is None or _better(alt, page, want_prices):
                    self._count("jina")
                    return alt
            except FetchError as e:
                err = err or e
        if page is None:
            self._count("failed")
            raise err or FetchError(f"Could not read {url}")
        self._count("direct")
        return page

    def _jina_counted(self, url: str) -> Page:
        try:
            p = self.jina(url)
        except FetchError:
            self._count("failed")
            raise
        self._count("jina")
        return p


def _better(a: Page, b: Page, want_prices: bool) -> bool:
    if want_prices:
        pa, pb = menu_score(a.text), menu_score(b.text)
        if pa != pb:
            return pa > pb
    return len(a.text) > len(b.text) * 1.3

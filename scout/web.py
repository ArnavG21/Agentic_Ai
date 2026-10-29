"""Fetching restaurant websites and turning them into readable text.

Kept free of any AI so the logic is deterministic and unit-testable:
- safe_get():          HTTP GET that refuses private/loopback addresses (SSRF guard)
- html_to_text():      strips scripts/nav/footer and returns visible text
- find_menu_links():   scores every link on a page by how likely it is the menu,
                       using menu words in many languages
- count_prices():      how many price-looking strings a text has - our signal that
                       a page really is a menu and not a landing page
"""
from __future__ import annotations

import io
import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0 Safari/537.36 RestaurantScout/1.0"
)

# Words that mean "menu" in languages around the world (the AI adds the local word for each city too).
MENU_WORDS = {
    "menu": 5, "menus": 5, "menú": 5, "menù": 5, "carta": 5, "la-carta": 5, "nuestra-carta": 5,
    "carte": 5, "la-carte": 5, "speisekarte": 5, "karte": 3, "menukaart": 5, "kaart": 3,
    "cardapio": 5, "cardápio": 5, "jídelní": 4, "menü": 5, "etlap": 4, "étlap": 4, "meny": 5,
    "ruokalista": 5, "jadłospis": 5, "karta": 3, "meniu": 5, "μενού": 5, "меню": 5, "תפריט": 5,
    "قائمة": 4, "メニュー": 5, "お品書き": 5, "菜单": 5, "菜單": 5, "메뉴": 5, "เมนู": 5, "thực đơn": 5,
    "food-menu": 5, "our-menu": 5, "food": 2, "eat": 1, "dishes": 2, "platos": 3, "plats": 3,
    "piatti": 3, "gerichte": 3, "menu-del-dia": 4, "menú-del-día": 4, "brunch": 2, "breakfast": 2,
    "desayuno": 2, "lunch": 2, "dinner": 2, "cena": 2, "comida": 2, "thali": 2, "a-la-carte": 4,
}
# Pages that look like menus but usually are not the food menu.
PENALTY_WORDS = {
    "wine": -2, "vinos": -2, "vins": -2, "wein": -2, "drinks": -2, "bebidas": -2, "cocktail": -2,
    "bar-menu": -2, "reserv": -3, "booking": -3, "gift": -3, "voucher": -3, "job": -4, "careers": -4,
    "press": -3, "privacy": -5, "cookie": -5, "legal": -5, "instagram": -5, "facebook": -5,
}

# Prices in any common format: ¥1,200 · 1.200 円 · ₩12,000 · Rp 45.000 · Rs. 350 · AED 45 · R$ 30,90 ·
# 12,50 € · €3 · £7 · $14.99 · 150 บาท · 45 TL ... and, as a last resort, a bare decimal like 18.00.
_NUM = r"(?:\d{1,3}(?:[.,\u00a0\u202f ]\d{3})+(?:[.,]\d{1,2})?|\d{1,6}(?:[.,]\d{1,2})?)"
_SYM = r"[€$£₹¥₩฿₫₺₪₱₦₴₽]"
_CODE_BEFORE = (r"(?<![a-z])(?:rs\.?|inr|rp|idr|aed|sar|qar|usd|us\$|eur|gbp|r\$|rm|myr|s\$|sgd|a\$|aud|"
                r"c\$|cad|hk\$|hkd|nt\$|twd|mx\$|mxn|chf|egp|kes|zar|php|thb|vnd|jpy|cny|rmb|krw|try|mad|"
                r"dh|nzd|nz\$)")
_CODE_AFTER = (r"(?:円|元|원|บาท|đ|(?<![a-z])(?:eur|euros?|kr|kč|zł|chf|ft|lei|tl|lira|yen|baht|dhs?|aed|sar|"
               r"rs|inr|rupees?|rp|vnd|dong|won|yuan|rmb|sek|nok|dkk|pln|czk|huf|ron)(?![a-z]))")
PRICE_RE = re.compile(
    rf"(?:(?:{_SYM}|{_CODE_BEFORE})\s?{_NUM})"        # €12 · ¥1,200 · Rs. 350 · AED 45
    rf"|(?:{_NUM}\s?(?:{_SYM}|{_CODE_AFTER}))"       # 12,50 € · 1.200 円 · 45 TL
    r"|(?:\b\d{1,3}[.,]\d{2}\b)",                   # bare 12.50 as a last resort
    re.IGNORECASE,
)


class FetchError(RuntimeError):
    pass


@dataclass
class Page:
    url: str
    content_type: str
    text: str
    html: str = ""
    via: str = "direct"      # direct download, or "jina" (Jina Reader scraping service)
    raw: bytes = b""         # the file itself for PDFs, so photo-only PDF menus can still be read


def _is_public_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise FetchError(f"Could not look up {host} (no internet, or the site no longer exists)") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def check_url(url: str) -> None:
    """Only public http(s) sites - never addresses inside a private network."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise FetchError(f"Not an http(s) URL: {url}")
    if not _is_public_host(parsed.hostname):
        raise FetchError(f"Refusing non-public host: {parsed.hostname}")


def safe_get(url: str, timeout: float = 15.0, max_bytes: int = 8_000_000) -> Page:
    check_url(url)
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en,*;q=0.5"}
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as c:
            r = c.get(url)
    except httpx.HTTPError as e:
        raise FetchError(f"{type(e).__name__}: {e}") from e
    if r.status_code >= 400:
        raise FetchError(f"HTTP {r.status_code} for {url}")
    body = r.content[:max_bytes]
    ctype = r.headers.get("content-type", "").lower()
    final_url = str(r.url)
    if "pdf" in ctype or final_url.lower().endswith(".pdf"):
        return Page(url=final_url, content_type="pdf", text=pdf_to_text(body), raw=body)
    html = body.decode(r.encoding or "utf-8", errors="replace")
    return Page(url=final_url, content_type="html", text=html_to_text(html), html=html)


def pdf_to_text(data: bytes, max_pages: int = 15) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = reader.pages[:max_pages]
        return "\n".join((p.extract_text() or "") for p in pages).strip()
    except Exception as e:  # malformed PDFs are common on small restaurant sites
        raise FetchError(f"Could not read PDF: {e}") from e


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "header", "nav", "footer", "form"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def count_prices(text: str) -> int:
    return len(PRICE_RE.findall(text or ""))


# A menu line that ends in a bare price, as many European menus print them:
# "Croquetas de jamón ...... 9" · "Patatas bravas 6,5" · "Pizza Margherita 12.00"
DISH_LINE = re.compile(r"^(?=.*[^\W\d_]{3}).{3,90}?(?<!\d)[\s.·…_|-]+(\d{1,3}(?:[.,]\d{1,2})?)\s*(?:€|eur|euros?)?\s*$",
                       re.IGNORECASE | re.MULTILINE)


def menu_score(text: str) -> int:
    """How menu-like a text is: prices with a currency, or dish lines ending in a plain number."""
    text = text or ""
    lines = sum(1 for m in DISH_LINE.finditer(text) if 0 < float(m.group(1).replace(",", ".")) < 500)
    return max(count_prices(text), lines)


MENU_IMG_WORDS = ("menu", "menú", "menù", "carta", "carte", "speisekarte", "karte", "menukaart", "cardapio",
                  "dishes", "platos", "plats", "piatti", "food", "brunch", "lunch", "dinner", "dessert", "postres")
SKIP_IMG_WORDS = ("logo", "icon", "favicon", "sprite", "avatar", "facebook", "instagram", "tripadvisor",
                  "badge", "flag", "arrow", "banner-cookie", "pixel", "loader", "spinner")


def find_menu_images(html: str, base_url: str, page_is_menu: bool = False, limit: int = 4,
                     extra_words: tuple[str, ...] = ()) -> list[str]:
    """Photos of the menu on a page: images whose file name, alt text or class mention the menu. On a page
    that is itself the menu page (e.g. /carta) with little text, its large images are taken too."""
    soup = BeautifulSoup(html or "", "html.parser")
    words = MENU_IMG_WORDS + tuple(w.lower() for w in extra_words if w)
    named, others = [], []
    for img in soup.find_all("img"):
        src = img.get("data-src") or img.get("data-lazy-src") or img.get("src") or ""
        if not src and img.get("srcset"):
            src = img["srcset"].split(",")[-1].strip().split(" ")[0]
        if not src or src.startswith("data:") or src.lower().endswith((".svg", ".gif")):
            continue
        url = urljoin(base_url, src)
        label = f"{src} {img.get('alt', '')} {' '.join(img.get('class', []))} {img.get('title', '')}".lower()
        if any(w in label for w in SKIP_IMG_WORDS):
            continue
        try:
            w = int(str(img.get("width", "0")).rstrip("px") or 0)
        except ValueError:
            w = 0
        if 0 < w < 300:
            continue
        if any(re.search(rf"(?<![a-z]){re.escape(x)}(?![a-z])", label) for x in words):
            named.append(url)
        elif page_is_menu:
            others.append(url)
    out = list(dict.fromkeys(named + others))
    return out[:limit]


def sitemap_menu_urls(xml: str, extra_words: tuple[str, ...] = (), limit: int = 4) -> list[str]:
    """Menu pages listed in a site's sitemap.xml (many sites hide the menu behind a JavaScript button)."""
    words = [w for w in ("menu", "menus", "carta", "carte", "speisekarte", "menukaart", "cardapio", "karte")
             + tuple(w.lower() for w in extra_words if w)]
    urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml or "")
    hits = [u for u in urls if any(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", u.lower()) for w in words)
            and not any(b in u.lower() for b in ("wine", "vino", "drink", "cocktail", "privacy", "cookie"))]
    hits.sort(key=lambda u: (u.lower().endswith(".pdf") is False, len(u)))
    return hits[:limit]


def pdf_images(data: bytes, max_images: int = 4, max_pages: int = 6) -> list[tuple[bytes, str]]:
    """The pictures inside a PDF - for menus that are a scan or a photo saved as PDF."""
    from pypdf import PdfReader

    out: list[tuple[bytes, str]] = []
    try:
        for page in PdfReader(io.BytesIO(data)).pages[:max_pages]:
            for im in page.images:
                if len(out) >= max_images:
                    return out
                if len(im.data) > 20_000:                 # skip logos and decorations
                    name = im.name.lower()
                    mime = "image/png" if name.endswith(".png") else "image/jpeg"
                    out.append((im.data, mime))
    except Exception:  # noqa: BLE001 - odd PDFs are common; just return what we have
        pass
    return out


def fetch_bytes(url: str, timeout: float = 20.0, max_bytes: int = 8_000_000) -> tuple[bytes, str]:
    """Download a file (an image or PDF of the menu). Returns (bytes, content type)."""
    check_url(url)
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent": USER_AGENT}) as c:
            r = c.get(url)
    except httpx.HTTPError as e:
        raise FetchError(f"{type(e).__name__}") from e
    if r.status_code >= 400:
        raise FetchError(f"HTTP {r.status_code}")
    if len(r.content) > max_bytes:
        raise FetchError("file too large")
    return r.content, r.headers.get("content-type", "").split(";")[0].strip().lower()


def shrink_image(data: bytes, mime: str, max_side: int = 1800) -> tuple[bytes, str]:
    """Make a photo small enough for the AI (and turn webp/png into jpeg). Returns it unchanged on failure."""
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(data))
        if im.width < 250 or im.height < 250:
            return b"", mime                                  # too small to be a menu
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        out = io.BytesIO()
        im.save(out, "JPEG", quality=85)
        return out.getvalue(), "image/jpeg"
    except Exception:  # noqa: BLE001
        return data, mime


def find_menu_links(html: str, base_url: str, limit: int = 4,
                    extra_words: tuple[str, ...] = ()) -> list[tuple[str, int, str]]:
    """Return [(absolute_url, score, reason)] for the most menu-like links, best first.
    extra_words: the local word(s) for "menu" in this city's language, from the AI."""
    words = dict(MENU_WORDS)
    for w in extra_words:
        if w and len(w) >= 2:
            words.setdefault(w.lower(), 5)
    soup = BeautifulSoup(html, "html.parser")
    base_host = urlparse(base_url).hostname or ""
    scored: dict[str, tuple[int, str]] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        url = urljoin(base_url, href).split("#")[0]
        host = urlparse(url).hostname or ""
        label = f"{a.get_text(' ', strip=True)} {href}".lower()
        score, hits = 0, []
        for word, w in words.items():
            if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", label):
                score += w
                hits.append(word)
        for word, w in PENALTY_WORDS.items():
            if word in label:
                score += w
        if url.lower().endswith(".pdf") and hits:
            score += 3
            hits.append("pdf")
        if re.search(r"/(en|en-gb|en-us|english)(/|$)", url.lower()):
            score += 1  # English version of the menu page, when the site has one
        # external menu hosts are fine (many restaurants link to a PDF on a CDN or a menu platform)
        if host and base_host and host != base_host and not hits:
            continue
        if score >= 3 and url != base_url:
            prev = scored.get(url)
            if prev is None or score > prev[0]:
                scored[url] = (score, "matched: " + ", ".join(hits))
    ranked = sorted(scored.items(), key=lambda kv: kv[1][0], reverse=True)
    return [(u, s, why) for u, (s, why) in ranked[:limit]]

"""The menu reader against the ways real restaurants publish menus."""
import io

from PIL import Image

from scout.menu import read_menu, translate_menu
from scout.models import Candidate, MenuResult, UserQuery
from scout.search import WebSearch
from scout.web import FetchError, Page, html_to_text, menu_score

Q = UserQuery(city="Barcelona", country_code="ES", currency="EUR", local_menu_words=["carta"])
NO_SEARCH = WebSearch(min_gap=0, backend=lambda q, r, n: [])


def dishes(n, start=0, euro=True):
    return "".join(f"<p>Plato {i} {'<b>' if i % 2 else ''}{8 + i},50{' €' if euro else ''}</p>"
                   for i in range(start, start + n))


def site(pages: dict, pdfs: dict | None = None):
    pdfs = pdfs or {}

    def read(url, **_):
        if url in pdfs:
            return pdfs[url]
        if url not in pages:
            raise FetchError("HTTP 404")
        html = pages[url]
        return Page(url=url, content_type="html", text=html_to_text(html), html=html)
    return read


def jpeg(w=900, h=1200) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (w, h), (250, 245, 230)).save(out, "JPEG")
    return out.getvalue()


def test_plain_number_prices_count_as_a_menu():
    assert menu_score("Croquetas de jamón ...... 9\nPatatas bravas 6,5\nPulpo a la gallega 18.00\nPan con tomate 3") == 4
    assert menu_score("Teléfono 933 123 456\nAbierto de 13:00 a 16:00") == 0


def test_hub_page_is_followed_to_every_menu_and_all_are_kept():
    read = site({
        "https://casa.example": '<a href="/menus">Our menus</a><a href="/reservas">Book</a>',
        "https://casa.example/menus": '<h1>Menus</h1><a href="/menu-lunch">Lunch menu</a><a href="/menu-dinner">Dinner menu</a>',
        "https://casa.example/menu-lunch": dishes(6),
        "https://casa.example/menu-dinner": dishes(7, start=20),
    })
    res, text = read_menu(Candidate(place_id="x", name="Casa", website="https://casa.example"), Q, NO_SEARCH, read)
    assert set(res.menu_urls) == {"https://casa.example/menu-lunch", "https://casa.example/menu-dinner"}
    assert "Plato 0" in text and "Plato 26" in text                     # both menus, not just the best one


def test_sitemap_and_usual_addresses_find_menus_nobody_links_to():
    read = site({"https://a.example": "<p>Welcome</p>", "https://a.example/la-carta": dishes(8),
                 "https://b.example": "<p>Welcome</p>", "https://b.example/carta": dishes(8)})

    def fetch(url):
        if url == "https://a.example/sitemap.xml":
            return b"<urlset><url><loc>https://a.example/la-carta</loc></url></urlset>", "application/xml"
        raise FetchError("HTTP 404")
    res, _ = read_menu(Candidate(place_id="a", name="A", website="https://a.example"), Q, NO_SEARCH, read, fetch=fetch)
    assert res.menu_url == "https://a.example/la-carta" and "sitemap" in res.notes
    res, _ = read_menu(Candidate(place_id="b", name="B", website="https://b.example"), Q, NO_SEARCH, read, fetch=fetch)
    assert res.menu_url == "https://b.example/carta"                       # guessed /carta


def test_menu_shown_only_as_photos_is_read_by_the_vision_ai():
    read = site({"https://foto.example": '<a href="/carta">Carta</a>',
                 "https://foto.example/carta": '<h1>Carta</h1><img src="/img/carta-1.jpg" alt="Nuestra carta">'
                                               '<img src="/img/logo.png" alt="logo">'})
    sent = []

    def vision(prompt, images):
        sent.append(images)
        return "ENTRANTES\nCroquetas de jamón - 9\nPatatas bravas - 6,5\nPulpo a la gallega - 18\nPan con tomate - 3,5"
    fetch = lambda url: (jpeg(), "image/jpeg") if url.endswith("carta-1.jpg") else (_ for _ in ()).throw(FetchError("404"))
    res, text = read_menu(Candidate(place_id="f", name="Foto", website="https://foto.example"), Q, NO_SEARCH, read,
                          vision=vision, fetch=fetch)
    assert len(sent) == 1 and len(sent[0]) == 1 and sent[0][0][1] == "image/jpeg"   # the menu photo, not the logo
    assert res.how_found == "menu photos on their website" and "Croquetas" in text


def test_scanned_pdf_menu_is_read_from_its_pictures():
    buf = io.BytesIO()
    Image.new("RGB", (1000, 1400), (255, 255, 255)).save(buf, "PDF")
    pdf = Page(url="https://scan.example/carta.pdf", content_type="pdf", text="", raw=buf.getvalue())
    read = site({"https://scan.example": '<a href="/carta.pdf">Carta (PDF)</a>'}, {"https://scan.example/carta.pdf": pdf})
    seen = []
    vision = lambda p, imgs: seen.append(len(imgs)) or "Paella 18\nFideuà 17\nCrema catalana 6\nEnsalada 9"
    res, _ = read_menu(Candidate(place_id="s", name="Scan", website="https://scan.example"), Q, NO_SEARCH, read,
                       vision=vision, fetch=lambda u: (_ for _ in ()).throw(FetchError("404")))
    assert seen == [1] and res.menu_urls


def test_long_menus_are_translated_in_full_in_parts():
    calls = []

    class LLM:
        def structured(self, *, prompt, **kw):
            calls.append(prompt)
            n = len(calls)
            return {"detected_language": "Spanish", "currency": "EUR", "notes": "", "items": [
                {"original_name": f"Plato {n}", "english_name": f"Dish {n}", "description": "", "section": "Mains",
                 "price": 10 + n, "diet": "unknown"},
                {"original_name": "Pan", "english_name": "Bread", "description": "", "section": "Sides",
                 "price": 2, "diet": "vegan"}]}
    raw = "\n".join(f"Plato largo número {i} con descripción {i} ..... {i % 40 + 5}" for i in range(500))
    res = translate_menu(raw, Candidate(place_id="x", name="X"), Q, LLM(), MenuResult())
    assert len(calls) > 1 and "part 1 of" in calls[0]
    assert [it.original_name for it in res.items].count("Pan") == 1          # repeated dish kept once
    assert len(res.items) == len(calls) + 1

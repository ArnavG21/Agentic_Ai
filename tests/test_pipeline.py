"""End-to-end run of all nine stages with a fake AI, fake search engine, fake map and fake websites.

Checks the plumbing - stage order, data handed between stages, the website-discovery
fallback, rejected candidates, and fact-checking of the final picks - with no network or keys.
"""
import httpx

from scout.status import SiteCheck
import pytest

from scout.config import Settings
from scout.llm import LLM
from scout.osm import OSM
from scout.pipeline import STAGES, Agent, load_run, save_run
from scout.scrape import Scraper
from scout.search import WebSearch
from scout.web import FetchError, Page, html_to_text

GUIDE_A, GUIDE_B = "https://guide-a.example/best-veg", "https://guide-b.example/veg-bcn"
REVIEW_VV = "https://blog.example/verdura-viva-review"

SITES = {
    GUIDE_A: "<h1>Best vegetarian dinners</h1><p>Verdura Viva: gorgeous vegetable paella.</p><p>Sitio Caído: cosy.</p>",
    GUIDE_B: "<p>Verdura Viva tops our list. Carne Only is great for steak.</p>",
    "https://verdura.example": '<a href="/carta">Nuestra carta</a>',
    "https://verdura.example/carta": "<ul>" + "".join(f"<li>Plato {i} {10 + i},50 €</li>" for i in range(10))
                                     + "<li>Pimientos de Padrón 7 €</li></ul>",
    "https://sitiocaido.example": '<a href="/menu.pdf">Menu</a><a href="/menu">Our menu</a>',
    "https://sitiocaido.example/menu": "<p>" + " ".join(f"Tapa {i} {4 + i} €" for i in range(9)) + "</p>",
    REVIEW_VV: "<p>Loved the Padrón peppers, a very good vegetarian option. Service was slow on Saturday.</p>",
}


def fake_fetch(url, timeout=0, **_):
    if url not in SITES:
        raise FetchError("ConnectError: site down")
    return Page(url=url, content_type="html", text=html_to_text(SITES[url]), html=SITES[url])


def fake_search(query, region, n):
    q = query.lower()
    if "sitio" in q and "menu" in q:
        return [{"title": "Sitio Caído - official", "href": "https://sitiocaido.example/", "body": ""}]
    if "verdura viva" in q and "review" in q:
        return [{"title": "Verdura Viva review", "href": REVIEW_VV, "body": "Great peppers"},
                {"title": "TA", "href": "https://www.tripadvisor.com/vv", "body": "4.6 of 5, slow service"}]
    if "guide" in q or "best" in q:
        return [{"title": "Guide A", "href": GUIDE_A, "body": ""}, {"title": "Guide B", "href": GUIDE_B, "body": ""}]
    return []


def nominatim(req):
    q = req.url.params.get("q", "")
    node = lambda i, name, tags: {"osm_type": "node", "osm_id": i, "lat": "41.38", "lon": "2.17", "type": "restaurant",
                                  "category": "amenity", "name": name, "address": {"road": "Carrer Test", "city": "Barcelona"},
                                  "extratags": tags}
    if q == "Barcelona":
        return httpx.Response(200, json=[{"name": "Barcelona", "boundingbox": ["41.3", "41.5", "2.0", "2.3"],
                                          "address": {"country_code": "es"}}])
    if q.startswith("Verdura Viva"):
        return httpx.Response(200, json=[node(1, "Verdura Viva", {"diet:vegetarian": "only", "website": "https://verdura.example",
                                                                  "opening_hours": "Mo-Su 13:00-23:30"})])
    if q.startswith("Sitio"):
        return httpx.Response(200, json=[node(2, "Sitio Caído", {})])
    if q == "vegetarian restaurant":
        return httpx.Response(200, json=[node(7, "Mapa Verde", {"diet:vegetarian": "yes"}),
                                         {**node(8, "", {}), "name": ""}])
    if q.startswith("Carne Only"):
        return httpx.Response(200, json=[node(3, "Carne Only", {"diet:vegetarian": "no"})])
    return httpx.Response(200, json=[])


class FakeLLM(LLM):
    def __init__(self, city="Barcelona"):
        super().__init__([], [])
        self.city = city

    def structured(self, *, system, prompt, schema, name="result", **_):
        self.usage["calls"] += 1
        self.prompts = getattr(self, "prompts", []) + [(name, system)]
        if name == "open_status":
            closed = "Sitio" in prompt and "cierra" in prompt
            return {"status": "permanently_closed" if closed else "open",
                    "evidence": "Its last service was on 3 May 2026." if closed else "", "source_number": 1}
        if name == "user_query":
            if "python" in prompt.lower():
                return {"on_topic": False, "city": "", "country_code": "", "meal_type": "any", "diet": "any",
                        "budget_min": None, "budget_max": None, "budget_estimated": False, "budget_note": "",
                        "currency": "", "cuisines": [], "requirements": [], "local_language": "",
                        "local_menu_words": [], "review_keywords": [], "search_queries": []}
            return {"on_topic": True, "city": self.city, "country_code": "ES", "meal_type": "dinner", "diet": "veg",
                    "budget_min": None, "budget_max": 35, "budget_estimated": False, "budget_note": "", "currency": "EUR", "cuisines": [],
                    "requirements": ["quiet"], "local_language": "Spanish/Catalan", "local_menu_words": ["carta"],
                    "review_keywords": ["vegetarian", "vegetariano", "quiet", "tranquilo"],
                    "search_queries": ["best vegetarian dinner Barcelona", "vegetarian Barcelona guide"]}
        if name == "guide_restaurants":
            assert "[1] Guide A" in prompt
            r = lambda n, src: {"name": n, "area": "", "rating": None, "price_per_person": None, "why": "recommended", "source_numbers": src}
            return {"restaurants": [r("Verdura Viva", [1, 2]), r("Sitio Caído", [1]), r("Carne Only", [2]),
                                    r("Ghost Place", [99])]}
        if name == "menu":
            return {"detected_language": "Spanish", "currency": "EUR", "notes": "", "items": [
                {"original_name": "Pimientos de Padrón", "english_name": "Padrón peppers", "description": "",
                 "section": "Starters", "price": 7, "diet": "vegan"},
                {"original_name": "Plato 1", "english_name": "Dish 1", "description": "", "section": "Mains",
                 "price": 11.5, "diet": "veg"},
                {"original_name": "Croquetas de setas", "english_name": "Mushroom croquettes", "description": "",
                 "section": "Starters", "price": 8.5, "diet": "veg"},
                {"original_name": "Escalivada", "english_name": "Roasted vegetables", "description": "",
                 "section": "Starters", "price": 9, "diet": "vegan"},
                {"original_name": "Arroz negro", "english_name": "Black rice", "description": "",
                 "section": "Mains", "price": 16, "diet": "non_veg"}]}
        if name == "review_balance":
            return {"positives": [{"point": "Great peppers", "source_url": REVIEW_VV}],
                    "negatives": [{"point": "Slow service", "source_url": "https://made-up.example"}],
                    "popular_dishes": ["Padron peppers"]}
        if name == "context_check":
            return {"context": [{"requirement": "quiet", "verdict": "mixed", "evidence": "Busy on weekends", "source_url": REVIEW_VV}]}
        if name == "recommendation":
            ids = [line.split('"')[3] for line in prompt.splitlines() if '"place_id"' in line]
            return {"summary": "Verdura Viva is fully vegetarian and the cheapest of the three.", "picks": [
                {"place_id": ids[0], "name": "x", "headline": "Veg-friendly",
                 "for_you": [{"need": "Quiet", "insight": "Reviewers call the back room calm on weekdays."},
                             {"need": "", "insight": "dropped: no label"}],
                 "good_to_know": ["Book ahead on Fridays.", ""],
                 "why_it_fits": ["menu read"], "watch_outs": ["slow service"], "estimated_spend": "€25-35",
                 "what_to_order": [{"item": "Padrón peppers", "why": "most praised"},
                                   {"item": "Truffle risotto", "why": "invented by the model"}]},
                {"place_id": "ghost", "name": "Not shortlisted", "headline": "", "for_you": [], "good_to_know": [],
                 "why_it_fits": [], "watch_outs": [], "estimated_spend": "", "what_to_order": []}]}
        raise AssertionError(f"unexpected schema {name}")


def no_download(url):
    raise FetchError("offline in tests")


def make_agent(llm):
    scraper = Scraper(direct=fake_fetch, use_jina=False)
    return Agent(settings=Settings(gemini_key="x", shortlist_size=3), llm=llm,
                 web=WebSearch(min_gap=0, backend=fake_search, fetch=scraper.read),
                 osm=OSM(transport=httpx.MockTransport(nominatim), min_gap=0), scraper=scraper,
                 probe=lambda url: SiteCheck("ok", text="Welcome! Open every day."),
                 fetch_bytes=no_download)


@pytest.fixture
def agent():
    return make_agent(FakeLLM())


def test_missing_city_asks_instead_of_failing():
    events = list(make_agent(FakeLLM(city="")).run("veg dinner somewhere nice"))
    st = events[-1].payload
    assert st.needs_info and "Which city" in st.error
    assert [e.stage for e in events if e.kind == "start"] == ["understand"]


def test_full_run(agent, tmp_path):
    events = list(agent.run("veg dinner in Barcelona, quiet, under 35 euros"))
    assert [e.stage for e in events if e.kind == "start"] == [k for k, _, _ in STAGES]
    assert not [e for e in events if e.kind == "error"], [e.message for e in events if e.kind == "error"]
    st = events[-1].payload

    # discovery: 4 restaurants from guides, invented source number dropped, 3 confirmed on the map
    by_name = {c.name: c for c in st.candidates}
    assert set(by_name) == {"Verdura Viva", "Sitio Caído", "Carne Only", "Ghost Place"}
    assert len(by_name["Verdura Viva"].mentions) == 2 and by_name["Ghost Place"].mentions == []
    assert by_name["Verdura Viva"].on_map and not by_name["Ghost Place"].on_map

    # shortlist: the map says Carne Only has no vegetarian food -> rejected with a reason
    carne = next(r for r in st.shortlist_rows if r.name == "Carne Only")
    assert not carne.passed and any("vegetarian" in x for x in carne.reasons)
    names = [d.candidate.name for d in st.dossiers]
    assert names[0] == "Verdura Viva" and "Carne Only" not in names

    vv = st.dossiers[0]
    assert vv.menu.how_found == "menu page of their website" and vv.menu.menu_url == "https://verdura.example/carta"
    assert vv.menu.items[0].english_name == "Padrón peppers"

    # no website on the map -> found by web search, adopted, and its menu read
    sc = next(d for d in st.dossiers if d.candidate.name == "Sitio Caído")
    assert sc.candidate.website == "https://sitiocaido.example"
    assert sc.menu.menu_url == "https://sitiocaido.example/menu"

    # review evidence: invented URLs blanked, real ones kept
    assert vv.reviews.positives[0].source_url == REVIEW_VV
    assert vv.reviews.negatives[0].source_url == ""
    assert vv.reviews.context[0].verdict == "mixed"

    # fact-check: ghost pick and invented dish dropped, price taken from the menu
    rec = st.recommendation
    assert [p.name for p in rec.picks] == ["Verdura Viva"]
    assert [o.item for o in rec.picks[0].what_to_order] == ["Padrón peppers (Pimientos de Padrón)"]
    assert rec.picks[0].what_to_order[0].price == 7
    assert any("Truffle risotto" in c for c in st.checks)

    assert load_run(save_run(st, tmp_path)).recommendation.picks[0].name == "Verdura Viva"


def test_every_pick_shows_website_and_menu(agent):
    from scout.links import all_links
    st = list(agent.run("veg dinner in Barcelona"))[-1].payload
    by_id = {d.candidate.place_id: d for d in st.dossiers}
    for p in st.recommendation.picks:
        site, menu, maps = all_links(by_id[p.place_id], "Barcelona")
        assert site.found and site.url == "https://verdura.example"
        assert menu.found and menu.url == "https://verdura.example/carta"
        assert by_id[p.place_id].menu.items          # the translated menu is there to show


def test_cost_per_person_is_calculated_from_the_menu(agent):
    st = list(agent.run("veg dinner in Barcelona under 35 euros"))[-1].payload
    vv = next(d for d in st.dossiers if d.candidate.name == "Verdura Viva")
    # prices 7, 8.5, 9, 11.5, 16 -> typical dish 8.5-11.5 -> 2 dishes = 17-23 per person, under the 35 limit
    assert (vv.cost.low, vv.cost.high, vv.cost.fit) == (17.0, 23.0, "fits")
    assert st.recommendation.picks[0].estimated_spend == "17–23 € per person"   # code's number, not the AI's


def test_menus_are_translated_into_the_chosen_language():
    llm = FakeLLM()
    st = list(make_agent(llm).run("veg dinner in Barcelona", menu_language="Hindi"))[-1].payload
    menu_systems = [sysmsg for name, sysmsg in llm.prompts if name == "menu"]
    assert menu_systems and all("translated into Hindi" in x for x in menu_systems)
    assert st.query.menu_language == "Hindi" and st.dossiers[0].menu.translated_to == "Hindi"


def test_where_the_data_came_from(agent):
    from scout.sources import source_rows, summarise, to_dot
    st = list(agent.run("veg dinner in Barcelona"))[-1].payload
    m = summarise(st)
    assert (m.restaurants_found, m.on_map, m.menus_read, m.suggestions) == (4, 3, 2, 1)
    assert "guide-a.example" in m.guide_domains and m.review_pages >= 1
    dot = to_dot(m)
    assert dot.startswith("digraph") and "OpenStreetMap" in dot and "Your 1 suggestion<" in dot
    rows = {r["Website"]: r for r in source_rows(st)}
    assert rows["verdura.example"]["Used for"] == "menu" and rows["guide-a.example"]["Used for"] == "finding restaurants"


def test_pdf_is_a_clean_document_with_every_link(agent):
    from pypdf import PdfReader
    import io
    from scout.pdf_report import build_history_pdf, build_pdf
    st = list(agent.run("Vegetarian dinner in Barcelona & somewhere quiet"))[-1].payload
    pdf = PdfReader(io.BytesIO(build_pdf(st)))
    text = " ".join(p.extract_text() for p in pdf.pages)
    for words in ("Where to eat in Barcelona", "Verdura Viva", "Padrón peppers", "17–23 € per person",
                  "translated from Spanish", "The menu", "For what you asked", "Reviewers call the back room calm",
                  "Good to know", "How this was put together"):
        assert words in text, words
    links = {a["/A"]["/URI"] for p in pdf.pages for a in [x.get_object() for x in p.get("/Annots", [])]
             if "/A" in a and "/URI" in a["/A"]}
    assert {"https://verdura.example", "https://verdura.example/carta"} <= links
    assert any(u.startswith("https://www.google.com/maps/search/") for u in links)
    assert "no evidence" not in text.lower() and "verdict" not in text.lower()
    history = PdfReader(io.BytesIO(build_history_pdf([st, st])))
    assert "Your restaurant searches" in history.pages[0].extract_text() and len(history.pages) > len(pdf.pages)


def test_closed_restaurant_is_skipped_and_replaced():
    news = "https://news.example/sitio-caido-cierra"

    def search_with_news(query, region, n):
        if "sitio" in query.lower() and ("closed" in query.lower() or "cierra" in query.lower()):
            return [{"title": "Cierra Sitio Caído, uno de los grandes de Barcelona", "href": news,
                     "body": "Sitio Caído hará su último servicio el 3 de mayo."}]
        return fake_search(query, region, n)

    agent = make_agent(FakeLLM())
    agent.web._backend = search_with_news
    st = list(agent.run("veg dinner in Barcelona"))[-1].payload
    row = next(r for r in st.shortlist_rows if r.name == "Sitio Caído")
    assert row.closed and not row.chosen and row.closed_source == news
    assert row.reasons[0].startswith("✗ Closed for good") and "3 May 2026" in row.reasons[0]
    names = [d.candidate.name for d in st.dossiers]
    assert "Sitio Caído" not in names and "Ghost Place" in names          # the next place took its spot
    assert any("closed for good" in line for line in st.stage_logs["shortlist"])


def test_reviews_saying_closed_remove_the_place_from_suggestions(agent):
    from scout import reviews as rv
    original = rv.summarise_balance

    def closed_balance(c, ev, llm, ins):
        out = original(c, ev, llm, ins)
        if c.name == "Verdura Viva":
            out.closed_note, out.closed_source = "A review source says it has closed for good.", REVIEW_VV
        return out
    rv.summarise_balance = closed_balance
    try:
        st = list(agent.run("veg dinner in Barcelona"))[-1].payload
    finally:
        rv.summarise_balance = original
    assert all(p.name != "Verdura Viva" for p in st.recommendation.picks)
    assert any("Left out Verdura Viva" in c for c in st.checks)


def test_cities_outside_europe_are_politely_refused():
    llm = FakeLLM(city="Tokyo")
    llm_structured = llm.structured

    def jp(**kw):
        out = llm_structured(**kw)
        if kw.get("name") == "user_query":
            out["country_code"] = "JP"
        return out
    llm.structured = jp
    st = list(make_agent(llm).run("ramen in Tokyo"))[-1].payload
    assert st.needs_info and "Europe" in st.error and "Barcelona" in st.error


def test_off_topic_requests_are_turned_away():
    events = list(make_agent(FakeLLM()).run("write me a python script to sort a list"))
    st = events[-1].payload
    assert st.needs_info and "only help you find restaurants" in st.error
    assert [e.stage for e in events if e.kind == "start"] == ["understand"]
    st = list(make_agent(FakeLLM()).run("x" * 700))[-1].payload          # too long: refused before any AI call
    assert st.needs_info and "under 600 characters" in st.error
    st = list(make_agent(FakeLLM()).run("???"))[-1].payload
    assert st.needs_info


def test_off_topic_follow_ups_are_turned_away():
    from types import SimpleNamespace as NS
    from scout.models import UserQuery
    from scout.recommend import FOLLOWUP_OFF_TOPIC, answer_followup
    q = UserQuery(city="Barcelona", country_code="ES")
    seen = []
    off = NS(chat=lambda system, messages: seen.append(system) or "OFF_TOPIC")
    assert answer_followup("what's the capital of France?", [], q, [], None, off) == FOLLOWUP_OFF_TOPIC
    assert "only answer questions about these restaurants" in seen[0]
    on = NS(chat=lambda system, messages: "Verdura Viva has Padrón peppers for 7 €.")
    assert answer_followup("cheapest dish?", [], q, [], None, on).startswith("Verdura Viva")


def test_when_web_search_finds_nothing_the_map_is_used():
    agent = make_agent(FakeLLM())
    agent.web = WebSearch(min_gap=0, backend=lambda q, r, n: [], fetch=agent.scraper.read)   # search blocked
    st = list(agent.run("veg dinner in Barcelona"))[-1].payload
    logs = " ".join(st.stage_logs["discover"])
    assert "web searches returned nothing" in logs and "Used OpenStreetMap instead" in logs
    assert [c.name for c in st.candidates] == ["Mapa Verde"] and st.candidates[0].on_map


def test_when_guide_places_are_closed_more_come_from_the_map(monkeypatch):
    from scout import pipeline, status
    real = status.check_open

    def fake_check(c, *a, **kw):
        if c.name != "Mapa Verde":
            return status.OpenStatus(True, "Google Maps lists it as permanently closed.", "https://maps.example")
        return real(c, *a, **kw)
    monkeypatch.setattr(pipeline.status, "check_open", fake_check)
    st = list(make_agent(FakeLLM()).run("veg dinner in Barcelona, quiet, under 35 euros"))[-1].payload
    assert [d.candidate.name for d in st.dossiers] == ["Mapa Verde"]
    assert any("more came from OpenStreetMap" in x for x in st.stage_logs["shortlist"])
    assert st.recommendation and [p.name for p in st.recommendation.picks] == ["Mapa Verde"]

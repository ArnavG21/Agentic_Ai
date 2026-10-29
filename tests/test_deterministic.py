"""Unit tests: hours, map tags, scoring, menus, search, scraping, AI fallbacks, links, fact-checks.
No network and no keys - every outside service is faked."""
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError, RateLimitError

from scout.config import Settings
from scout.hours import open_for_meal
from scout.links import all_links
from scout.llm import LLM, LLMError, RateLimiter, Route
from scout.models import Candidate, Dossier, MenuItem, MenuResult, Mention, ReviewInsights, ShortlistRow, UserQuery
from scout.osm import OSM, City, diet_flag
from scout.recommend import match_menu_item
from scout.money import budget_text, estimate_cost, fmt_price, fmt_range
from scout.reviews import keyword_hits, regular_items
from scout.scrape import Scraper, markdown_links_as_html
from scout.search import Evidence, GroqSearch, WebSearch, parse_executed_tools, parse_result_lines, region_for
from scout.shortlist import shortlist
from scout.web import FetchError, Page, count_prices, find_menu_links, html_to_text


# ---------------------------------------------------------------- hours & tags
@pytest.mark.parametrize("oh, meal, expected", [
    ("Mo-Fr 08:00-16:00; Sa,Su 10:00-23:30", "breakfast", True),
    ("Tu-Su 13:00-16:00,20:00-00:00; Mo off", "dinner", True),
    ("Mo-Su 12:00-16:00", "dinner", False),
    ("Mo-Su 19:00-01:00", "dinner", True),       # closes after midnight
    ("24/7", "lunch", True),
    ("sunrise-sunset", "lunch", None),            # unreadable -> unknown, not a wrong answer
    ("", "dinner", None),
])
def test_open_for_meal(oh, meal, expected):
    assert open_for_meal(oh, meal) is expected


def test_diet_flag():
    assert diet_flag({"diet:vegetarian": "yes"}, "veg") is True
    assert diet_flag({"cuisine": "vegan;regional"}, "veg") is True
    assert diet_flag({"diet:vegetarian": "no"}, "veg") is False
    assert diet_flag({}, "veg") is None
    assert diet_flag({"diet:vegetarian": "yes"}, "vegan") is None   # vegetarian doesn't imply vegan


# ------------------------------------------------------------------- shortlist
def cand(pid, guides=2, **kw):
    mentions = [Mention(url=f"https://guide{i}.example/{pid}", why="great vegetarian dinner") for i in range(guides)]
    return Candidate(place_id=pid, name=pid, mentions=mentions, on_map=True, **kw)


def test_shortlist_filters_with_reasons():
    q = UserQuery(city="Barcelona", diet="veg", meal_type="dinner", budget_max=35, currency="EUR",
                  review_keywords=["vegetarian"])
    cands = [
        cand("good", guides=3, tags={"diet:vegetarian": "yes"}, opening_hours="Mo-Su 13:00-23:00", website="https://a.es"),
        cand("no_veg", tags={"diet:vegetarian": "no"}),
        cand("lunch_only", opening_hours="Mo-Su 12:00-16:00"),
        cand("low", rating_hint=3.2),
        cand("pricey", guide_price=60),
        cand("unknowns", guides=1),
    ]
    rows, top = shortlist(cands, q, size=4)
    by = {r.place_id: r for r in rows}
    assert [c.place_id for c in top] == ["good", "unknowns"]          # most food guides first
    for bad in ("no_veg", "lunch_only", "low", "pricey"):
        assert not by[bad].passed and not by[bad].chosen and any(r.startswith("✗") for r in by[bad].reasons)
    assert by["good"].chosen and by["good"].guides == 3
    assert any(r.startswith("?") for r in by["unknowns"].reasons)   # unknowns are flagged, not rejected


def test_shortlist_order_is_simply_guide_count_then_map_then_website():
    q = UserQuery(city="x")
    cands = [cand("one_guide", guides=1, website="https://a"), cand("three_guides", guides=3),
             cand("two_no_map", guides=2), cand("two_on_map_site", guides=2, website="https://b")]
    cands[2].on_map = False
    _, top = shortlist(cands, q, size=3)
    assert [c.place_id for c in top] == ["three_guides", "two_on_map_site", "two_no_map"]


# ----------------------------------------------------------------------- menus
HOME = """<html><body><nav><a href="/reservas">Reservas</a></nav>
<a href="/es/nuestra-carta">Nuestra carta</a><a href="/en/menu">Menu</a><a href="/vinos">Carta de vinos</a>
<a href="/files/carta-2026.pdf">Descarga la carta (PDF)</a><a href="https://instagram.com/x">Instagram</a>
<a href="/aviso-legal">Aviso legal</a></body></html>"""


def test_find_menu_links_multilingual():
    urls = [u for u, _, _ in find_menu_links(HOME, "https://rest.es/")]
    assert set(urls[:3]) == {"https://rest.es/es/nuestra-carta", "https://rest.es/files/carta-2026.pdf",
                             "https://rest.es/en/menu"}
    assert all("instagram" not in u and "aviso-legal" not in u and "reservas" not in u for u in urls)
    assert urls[-1] == "https://rest.es/vinos"   # wine list ranks below every food menu


def test_prices_and_text():
    text = html_to_text("<script>x=1</script><p>Croquetas 8,50 €</p><p>Pan con tomate €3</p><p>Paella 18.00</p>")
    assert "x=1" not in text and count_prices(text) == 3
    assert count_prices("Tapa 4 € · Tapa 5 € · 12,50€ · 8 eur · €3 · £7") == 6   # symbol after the number too


def test_regular_items_filters_diet_and_flags_popular():
    menu = MenuResult(items=[
        MenuItem(original_name="Paella de verduras", english_name="Vegetable paella", price=18, diet="veg"),
        MenuItem(original_name="Jamón ibérico", english_name="Iberian ham", price=22, diet="non_veg"),
        MenuItem(original_name="Pimientos de Padrón", english_name="Padrón peppers", price=7, diet="vegan"),
    ])
    items = regular_items(menu, ReviewInsights(popular_dishes=["padron peppers"]), "veg")
    assert [i.english_name for i in items] == ["Padrón peppers", "Vegetable paella"] and items[0].popular


def test_match_menu_item_rejects_invented_dishes():
    items = [MenuItem(original_name="Paella de verduras", english_name="Vegetable paella", price=18)]
    assert match_menu_item("vegetable paella", items).price == 18
    assert match_menu_item("Paella de Verduras", items) is not None
    assert match_menu_item("Truffle risotto", items) is None


def test_fmt_price():
    assert fmt_price(7, "EUR") == "7 €" and fmt_price(12.5, "GBP") == "£12.50"
    assert fmt_price(None, "EUR") == "" and fmt_price(350, "INR") == "₹350"


# ---------------------------------------------------------------------- search
def test_research_numbers_sources_and_skips_unreadable_sites():
    rows = [{"title": "IG", "href": "https://www.instagram.com/x", "body": "photo"},
            {"title": "Blog", "href": "https://blog.example/veg", "body": "snippet"},
            {"title": "Blog", "href": "https://blog.example/veg", "body": "dup"}]
    fetched = []

    def fetch(url, want_prices=False):
        fetched.append(url)
        return Page(url=url, content_type="html", text="Full blog text about vegetarian food", via="jina")

    ev = WebSearch(min_gap=0, backend=lambda q, r, n: rows, fetch=fetch).research(["q1"], pages=3)
    assert [s["n"] for s in ev.sources] == [1, 2]
    assert fetched == ["https://blog.example/veg"]              # instagram kept as a preview only
    assert ev.sources[1]["via"] == "jina" and "Full blog text" in ev.as_prompt()


def test_search_survives_backend_errors(monkeypatch):
    import scout.search as s
    monkeypatch.setattr(s.time, "sleep", lambda *_: None)

    def boom(q, r, n):
        raise RuntimeError("rate limited")
    assert WebSearch(min_gap=0, backend=boom).search("x") == []


def test_fallback_search_used_only_when_duckduckgo_is_empty():
    backup = lambda q, r, n: [{"title": "G", "href": "https://g.example", "body": "from groq"}]
    web = WebSearch(min_gap=0, backend=lambda q, r, n: [], fallback=backup)
    assert [h.url for h in web.search("x")] == ["https://g.example"] and web.fallback_count == 1
    web2 = WebSearch(min_gap=0, backend=lambda q, r, n: [{"title": "D", "href": "https://d.example"}], fallback=backup)
    assert [h.url for h in web2.search("x")] == ["https://d.example"] and web2.fallback_count == 0


def test_parse_groq_results():
    tools = [{"type": "browser_search", "search_results": {"results": [
        {"title": "Best veg", "url": "https://guide.example/veg", "content": "Top places"}]}}]
    assert parse_executed_tools(tools) == [{"title": "Best veg", "href": "https://guide.example/veg", "body": "Top places"}]
    text = "Best veg ||| https://guide.example/veg ||| Top places\nnoise line\n- Other ||| <https://b.example/x>. ||| y"
    assert [r["href"] for r in parse_result_lines(text)] == ["https://guide.example/veg", "https://b.example/x"]


def test_keyword_hits_on_review_text():
    ev = Evidence(sources=[{"n": 1, "url": "u", "title": "", "snippet": "", "text": "Muy buena opción vegetariana y tranquila"}])
    assert {h["keyword"] for h in keyword_hits(ev, ["vegetariana", "tranquila", "kids"])} == {"vegetariana", "tranquila"}


# -------------------------------------------------------------------- scraping
JINA_MENU = "# Carta\n" + "\n".join(f"- Plato {i} ... {9 + i} €" for i in range(9)) + "\n[Postres](https://rest.es/postres)"


def jina_transport(seen):
    def handler(req):
        seen.append(str(req.url))
        return httpx.Response(200, json={"data": {"content": JINA_MENU, "url": "https://rest.es/carta", "title": "Carta"}})
    return httpx.MockTransport(handler)


@pytest.fixture
def public(monkeypatch):
    import scout.web as w
    monkeypatch.setattr(w, "_is_public_host", lambda host: True)


def test_scraper_uses_jina_when_direct_page_has_no_prices(public):
    seen = []
    js_shell = Page(url="https://rest.es/carta", content_type="html", text="Loading… " * 80, html="<div id=app></div>")
    sc = Scraper(direct=lambda url, timeout=0: js_shell, transport=jina_transport(seen))
    page = sc.read("https://rest.es/carta", want_prices=True)
    assert page.via == "jina" and count_prices(page.text) == 9
    assert seen == ["https://r.jina.ai/https://rest.es/carta"] and sc.stats["jina"] == 1
    assert 'href="https://rest.es/postres"' in page.html          # links still usable for the menu finder


def test_scraper_keeps_direct_page_when_it_is_good(public):
    seen = []
    good = Page(url="https://rest.es/carta", content_type="html", text=JINA_MENU * 3)
    sc = Scraper(direct=lambda url, timeout=0: good, transport=jina_transport(seen))
    assert sc.read("https://rest.es/carta", want_prices=True).via == "direct" and seen == []


def test_scraper_sends_blocking_sites_to_jina_and_never_reads_social(public):
    seen = []

    def direct(url, timeout=0):
        raise AssertionError("should not download directly")
    sc = Scraper(direct=direct, transport=jina_transport(seen))
    assert sc.read("https://www.tripadvisor.com/Restaurant_Review-x").via == "jina"
    with pytest.raises(FetchError):
        sc.read("https://www.instagram.com/somewhere")


def test_scraper_falls_back_to_jina_when_direct_fails(public):
    def direct(url, timeout=0):
        raise FetchError("HTTP 403")
    assert Scraper(direct=direct, transport=jina_transport([])).read("https://rest.es/").via == "jina"


def test_markdown_links_as_html():
    assert markdown_links_as_html("see [Our menu](https://a.es/menu) now") == '<a href="https://a.es/menu">Our menu</a>'


# ------------------------------------------------------------------------- osm
def test_osm_find_place_inside_city():
    seen = {}

    def handler(req):
        seen.update(dict(req.url.params))
        return httpx.Response(200, json=[{"osm_type": "node", "osm_id": 42, "lat": "41.38", "lon": "2.17",
                                          "type": "restaurant", "category": "amenity", "name": "Teresa Carles",
                                          "address": {"road": "Carrer de Jovellanos", "house_number": "2", "city": "Barcelona"},
                                          "extratags": {"diet:vegetarian": "only", "website": "https://teresacarles.com"}}])
    p = OSM(transport=httpx.MockTransport(handler), min_gap=0).find_place("Teresa Carles", City("Barcelona", "ES", 41.3, 41.5, 2.0, 2.3))
    assert p.website == "https://teresacarles.com" and p.osm_url.endswith("/node/42")
    assert seen["bounded"] == "1" and seen["viewbox"] == "2.0,41.5,2.3,41.3"


# ---------------------------------------------------------- settings & links
def test_keys_go_to_the_right_service_whatever_box_they_are_in():
    s = Settings()
    for k in ("gsk_abc", "AIzaXYZ", "jina_123"):
        s.add_key(k)
    assert (s.gemini_key, s.groq_key, s.jina_key) == ("AIzaXYZ", "gsk_abc", "jina_123")
    env = {"LLM_API_KEY": "gsk_only", "GROQ_API_KEY": ""}
    s2 = Settings.from_env(get=env.get)
    assert s2.groq_key == "gsk_only" and s2.has_ai


def test_links_are_only_real_pages_never_a_search():
    row = ShortlistRow(place_id="x", name="Casa", passed=True)
    bare = Dossier(candidate=Candidate(place_id="x", name="Casa Test"), shortlist=row)
    links = all_links(bare, "Barcelona")
    assert [ln.label for ln in links] == ["Google Maps"] and links[0].url.startswith("https://www.google.com/maps")
    assert not any("google.com/search" in ln.url for ln in links)
    full = Dossier(candidate=Candidate(place_id="x", name="Casa", website="https://casa.es"), shortlist=row,
                   menu=MenuResult(menu_url="https://casa.es/carta"))
    site, menu, _ = all_links(full, "Barcelona")
    assert (site.url, menu.url, menu.label) == ("https://casa.es", "https://casa.es/carta", "Original menu")


# ------------------------------------------------------------------------- llm
def _resp(text, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


class FakeCompletions:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def route(provider, model, replies, **kw):
    comp = FakeCompletions(replies)
    client = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    return Route(provider, model, client, RateLimiter(100_000), **kw), comp


REQ = httpx.Request("POST", "http://x")


def test_structured_parses_fenced_json_and_asks_for_low_thinking():
    r, comp = route("Gemini", "gemini-3.5-flash-lite", [_resp('```json\n{"a": 1}\n```')])
    llm = LLM([r], [r], sleep=lambda *_: None)
    assert llm.structured(system="s", prompt="p", schema={"type": "object"}) == {"a": 1}
    call = comp.calls[0]
    assert call["response_format"]["json_schema"]["strict"] is True and call["reasoning_effort"] == "low"


def test_timeout_moves_to_next_model_and_stays_there():
    slow, slow_calls = route("Gemini", "gemini-3.8-flash", [APITimeoutError(request=REQ), APITimeoutError(request=REQ)])
    ok, ok_calls = route("Gemini", "gemini-3.6-flash", [_resp('{"x": 1}'), _resp('{"x": 2}')])
    llm = LLM([slow, ok], [slow, ok], sleep=lambda *_: None)
    assert llm.structured(system="s", prompt="p", schema={}, fast=False) == {"x": 1}
    assert llm.structured(system="s", prompt="p", schema={}, fast=False) == {"x": 2}
    assert len(slow_calls.calls) == 2 and len(ok_calls.calls) == 2          # slow model not retried again
    assert llm.main_label == "gemini-3.6-flash (Gemini)" and "timed out" in llm.notes[0]


def test_bad_key_skips_provider_and_uses_groq_with_short_prompt():
    bad = APIStatusErrorFake(401)
    g1, _ = route("Gemini", "gemini-3.8-flash", [bad])
    g2, g2c = route("Gemini", "gemini-3.6-flash", [])
    groq, groq_calls = route("Groq", "openai/gpt-oss-120b", [_resp('{"ok": true}')], max_prompt_chars=3000)
    llm = LLM([g1, g2, groq], [g1, g2, groq], sleep=lambda *_: None)
    assert llm.structured(system="s", prompt="x" * 10_000, schema={}) == {"ok": True}
    assert g2c.calls == []                                                   # same bad key not tried twice
    assert len(groq_calls.calls[0]["messages"][1]["content"]) < 3100


def test_rate_limit_retries_then_succeeds():
    err = RateLimitError("slow down", response=httpx.Response(429, request=REQ), body=None)
    r, comp = route("Gemini", "m", [err, _resp('{"ok": 1}')])
    assert LLM([r], [r], sleep=lambda *_: None).structured(system="s", prompt="p", schema={}) == {"ok": 1}
    assert len(comp.calls) == 2


def test_schema_rejected_falls_back_to_json_mode():
    r, comp = route("Groq", "m", [APIStatusErrorFake(400, "response_format json_schema not supported"), _resp('{"a": 2}')])
    assert LLM([r], [r], sleep=lambda *_: None).structured(system="s", prompt="p", schema={"type": "object"}) == {"a": 2}
    assert comp.calls[1]["response_format"] == {"type": "json_object"} and "JSON schema" in comp.calls[1]["messages"][-1]["content"]


def test_all_models_failing_gives_a_clear_message():
    r, _ = route("Gemini", "m", [APITimeoutError(request=REQ), APITimeoutError(request=REQ)])
    with pytest.raises(LLMError, match="timed out"):
        LLM([r], [r], sleep=lambda *_: None).structured(system="s", prompt="p", schema={})
    with pytest.raises(LLMError, match="No AI key"):
        LLM([], []).chat(system="s", messages=[])


def test_groq_search_request_shape():
    msg = SimpleNamespace(content="A ||| https://a.example ||| s", model_extra={})
    comp = FakeCompletions([SimpleNamespace(choices=[SimpleNamespace(message=msg)])])
    gs = GroqSearch("k", per_minute=10_000, client=SimpleNamespace(chat=SimpleNamespace(completions=comp)))
    assert gs("veg Barcelona", "es-es", 5)[0]["href"] == "https://a.example"
    assert comp.calls[0]["tools"] == [{"type": "browser_search"}] and comp.calls[0]["tool_choice"] == "required"


def APIStatusErrorFake(code, text="error"):
    from openai import APIStatusError
    return APIStatusError(text, response=httpx.Response(code, request=REQ), body=None)


# ---------------------------------------------------------------- money
def test_fmt_range_and_budget_text():
    assert fmt_range(20, 30, "EUR") == "20–30 €" and fmt_range(20, 30, "GBP") == "£20–30"
    assert fmt_range(25, 25, "EUR") == "25 €"
    assert budget_text(UserQuery(city="x", budget_max=35, currency="EUR")) == "up to 35 € per person"
    est = UserQuery(city="x", budget_min=10, budget_max=20, currency="EUR", budget_estimated=True, budget_note="cheap")
    assert budget_text(est) == '10–20 € per person (my estimate for "cheap")'
    assert budget_text(UserQuery(city="x")) == "no limit given"


def menu_items(*prices, section="Mains"):
    return [MenuItem(original_name=f"d{i}", english_name=f"Dish {i}", price=p, currency="EUR", section=section)
            for i, p in enumerate(prices)]


def test_estimate_cost_from_menu_and_budget_verdicts():
    q = UserQuery(city="Barcelona", budget_max=35, currency="EUR")
    items = menu_items(8, 10, 12, 14, 16) + menu_items(3, 4, section="Drinks")   # drinks are left out
    cost = estimate_cost(items, q)
    assert (cost.low, cost.high, cost.source) == (20.0, 28.0, "menu")            # 2 x (10..14)
    assert cost.fit == "fits" and "5 priced dishes" in cost.basis
    assert estimate_cost(menu_items(15, 18, 20, 22), q).fit == "partly"          # 2x16.5=33 .. 2x20.5=41
    assert estimate_cost(menu_items(25, 30, 35), q).fit == "over"
    guide = estimate_cost([], q, guide_price=30)
    assert guide.source == "guide" and guide.low == guide.high == 30 and guide.fit == "fits"
    assert estimate_cost(menu_items(9, 10), q) is None                           # too few prices to say



# --------------------------------------------------------------- worldwide
@pytest.mark.parametrize("text", ["¥1,200", "1.200 円", "₩12,000", "Rp 45.000", "Rs. 350", "AED 45", "R$ 30,90",
                                  "45 TL", "150 บาท", "12,50 €", "$14.99", "₹180", "฿120"])
def test_prices_in_any_currency(text):
    assert count_prices(text) == 1


@pytest.mark.parametrize("text", ["open 9 hours 350 seats", "call 0034 933 123 456", "since 1998", "2 people"])
def test_not_prices(text):
    assert count_prices(text) == 0


def test_meal_times_work_outside_europe():
    assert open_for_meal("Mo-Su 17:00-19:30", "dinner") is True      # early dinner (US, UK)
    assert open_for_meal("Mo-Su 11:30-13:30", "lunch") is True        # early lunch (Japan, India)
    assert open_for_meal("Mo-Su 06:00-09:00", "breakfast") is True


def test_local_menu_words_find_menu_pages():
    html = '<a href="/omakase">お品書き</a><a href="/access">アクセス</a>'
    assert [u for u, _, _ in find_menu_links(html, "https://sushi.jp/")] == ["https://sushi.jp/omakase"]
    html2 = '<a href="/lista">Lista de pratos</a><a href="/sobre">Sobre</a>'
    assert find_menu_links(html2, "https://x.br/") == []
    assert [u for u, _, _ in find_menu_links(html2, "https://x.br/", extra_words=("lista de pratos",))] == ["https://x.br/lista"]


def test_search_region_and_money_formats_worldwide():
    assert (region_for("JP"), region_for("in"), region_for("ZZ")) == ("jp-jp", "in-en", "wt-wt")
    assert fmt_price(1200, "JPY") == "¥1,200" and fmt_price(90000, "VND") == "90,000 ₫"
    assert fmt_range(1500, 2400, "JPY") == "¥1,500–2,400" and fmt_range(400, 700, "INR") == "₹400–700"



def test_daily_limit_skips_model_at_once_and_for_the_session():
    daily = RateLimitError("Rate limit reached ... on tokens per day (TPD): Limit 200000",
                           response=httpx.Response(429, request=REQ), body=None)
    tired, tired_calls = route("Groq", "openai/gpt-oss-20b", [daily])
    fresh, fresh_calls = route("Groq", "qwen/qwen3.8-27b", [_resp('{"a": 1}'), _resp('{"a": 2}')])
    llm = LLM([tired, fresh], [tired, fresh], sleep=lambda *_: (_ for _ in ()).throw(AssertionError("no waiting")))
    assert llm.structured(system="s", prompt="p", schema={}) == {"a": 1}
    assert tired.out_for_today and len(tired_calls.calls) == 1            # tried once, no retries
    llm._start["fast"] = 0                                                # even if asked to start there again...
    assert llm.structured(system="s", prompt="p", schema={}) == {"a": 2}
    assert len(tired_calls.calls) == 1                                    # ...it is not called again


def test_busy_service_moves_on_quickly():
    busy = APIStatusErrorFake(503, "overloaded")
    g, g_calls = route("Gemini", "gemini-3.8-flash", [busy, busy])
    q, _ = route("Groq", "openai/gpt-oss-120b", [_resp('{"ok": 1}')])
    assert LLM([g, q], [g, q], sleep=lambda *_: None).structured(system="s", prompt="p", schema={}) == {"ok": 1}
    assert len(g_calls.calls) == 2


def test_groq_search_switches_model_when_one_is_out_for_today():
    daily = RateLimitError("tokens per day (TPD)", response=httpx.Response(429, request=REQ), body=None)
    msg = SimpleNamespace(content="A ||| https://a.example ||| s", model_extra={})
    comp = FakeCompletions([daily, SimpleNamespace(choices=[SimpleNamespace(message=msg)])])
    gs = GroqSearch("k", per_minute=10_000, client=SimpleNamespace(chat=SimpleNamespace(completions=comp)))
    assert gs("veg Barcelona", "es-es", 5)[0]["href"] == "https://a.example"
    assert [c["model"] for c in comp.calls] == ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    assert gs.models == ["openai/gpt-oss-120b"]


# ------------------------------------------------------------- optional Claude
def test_claude_adapter_translates_requests_answers_and_errors():
    import anthropic

    from scout.claude_adapter import ClaudeChat

    calls = []
    areq = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    class FakeMessages:
        def __init__(self, replies):
            self.replies = list(replies)

        def create(self, **kw):
            calls.append(kw)
            r = self.replies.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

    ok = SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking="..."), SimpleNamespace(type="text", text='{"a": 1}')],
                         stop_reason="end_turn", usage=SimpleNamespace(input_tokens=12, output_tokens=3))
    effort_err = anthropic.BadRequestError("output_config.effort: not supported",
                                           response=httpx.Response(400, request=areq), body=None)
    busy = anthropic.RateLimitError("rate limited", response=httpx.Response(429, request=areq), body=None)
    fake = SimpleNamespace(messages=FakeMessages([effort_err, ok, busy, ok]))
    claude = Route("Claude", "claude-haiku-5-5", ClaudeChat("sk-ant-x", client=fake), RateLimiter(100_000))
    llm = LLM([claude], [claude], sleep=lambda *_: None)

    assert llm.structured(system="Be brief.", prompt="p", schema={"type": "object"}) == {"a": 1}
    first, second = calls[0], calls[1]
    assert first["system"] == "Be brief." and first["messages"] == [{"role": "user", "content": "p"}]
    assert first["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert "effort" not in second["output_config"]                    # retried without effort after the 400
    assert llm.usage["input_tokens"] == 12
    assert llm.structured(system="s", prompt="p", schema={}) == {"a": 1}   # 429 -> waited and retried


def test_claude_key_is_recognised_and_used_only_when_set():
    from scout.llm import build_routes
    s = Settings()
    s.add_key("sk-ant-api03-xyz")
    assert s.claude_key and s.has_ai
    s.gemini_key = "AIzaX"
    smart, fast = build_routes(s)
    assert [r.provider for r in smart][0] == "Gemini" and smart[-1].provider == "Claude"   # paid one last
    s.main_ai = "claude"
    assert build_routes(s)[0][0].label == "claude-sonnet-5-5 (Claude)"
    assert all(r.provider != "Claude" for r in build_routes(Settings(gemini_key="AIzaX"))[0])



# ------------------------------------------------------------ open or closed
def test_closure_words_are_spotted_in_local_languages():
    from scout.europe import closed_words, is_european
    from scout.status import suspicious
    es = closed_words("ES")
    assert suspicious("Cierra Rasoterra, uno de los grandes restaurantes veganos", "Rasoterra", es)
    assert suspicious("Rasoterra hará su último servicio el 3 de mayo", "Rasoterra", es)
    assert not suspicious("Rasoterra: vegan tasting menu, natural wines", "Rasoterra", es)
    assert not suspicious("Another place cierra this week", "Rasoterra", es)          # must mention the restaurant
    assert "fermé définitivement" in closed_words("FR") and "cierra" not in closed_words("FR")
    assert is_european("es") and is_european("GB") and not is_european("JP") and not is_european("")


def test_check_open_uses_ai_only_when_there_is_closure_news():
    from scout.status import check_open
    from types import SimpleNamespace as NS
    q = UserQuery(city="Barcelona", country_code="ES")
    c = Candidate(place_id="x", name="Rasoterra")
    asked = []
    llm = NS(structured=lambda **kw: asked.append(kw) or {"status": "permanently_closed",
                                                           "evidence": "Last service 3 May 2026.", "source_number": 1})
    quiet = WebSearch(min_gap=0, backend=lambda q_, r, n: [{"title": "Rasoterra menu", "href": "https://a.example", "body": "vegan"}])
    assert not check_open(c, q, quiet, lambda u: None, llm).closed and asked == []     # no news -> no AI call
    news = WebSearch(min_gap=0, backend=lambda q_, r, n: [{"title": "Cierra Rasoterra", "href": "https://timeout.example/x",
                                                            "body": "último servicio el 3 de mayo"}])
    page = Page(url="https://timeout.example/x", content_type="html", text="El proyecto Rasoterra ha llegado a su fin.")
    res = check_open(c, q, news, lambda u: page, llm)
    assert res.closed and res.source_url == "https://timeout.example/x" and "3 May" in res.note
    assert "ha llegado a su fin" in asked[0]["prompt"]
    tagged = Candidate(place_id="y", name="Old Place", tags={"disused:amenity": "restaurant"}, osm_url="https://osm/x")
    assert check_open(tagged, q, quiet, lambda u: None, llm).closed                     # the map already knows


def test_every_european_country_has_closure_words():
    from scout.europe import COUNTRY_LANGS, ENGLISH_ONLY, EUROPE, closed_words
    for cc in EUROPE:
        assert cc in COUNTRY_LANGS or cc in ENGLISH_ONLY, cc
        assert closed_words(cc)
    assert "chiude" in closed_words("IT") and "suljettu" in closed_words("FI") and "kapandı" in closed_words("TR")


# ------------------------------------------------------------ still open?
def _places(status_by_name, seen=None):
    import json as _json
    from scout.places import GooglePlaces

    def handler(req):
        body = _json.loads(req.content)
        if seen is not None:
            seen.append(body)
        if req.headers.get("X-Goog-Api-Key") == "bad":
            return httpx.Response(403, json={"error": {"message": "API key not valid."}})
        return httpx.Response(200, json={"places": [
            {"displayName": {"text": n}, "businessStatus": st_, "formattedAddress": "Carrer X, Barcelona",
             "googleMapsUri": f"https://maps.google.com/?cid={i}", "location": {"latitude": lat, "longitude": 2.17}}
            for i, (n, (st_, lat)) in enumerate(status_by_name.items())]})
    return GooglePlaces("good", transport=httpx.MockTransport(handler))


def test_place_names_match_carefully():
    from scout.places import same_name
    assert same_name("Rasoterra", "Rasoterra Bistró Vegetarià")
    assert same_name("Cal Pep", "Restaurant Cal Pep")
    assert same_name("Café de l'Òpera", "Cafe de l'Opera")
    assert not same_name("Cal Pep", "Can Pep")
    assert not same_name("Teresa Carles", "Flax & Kale")


def test_google_maps_decides_open_or_closed():
    from scout.places import PlacesError
    from scout.status import check_open
    from types import SimpleNamespace as NS
    q = UserQuery(city="Barcelona", country_code="ES")
    searched = []
    web = WebSearch(min_gap=0, backend=lambda q_, r, n: searched.append(q_) or [])
    llm = NS(structured=lambda **kw: pytest.fail("no AI call needed"))
    c = Candidate(place_id="x", name="Rasoterra", lat=41.3825, lon=2.17)
    places = _places({"Rasoterra": ("CLOSED_PERMANENTLY", 41.3826), "Rasoterra Shop": ("OPERATIONAL", 41.50)})
    res = check_open(c, q, web, lambda u: None, llm, places=places)
    assert res.closed and not res.temporary and "permanently closed" in res.note and res.maps_url.startswith("https://maps")
    assert searched == []                                                   # Google was decisive
    res = check_open(c, q, web, lambda u: None, llm, places=_places({"Rasoterra": ("CLOSED_TEMPORARILY", 41.3826)}))
    assert res.closed and res.temporary
    res = check_open(c, q, web, lambda u: None, llm, places=_places({"Rasoterra": ("OPERATIONAL", 41.3826)}))
    assert not res.closed and res.confirmed_open
    # a same-name place 13 km away is a different restaurant: ignored, so the free checks run
    res = check_open(c, q, web, lambda u: None, llm, places=_places({"Rasoterra": ("CLOSED_PERMANENTLY", 41.50)}))
    assert not res.closed and "Not found on Google Maps" in res.note and searched
    bad = _places({})
    bad.key = "bad"
    with pytest.raises(PlacesError, match="refused the key"):
        check_open(c, q, web, lambda u: None, llm, places=bad)


def test_website_probe_spots_dead_sites():
    import socket
    from scout.status import probe_website

    def resolve(host, port):
        if host.endswith("gone.example"):
            raise socket.gaierror("no such host")
        return [("ok",)]

    def handler(req):
        if req.url.host == "removed.example":
            return httpx.Response(404, text="Not found")
        if req.url.host == "parked.example":
            return httpx.Response(200, text="<h1>This domain is for sale!</h1> Buy this domain today.")
        return httpx.Response(200, text="<p>Carta · Menú del día 14 €</p>")
    t = httpx.MockTransport(handler)
    assert probe_website("https://gone.example", resolve=resolve, transport=t).state == "gone"
    assert probe_website("https://removed.example/carta", resolve=resolve, transport=t).state == "gone"
    assert probe_website("https://parked.example", resolve=resolve, transport=t).state == "gone"
    ok = probe_website("https://fine.example", resolve=resolve, transport=t)
    assert ok.state == "ok" and "Menú del día" in ok.text
    assert probe_website("https://instagram.com/somewhere", resolve=resolve, transport=t).state == "unknown"

    def offline(host, port):
        raise socket.gaierror("offline")
    assert probe_website("https://gone.example", resolve=offline, transport=t).state == "unknown"   # we're offline


def test_dead_website_or_closed_listing_drops_the_place():
    from scout.status import SiteCheck, check_open
    from types import SimpleNamespace as NS
    q = UserQuery(city="Barcelona", country_code="ES")
    quiet = WebSearch(min_gap=0, backend=lambda q_, r, n: [])
    llm = NS(structured=lambda **kw: pytest.fail("no AI call needed"))
    c = Candidate(place_id="x", name="Sitio Viejo", website="https://sitioviejo.example")
    res = check_open(c, q, quiet, lambda u: None, llm, probe=lambda u: SiteCheck("gone", "its website domain no longer exists"))
    assert res.closed and "strong sign" in res.note
    res = check_open(c, q, quiet, lambda u: None, llm, probe=lambda u: SiteCheck("ok", text="Bienvenidos"))
    assert not res.closed and "website is working" in res.note
    # Tripadvisor-style "CLOSED" in a listing title is enough to ask the AI
    asked = []
    listing = WebSearch(min_gap=0, backend=lambda q_, r, n: [{"title": "SITIO VIEJO, Barcelona - CLOSED - Restaurant Reviews",
                                                               "href": "https://tripadvisor.example/x", "body": "Tapas"}])
    llm2 = NS(structured=lambda **kw: asked.append(kw) or {"status": "permanently_closed",
                                                            "evidence": "Listed as CLOSED.", "source_number": 1})
    res = check_open(c, q, listing, lambda u: Page(url=u, content_type="html", text="CLOSED"), llm2,
                     probe=lambda u: SiteCheck("unknown"))
    assert res.closed and asked and "CLOSED" in asked[0]["prompt"]


def test_menu_search_only_uses_pages_about_this_restaurant():
    from scout.menu import about_this_restaurant, read_menu
    assert about_this_restaurant("Casa Leopoldo, Barcelona - Menu & prices - TheFork",
                                 "https://www.thefork.com/restaurant/casa-leopoldo-r123", "Casa Leopoldo")[0]
    assert not about_this_restaurant("The 25 best restaurants in Barcelona", "https://x.example/best", "Casa Leopoldo")[0]
    assert not about_this_restaurant("Can Culleretes menu", "https://y.example/can-culleretes", "Casa Leopoldo")[0]

    menu_text = " ".join(f"Plato {i} {8 + i} €" for i in range(10))
    pages = {
        "https://list.example/best": Page(url="https://list.example/best", content_type="html",
                                          text="Casa Leopoldo " + menu_text),
        "https://other.example/can-culleretes": Page(url="https://other.example/can-culleretes", content_type="html",
                                                     text=menu_text),
        "https://www.thefork.example/restaurant/casa-leopoldo": Page(
            url="https://www.thefork.example/restaurant/casa-leopoldo", content_type="html",
            text="Casa Leopoldo menu " + menu_text),
    }
    web = WebSearch(min_gap=0, backend=lambda q_, r, n: [
        {"title": "10 best tapas bars in Barcelona", "href": "https://list.example/best", "body": ""},
        {"title": "Can Culleretes - menu", "href": "https://other.example/can-culleretes", "body": ""},
        {"title": "Casa Leopoldo - TheFork", "href": "https://www.thefork.example/restaurant/casa-leopoldo", "body": ""}])
    c = Candidate(place_id="x", name="Casa Leopoldo")
    res, text = read_menu(c, UserQuery(city="Barcelona", country_code="ES"), web,
                          lambda u, **kw: pages[u])
    assert res.menu_url == "https://www.thefork.example/restaurant/casa-leopoldo"
    assert "thefork.example page for this restaurant" in res.how_found
    assert "a list of several restaurants" in res.notes and "not about this restaurant" in res.notes

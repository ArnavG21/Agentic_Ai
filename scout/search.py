"""Free web search plus page reading.

Primary: DuckDuckGo via the `ddgs` package (no key).
Backup:  Groq's free browser_search tool on its gpt-oss models (free Groq key).
         Used automatically when DuckDuckGo returns nothing - for example when it
         blocks requests coming from a cloud server.

search()   -> list of SearchHit(title, url, snippet)
research() -> runs a few searches, opens the most useful pages (through the
              Scraper: direct download, or Jina Reader for hard pages) and returns
              the evidence as numbered sources [1], [2]... so the AI can cite them
              and the UI can link to them.

Searches are spaced out (one every ~1.5 s) because DuckDuckGo blocks bursts.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .scrape import readable
from .web import FetchError, Page, safe_get


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str = ""


@dataclass
class Evidence:
    queries: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)   # {n, url, title, snippet, text}

    def as_prompt(self, chars_per_source: int = 2500) -> str:
        blocks = []
        for s in self.sources:
            body = s["text"][:chars_per_source] if s["text"] else s["snippet"]
            blocks.append(f"[{s['n']}] {s['title']}\nURL: {s['url']}\n{body}")
        return "\n\n".join(blocks) or "(no sources found)"

    @property
    def urls(self) -> set[str]:
        return {s["url"] for s in self.sources}


def _direct_read(url: str, want_prices: bool = False) -> Page:
    return safe_get(url)


class WebSearch:
    def __init__(self, region: str = "wt-wt", min_gap: float = 1.5,
                 backend: Optional[Callable[[str, str, int], list[dict]]] = None,
                 fetch: Callable[..., Page] = _direct_read,
                 fallback: Optional[Callable[[str, str, int], list[dict]]] = None):
        self.region = region
        self.fetch = fetch
        self.min_gap = min_gap
        self._backend = backend or _ddgs_text
        self._fallback = fallback
        self._lock = threading.Lock()
        self._last = 0.0
        self.count = 0
        self.fallback_count = 0     # searches answered by the backup engine

    def search(self, query: str, max_results: int = 8) -> list[SearchHit]:
        with self._lock:  # one search at a time, spaced out
            gap = self.min_gap - (time.monotonic() - self._last)
            if gap > 0:
                time.sleep(gap)
            try:
                rows = self._backend(query, self.region, max_results)
            except Exception:
                time.sleep(4)
                try:
                    rows = self._backend(query, self.region, max_results)
                except Exception:
                    rows = []
            if not rows and self._fallback is not None:
                try:
                    rows = self._fallback(query, self.region, max_results)
                    self.fallback_count += 1
                except Exception:
                    rows = []
            self._last = time.monotonic()
            self.count += 1
        hits, seen = [], set()
        for r in rows:
            url = r.get("href") or r.get("url") or ""
            if url and url not in seen:
                seen.add(url)
                hits.append(SearchHit(title=r.get("title", ""), url=url, snippet=r.get("body", "")))
        return hits

    def research(self, queries: list[str], pages: int = 3, max_sources: int = 10) -> Evidence:
        """Search, then read up to `pages` result pages in full; keep the rest as snippets."""
        ev = Evidence()
        hits: list[SearchHit] = []
        for q in queries:
            ev.queries.append(q)
            hits.extend(self.search(q))
        seen, opened = set(), 0
        for h in hits:
            if h.url in seen or len(ev.sources) >= max_sources:
                continue
            seen.add(h.url)
            text, via = "", ""
            if opened < pages and readable(h.url):
                try:
                    page = self.fetch(h.url)
                    text, via = page.text, page.via
                    opened += 1
                except FetchError:
                    text = ""
            ev.sources.append({"n": len(ev.sources) + 1, "url": h.url, "title": h.title,
                               "snippet": h.snippet, "text": text, "via": via})
        return ev


def _ddgs_text(query: str, region: str, max_results: int) -> list[dict]:
    from ddgs import DDGS

    return DDGS().text(query, region=region, max_results=max_results) or []


class GroqSearch:
    """Backup search through Groq's built-in `browser_search` tool (free gpt-oss models).

    Groq runs the search on its servers. We ask the model to list what it found,
    one result per line as `TITLE ||| URL ||| SUMMARY`, and also read any raw
    search results Groq attaches to the response (`executed_tools`), which are
    preferred because they come straight from the search engine, not the model.
    """

    BASE_URL = "https://api.groq.com/openai/v1"
    SYSTEM = ("You are a web search tool. Use browser search for the user's query, then list the results "
              "you found, one per line, exactly as: TITLE ||| URL ||| one-sentence summary of the page. "
              "Only list real pages from the search results, with their full URLs. No other text.")

    MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b")   # the Groq models that have browser search

    def __init__(self, api_key: str, model: str = "", per_minute: int = 4, client=None):
        from openai import OpenAI

        from .llm import RateLimiter

        self.client = client or OpenAI(api_key=api_key, base_url=self.BASE_URL, max_retries=1, timeout=90)
        self.models = [model] if model else list(self.MODELS)
        self.limiter = RateLimiter(per_minute)   # free tier: few tokens per minute, so go slowly

    def __call__(self, query: str, region: str, max_results: int) -> list[dict]:
        from openai import RateLimitError

        from .llm import is_daily_limit

        resp = None
        for model in list(self.models):
            self.limiter.wait()
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": self.SYSTEM},
                              {"role": "user", "content": f"{query}\n(List up to {max_results} results.)"}],
                    tools=[{"type": "browser_search"}], tool_choice="required",
                    reasoning_effort="low", max_completion_tokens=2500,
                )
                break
            except RateLimitError as e:
                if is_daily_limit(e) and len(self.models) > 1:
                    self.models.remove(model)      # out for today - use the other model from now on
                    continue
                raise
        if resp is None:
            return []
        msg = resp.choices[0].message
        extra = getattr(msg, "model_extra", None) or {}
        rows = parse_executed_tools(extra.get("executed_tools"))
        return rows[:max_results] if rows else parse_result_lines(msg.content or "")[:max_results]


def parse_executed_tools(tools) -> list[dict]:
    """Collect {title, href, body} from any dicts carrying a URL inside Groq's executed_tools."""
    rows: list[dict] = []

    def walk(x):
        if isinstance(x, dict):
            url = x.get("url") or x.get("link")
            if isinstance(url, str) and url.startswith("http"):
                rows.append({"title": str(x.get("title", "")), "href": url,
                             "body": str(x.get("content") or x.get("snippet") or "")[:500]})
                return
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(tools or [])
    return rows


def parse_result_lines(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        parts = [p.strip(" -*•\t") for p in line.split("|||")]
        if len(parts) >= 2:
            m = re.search(r"https?://[^\s<>\"')\]]+", parts[1])
            if m:
                rows.append({"title": parts[0], "href": m.group(0).rstrip(".,"),
                             "body": parts[2] if len(parts) > 2 else ""})
    return rows


REGIONS = {  # country code -> DuckDuckGo region, so results are local to the city being searched
    "AR": "ar-es", "AU": "au-en", "AT": "at-de", "BE": "be-fr", "BR": "br-pt", "BG": "bg-bg", "CA": "ca-en",
    "CL": "cl-es", "CN": "cn-zh", "CO": "co-es", "HR": "hr-hr", "CZ": "cz-cs", "DK": "dk-da", "EE": "ee-et",
    "FI": "fi-fi", "FR": "fr-fr", "DE": "de-de", "GR": "gr-el", "HK": "hk-tzh", "HU": "hu-hu", "IN": "in-en",
    "ID": "id-id", "IE": "ie-en", "IL": "il-he", "IT": "it-it", "JP": "jp-jp", "KR": "kr-kr", "LV": "lv-lv",
    "LT": "lt-lt", "MY": "my-en", "MX": "mx-es", "NL": "nl-nl", "NZ": "nz-en", "NO": "no-no", "PE": "pe-es",
    "PH": "ph-en", "PL": "pl-pl", "PT": "pt-pt", "RO": "ro-ro", "RU": "ru-ru", "SA": "xa-ar", "SG": "sg-en",
    "SK": "sk-sk", "SI": "sl-sl", "ZA": "za-en", "ES": "es-es", "SE": "se-sv", "CH": "ch-de", "TW": "tw-tzh",
    "TH": "th-th", "TR": "tr-tr", "UA": "ua-uk", "GB": "uk-en", "US": "us-en", "VN": "vn-vi", "AE": "xa-ar",
    "EG": "xa-ar", "MA": "xa-ar", "QA": "xa-ar", "PK": "pk-en", "BD": "wt-wt", "LK": "wt-wt", "NP": "wt-wt",
}


def region_for(country_code: str) -> str:
    return REGIONS.get((country_code or "").upper(), "wt-wt")

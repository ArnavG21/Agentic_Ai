"""Settings, read from a .env file, environment variables or Streamlit secrets.

Everything the agent uses is free:
- AI: Google Gemini free tier (main) and Groq free tier (backup). Free keys, no card.
  Optional: Anthropic's Claude API (paid, pay-as-you-go) if ANTHROPIC_API_KEY is set.
- Restaurant facts: OpenStreetMap (Nominatim). No key.
- Web search: DuckDuckGo (no key), with Groq's browser search as backup.
- Page reading: direct download, with Jina Reader as the scraping backup for pages
  that need a real browser (JavaScript menus, PDFs, sites that block simple readers).
  No key needed; a free Jina key only raises the speed limit.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional
    pass

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_URL = "https://api.groq.com/openai/v1"
CLAUDE_SMART = ["claude-sonnet-5-5"]      # optional, paid: only used when ANTHROPIC_API_KEY is set
CLAUDE_FAST = ["claude-haiku-5-5"]
CLAUDE_RPM = 40

# Models are tried in order. If one times out or is unavailable on the free plan,
# the agent moves to the next and keeps using whichever works.
GEMINI_SMART = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]
GEMINI_FAST = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash"]
# Each Groq model has its own daily allowance, so listing several multiplies the free daily budget.
GROQ_SMART = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
GROQ_FAST = ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
GEMINI_RPM = 9      # free plan allows about 10 requests a minute
GROQ_RPM = 20       # free plan allows 30, but only 8,000 tokens a minute
GROQ_MAX_PROMPT_CHARS = 14000   # keeps a request under Groq's free per-minute token limit


@dataclass
class Settings:
    main_ai: str = "gemini"            # gemini | groq | claude
    gemini_key: str = ""
    groq_key: str = ""
    claude_key: str = ""               # optional, paid Anthropic API key
    jina_key: str = ""                 # optional: raises Jina Reader from 20 to 500 pages a minute
    google_maps_key: str = ""          # optional: Google Maps Places API key - the strongest "is it still open?" check
    contact_email: str = ""            # sent to OpenStreetMap in the User-Agent, as their policy asks
    search_mode: str = "auto"          # auto (DuckDuckGo, Groq if empty) | duckduckgo | groq
    use_jina: bool = True
    candidates_to_verify: int = 14     # restaurants looked up on the map
    shortlist_size: int = 4
    pages_per_search: int = 3          # pages opened per web search
    menu_char_limit: int = 24000      # whole menu across pages, translated in parts
    http_timeout: float = 15.0
    ai_timeout: float = 60.0
    max_workers: int = 3
    extra: dict = field(default_factory=dict)

    def add_key(self, key: str) -> None:
        """Put a key in the right slot by its shape: Gemini keys start 'AIza', Groq keys 'gsk_'."""
        key = (key or "").strip().strip('"').strip("'")
        if key.startswith("gsk_"):
            self.groq_key = self.groq_key or key
        elif key.startswith("sk-ant-"):
            self.claude_key = self.claude_key or key
        elif key.startswith("jina_"):
            self.jina_key = self.jina_key or key
        elif key:
            self.gemini_key = self.gemini_key or key

    @property
    def has_ai(self) -> bool:
        return bool(self.gemini_key or self.groq_key or self.claude_key)

    @classmethod
    def from_env(cls, get=os.getenv) -> "Settings":
        s = cls()
        s.main_ai = (get("MAIN_AI") or get("LLM_PROVIDER") or "gemini").lower()
        s.gemini_key = (get("GEMINI_API_KEY") or "").strip()
        s.groq_key = (get("GROQ_API_KEY") or "").strip()
        s.jina_key = (get("JINA_API_KEY") or "").strip()
        s.claude_key = (get("ANTHROPIC_API_KEY") or get("CLAUDE_API_KEY") or "").strip()
        s.add_key(get("LLM_API_KEY") or "")          # older setting name - accepted either way
        s.google_maps_key = (get("GOOGLE_MAPS_API_KEY") or get("GOOGLE_PLACES_API_KEY") or "").strip()
        s.contact_email = get("CONTACT_EMAIL") or ""
        s.search_mode = (get("SEARCH_MODE") or s.search_mode).lower()
        s.shortlist_size = int(get("SCOUT_SHORTLIST_SIZE") or s.shortlist_size)
        if s.main_ai in ("anthropic",):
            s.main_ai = "claude"
        if s.main_ai not in ("gemini", "groq", "claude"):
            s.main_ai = "gemini"
        return s

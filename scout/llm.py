"""The AI client: free models, tried in order, with clear errors.

Two call patterns:
- structured(): the answer must be JSON matching a schema (used for every
  pipeline step), so the code never has to parse free text.
- chat(): plain text, for follow-up questions.

Reliability on free plans:
- Routes. A route is one model at one provider (e.g. Gemini 3.8 Flash). Calls try
  routes in order - several Gemini models, then Groq if a Groq key is set - and
  stay on the first one that works, so one slow or unavailable model can't stall
  the whole search.
- Low "thinking". Gemini 3 and gpt-oss models reason before answering; asking for
  low effort makes them much faster and is plenty for extraction and summaries.
- Rate limits. Each provider has its own limiter (requests per minute), and
  "429 Too Many Requests" answers are retried with back-off.
- JSON fallback. If a provider rejects strict JSON-schema mode, the request is
  retried in plain JSON mode with the schema written into the instructions.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from .config import (CLAUDE_FAST, CLAUDE_RPM, CLAUDE_SMART, GEMINI_FAST, GEMINI_RPM, GEMINI_SMART, GEMINI_URL,
                     GROQ_FAST, GROQ_MAX_PROMPT_CHARS, GROQ_RPM, GROQ_SMART, GROQ_URL, Settings)


class LLMError(RuntimeError):
    pass


class RateLimiter:
    """Allow at most `per_minute` calls per minute, spaced evenly, across threads."""

    def __init__(self, per_minute: int):
        self.interval = 60.0 / max(per_minute, 1)
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        time.sleep(max(0.0, slot - now))


@dataclass
class Route:
    provider: str                     # "Gemini" | "Groq" | "Claude"
    model: str
    client: Any
    limiter: RateLimiter
    max_prompt_chars: Optional[int] = None
    reasoning_effort: str = "low"
    json_schema_ok: bool = True       # flips to False if the provider rejects strict schemas
    out_for_today: bool = False       # daily free allowance used up - skip for the rest of the session

    @property
    def label(self) -> str:
        return f"{self.model} ({self.provider})"


def build_routes(s: Settings, timeout: Optional[float] = None) -> tuple[list[Route], list[Route]]:
    """Smart and fast route lists for the keys available, main AI first."""
    timeout = timeout or s.ai_timeout
    groups: dict[str, tuple[list[Route], list[Route]]] = {}
    if s.gemini_key:
        c = OpenAI(api_key=s.gemini_key, base_url=GEMINI_URL, max_retries=0, timeout=timeout)
        lim = RateLimiter(GEMINI_RPM)
        groups["gemini"] = ([Route("Gemini", m, c, lim) for m in GEMINI_SMART],
                            [Route("Gemini", m, c, lim) for m in GEMINI_FAST])
    if s.groq_key:
        c = OpenAI(api_key=s.groq_key, base_url=GROQ_URL, max_retries=0, timeout=timeout)
        lim = RateLimiter(GROQ_RPM)
        groups["groq"] = ([Route("Groq", m, c, lim, GROQ_MAX_PROMPT_CHARS) for m in GROQ_SMART],
                          [Route("Groq", m, c, lim, GROQ_MAX_PROMPT_CHARS) for m in GROQ_FAST])
    if s.claude_key:
        from .claude_adapter import ClaudeChat

        c = ClaudeChat(s.claude_key, timeout=timeout)
        lim = RateLimiter(CLAUDE_RPM)
        groups["claude"] = ([Route("Claude", m, c, lim) for m in CLAUDE_SMART],
                            [Route("Claude", m, c, lim) for m in CLAUDE_FAST])
    # main AI first; then the free ones; the paid Claude API last unless it was chosen as main
    order = [s.main_ai] + [k for k in ("gemini", "groq", "claude") if k != s.main_ai]
    smart = [r for k in order if k in groups for r in groups[k][0]]
    fast = [r for k in order if k in groups for r in groups[k][1]]
    return smart, fast


class _RouteFailed(Exception):
    def __init__(self, reason: str, fatal: bool = False):
        super().__init__(reason)
        self.fatal = fatal


DAILY_LIMIT_WORDS = ("per day", "perday", "(tpd)", "(rpd)", "daily")


def is_daily_limit(err: Exception) -> bool:
    """True when a 429 means the free allowance for TODAY is gone (not just 'slow down for a minute')."""
    text = str(err).lower().replace("_", " ")
    return any(w in text for w in DAILY_LIMIT_WORDS)


class LLM:
    def __init__(self, smart_routes: list[Route], fast_routes: list[Route], sleep=time.sleep):
        self.routes = {"smart": smart_routes, "fast": fast_routes}
        self._start = {"smart": 0, "fast": 0}     # index of the route that last worked
        self._lock = threading.Lock()
        self._sleep = sleep
        self.usage: dict = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "models": {}}
        self.notes: list[str] = []                # human-readable notes about fallbacks

    @classmethod
    def from_settings(cls, s: Settings) -> "LLM":
        return cls(*build_routes(s))

    VISION_PROVIDERS = ("gemini", "claude")       # models that can read photos (Groq's free ones can't)

    @property
    def can_see(self) -> bool:
        return any(r.provider in self.VISION_PROVIDERS for r in self.routes["fast"] + self.routes["smart"])

    def read_images(self, prompt: str, images: list[tuple[bytes, str]], max_tokens: int = 8000) -> str:
        """Ask a vision-capable model about photos (e.g. transcribe a photographed menu)."""
        content: list[dict] = [{"type": "text", "text": prompt}]
        for data, mime in images:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}})
        seen, failures = set(), []
        for r in self.routes["fast"] + self.routes["smart"]:
            if r.provider not in self.VISION_PROVIDERS or r.out_for_today or r.label in seen:
                continue
            seen.add(r.label)
            try:
                resp = self._call_route(r, [{"role": "user", "content": content}], max_tokens, None, "menu_photos")
            except _RouteFailed as e:
                failures.append(f"{r.label}: {e}")
                if e.fatal:
                    break
                continue
            text = (resp.choices[0].message.content or "").strip()
            if text:
                with self._lock:
                    self.usage["calls"] += 1
                    self.usage["models"][r.label] = self.usage["models"].get(r.label, 0) + 1
                return text
        raise LLMError("No AI model could read the photos. " + " | ".join(failures))

    @property
    def main_label(self) -> str:
        r = self.routes["smart"]
        return r[self._start["smart"]].label if r else "no AI configured"

    # ------------------------------------------------------------- one route
    def _call_route(self, r: Route, messages: list[dict], max_tokens: int, schema: Optional[dict], name: str) -> Any:
        msgs = [dict(m) for m in messages]
        if r.max_prompt_chars:
            budget = r.max_prompt_chars
            for m in msgs:
                if len(m["content"]) > budget:
                    m["content"] = m["content"][:budget] + "\n[...shortened to fit the free plan...]"
                budget = max(2000, budget - len(m["content"]))
        kwargs: dict = {"model": r.model, "messages": msgs, "max_tokens": max_tokens}
        if r.reasoning_effort:
            kwargs["reasoning_effort"] = r.reasoning_effort
        if schema is not None:
            if r.json_schema_ok:
                kwargs["response_format"] = {"type": "json_schema",
                                             "json_schema": {"name": name, "strict": True, "schema": schema}}
            else:
                kwargs["response_format"] = {"type": "json_object"}
                kwargs["messages"] = msgs[:-1] + [{**msgs[-1], "content": msgs[-1]["content"]
                                                   + "\n\nReply with JSON only, matching this JSON schema exactly:\n"
                                                   + json.dumps(schema)}]
        delay, timeouts, attempts = 6.0, 0, 0
        while True:
            attempts += 1
            r.limiter.wait()
            try:
                return r.client.chat.completions.create(**kwargs)
            except RateLimitError as e:
                if is_daily_limit(e):             # no point waiting: move to the next model at once
                    r.out_for_today = True
                    raise _RouteFailed("free daily allowance used up (resets within 24 hours)")
                if attempts >= 4:
                    raise _RouteFailed("free quota used up for now (too many requests)")
                self._sleep(delay)
                delay *= 2
            except (APITimeoutError, APIConnectionError) as e:
                timeouts += 1
                if timeouts >= 2:
                    raise _RouteFailed("timed out" if isinstance(e, APITimeoutError) else "could not be reached")
                self._sleep(2)
            except APIStatusError as e:
                text = str(e).lower()
                if e.status_code in (401, 403):
                    raise _RouteFailed("the API key was rejected", fatal=True)
                if e.status_code == 400 and "reasoning" in text and "reasoning_effort" in kwargs:
                    kwargs.pop("reasoning_effort")
                    r.reasoning_effort = ""
                    continue
                if e.status_code == 400 and schema is not None and r.json_schema_ok and \
                        ("schema" in text or "response_format" in text):
                    r.json_schema_ok = False
                    return self._call_route(r, messages, max_tokens, schema, name)
                if e.status_code in (500, 502, 503, 504):
                    if attempts < 2:
                        self._sleep(3)
                        continue
                    raise _RouteFailed(f"the service is busy right now (error {e.status_code})")
                if e.status_code == 404:
                    raise _RouteFailed("this model is not available on your plan")
                if e.status_code == 413:
                    raise _RouteFailed("the request was too large for the free plan")
                raise _RouteFailed(f"error {e.status_code}: {str(e)[:160]}")

    # ------------------------------------------------------------ all routes
    def _complete(self, kind: str, messages: list[dict], max_tokens: int,
                  schema: Optional[dict] = None, name: str = "result") -> str:
        routes = self.routes[kind]
        if not routes:
            raise LLMError("No AI key is set. Add GEMINI_API_KEY or GROQ_API_KEY to the .env file.")
        start = self._start[kind]
        order = routes[start:] + routes[:start]
        failures: list[str] = []
        dead: set[str] = set()   # providers whose key was rejected
        for r in order:
            if r.provider in dead or r.out_for_today:
                continue
            try:
                resp = self._call_route(r, messages, max_tokens, schema, name)
            except _RouteFailed as e:
                failures.append(f"{r.label}: {e}")
                if e.fatal:
                    dead.add(r.provider)
                continue
            choice = resp.choices[0]
            text = (choice.message.content or "").strip()
            if choice.finish_reason == "length" and schema is not None:
                failures.append(f"{r.label}: answer was cut off")
                continue
            if schema is not None:
                try:
                    json.loads(_strip_fences(text))
                except json.JSONDecodeError:
                    failures.append(f"{r.label}: answer was not valid JSON")
                    continue
            with self._lock:
                i = routes.index(r)
                if failures:
                    self.notes.append(f"Used {r.label} because: " + "; ".join(failures))
                self._start[kind] = i
                self.usage["calls"] += 1
                self.usage["models"][r.label] = self.usage["models"].get(r.label, 0) + 1
                u = getattr(resp, "usage", None)
                if u is not None:
                    self.usage["input_tokens"] += getattr(u, "prompt_tokens", 0) or 0
                    self.usage["output_tokens"] += getattr(u, "completion_tokens", 0) or 0
            return text
        raise LLMError("Every AI model failed. " + " | ".join(failures)
                       + ". Check your keys in the .env file (run python check.py), or try again in a minute.")

    def structured(self, *, system: str, prompt: str, schema: dict, name: str = "result",
                   fast: bool = True, max_tokens: int = 8000) -> dict:
        text = self._complete("fast" if fast else "smart",
                              [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                              max_tokens, schema, name)
        return json.loads(_strip_fences(text))

    def chat(self, *, system: str, messages: list[dict], fast: bool = False, max_tokens: int = 2000) -> str:
        return self._complete("fast" if fast else "smart", [{"role": "system", "content": system}, *messages], max_tokens)


def _strip_fences(text: str) -> str:
    """Some models wrap JSON in ```json fences even in JSON mode."""
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
    return text.strip()

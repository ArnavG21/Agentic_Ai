"""Lets the agent use Anthropic's Claude (optional, paid API) through the same code path as Gemini and Groq.

Gemini and Groq speak the OpenAI chat format; Claude has its own Messages API.
ClaudeChat looks like an OpenAI client (client.chat.completions.create(...)) but
calls Claude underneath, translating:
  - system messages      -> Claude's `system` parameter
  - JSON schema answers  -> Claude structured outputs (output_config.format)
  - reasoning_effort     -> Claude's effort setting (output_config.effort)
  - errors               -> the matching OpenAI error types, so retries and
                            fallbacks in llm.py work unchanged.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import httpx
import openai

_REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_error(code: int, message: str) -> openai.APIStatusError:
    resp = httpx.Response(code, request=_REQ)
    if code == 429:
        return openai.RateLimitError(message, response=resp, body=None)
    return openai.APIStatusError(message, response=resp, body=None)


class ClaudeChat:
    def __init__(self, api_key: str, timeout: float = 60.0, client: Any = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=timeout)
        self._client = client
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model: str, messages: list[dict], max_tokens: int,
                response_format: Optional[dict] = None, reasoning_effort: str = "", **_: Any) -> Any:
        import anthropic

        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [{"role": m["role"], "content": _content(m["content"])} for m in messages if m["role"] != "system"]
        output_config: dict = {}
        if reasoning_effort:
            output_config["effort"] = reasoning_effort
        if response_format and response_format.get("type") == "json_schema":
            output_config["format"] = {"type": "json_schema", "schema": response_format["json_schema"]["schema"]}
        kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": chat}
        if system:
            kwargs["system"] = system
        if output_config:
            kwargs["output_config"] = output_config
        try:
            resp = self._client.messages.create(**kwargs)
        except anthropic.APITimeoutError as e:
            raise openai.APITimeoutError(request=_REQ) from e
        except anthropic.APIConnectionError as e:
            raise openai.APIConnectionError(request=_REQ) from e
        except anthropic.APIStatusError as e:
            msg = str(e)
            if "effort" in msg.lower():
                msg = "reasoning_effort not supported: " + msg    # llm.py then retries without it
            raise _status_error(e.status_code, msg) from e

        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        finish = "length" if resp.stop_reason == "max_tokens" else "stop"
        usage = getattr(resp, "usage", None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text, model_extra={}), finish_reason=finish)],
            usage=SimpleNamespace(prompt_tokens=getattr(usage, "input_tokens", 0),
                                  completion_tokens=getattr(usage, "output_tokens", 0)),
        )


def _content(content: Any) -> Any:
    """OpenAI-style content (text, or a list with text and data-URL images) -> Claude's content blocks."""
    if isinstance(content, str):
        return content
    blocks = []
    for part in content:
        if part.get("type") == "text":
            blocks.append({"type": "text", "text": part["text"]})
        elif part.get("type") == "image_url":
            url = part["image_url"]["url"]
            head, _, data = url.partition(",")
            mime = head.split(":", 1)[-1].split(";", 1)[0] or "image/jpeg"
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
    return blocks

"""
Shared LLM utility — resilient OpenAI chat wrapper.

Handles the max_tokens vs max_completion_tokens difference automatically
using a try/except approach that works for ALL models.
"""
from __future__ import annotations

from typing import Any, Optional

from openai import OpenAI

from .config import settings

# Module-level singleton
_client: Optional[OpenAI] = None


def get_openai_client() -> Optional[OpenAI]:
    """Return a shared OpenAI client, or None if no API key."""
    global _client
    if _client is not None:
        return _client
    if settings.openai_api_key:
        _client = OpenAI(api_key=settings.openai_api_key)
        return _client
    return None


# Track which token param this model needs (learned at runtime)
_use_max_completion_tokens: Optional[bool] = None


def chat_completion(
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 1024,
    client: Optional[OpenAI] = None,
) -> str:
    """Call OpenAI chat completion with automatic token-param fallback.

    Tries max_completion_tokens first; if the API rejects it, retries
    with max_tokens. Caches the working param for subsequent calls.
    """
    global _use_max_completion_tokens

    c = client or get_openai_client()
    if c is None:
        raise RuntimeError("No OpenAI API key configured")

    base_kwargs: dict[str, Any] = {
        "model": settings.openai_model,
        "messages": messages,
        "temperature": temperature,
    }

    # If we already know which param works, use it directly
    if _use_max_completion_tokens is True:
        base_kwargs["max_completion_tokens"] = max_tokens
        resp = c.chat.completions.create(**base_kwargs)
        return resp.choices[0].message.content.strip()

    if _use_max_completion_tokens is False:
        base_kwargs["max_tokens"] = max_tokens
        resp = c.chat.completions.create(**base_kwargs)
        return resp.choices[0].message.content.strip()

    # First call — try max_completion_tokens, fall back to max_tokens
    try:
        resp = c.chat.completions.create(
            **base_kwargs,
            max_completion_tokens=max_tokens,
        )
        _use_max_completion_tokens = True
        return resp.choices[0].message.content.strip()
    except Exception as e:
        err_msg = str(e).lower()
        if "max_tokens" in err_msg or "max_completion_tokens" in err_msg or "unsupported parameter" in err_msg:
            # This model wants the old param
            resp = c.chat.completions.create(
                **base_kwargs,
                max_tokens=max_tokens,
            )
            _use_max_completion_tokens = False
            return resp.choices[0].message.content.strip()
        raise

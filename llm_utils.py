"""
Shared LLM utility — resilient OpenAI chat wrapper.

Handles the max_tokens vs max_completion_tokens difference automatically.
Also handles reasoning models that don't accept temperature.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

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


# Prefixes for reasoning models that reject temperature
_REASONING_PREFIXES = ("o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    """Check if model is a reasoning model (rejects temperature)."""
    m = model.lower().strip()
    # o1, o1-mini, o1-preview, o3, o3-mini, o4-mini etc.
    # but NOT gpt-4o, gpt-4o-mini (those accept temperature)
    for prefix in _REASONING_PREFIXES:
        if m == prefix or m.startswith(prefix + "-"):
            return True
    return False


def chat_completion(
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 1024,
    client: Optional[OpenAI] = None,
) -> str:
    """Call OpenAI chat completion with robust parameter handling.

    Strategy:
      1. Detect reasoning models → skip temperature
      2. Try max_completion_tokens first
      3. If that fails (unsupported param), try max_tokens
      4. If both fail, try with no token limit (last resort)
    """
    c = client or get_openai_client()
    if c is None:
        raise RuntimeError("No OpenAI API key configured")

    model = settings.openai_model
    is_reasoning = _is_reasoning_model(model)

    base_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    # Reasoning models reject temperature
    if not is_reasoning:
        base_kwargs["temperature"] = temperature

    # Try token param variants in order
    token_params = ["max_completion_tokens", "max_tokens"]

    last_error = None
    for token_param in token_params:
        try:
            call_kwargs = {**base_kwargs, token_param: max_tokens}
            logger.debug("LLM call attempt: model=%s, token_param=%s, has_temperature=%s",
                         model, token_param, "temperature" in call_kwargs)
            resp = c.chat.completions.create(**call_kwargs)
            logger.debug("LLM call succeeded with %s", token_param)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            err_msg = str(e).lower()
            logger.warning("LLM call failed with %s: %s", token_param, str(e)[:200])
            # Only continue to next variant if the error is about unsupported params
            if "unsupported" in err_msg or "max_tokens" in err_msg or "max_completion_tokens" in err_msg:
                continue
            # If temperature was the issue, retry without it
            if "temperature" in err_msg:
                base_kwargs.pop("temperature", None)
                try:
                    call_kwargs = {**base_kwargs, token_param: max_tokens}
                    resp = c.chat.completions.create(**call_kwargs)
                    return resp.choices[0].message.content.strip()
                except Exception as e2:
                    last_error = e2
                    continue
            raise

    # Last resort: no token limit, no temperature
    try:
        resp = c.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content.strip()
    except Exception:
        pass

    # Re-raise the last meaningful error
    if last_error:
        raise last_error
    raise RuntimeError("All LLM call attempts failed")

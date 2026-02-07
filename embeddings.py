import numpy as np
import hashlib
from typing import Optional

# Global OpenAI client cache
_openai_client = None
_openai_dim: Optional[int] = None


def _get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    try:
        from .config import settings
        if settings.openai_api_key:
            from openai import OpenAI
            _openai_client = OpenAI(api_key=settings.openai_api_key)
            return _openai_client
    except Exception:
        pass
    return None


def embed_text(text: str, dim: int = 384) -> list[float]:
    """Embed text using OpenAI if available, else deterministic hash fallback."""
    text = (text or "").strip()
    if not text:
        text = "empty"

    # Try OpenAI embeddings first (semantic similarity)
    client = _get_openai_client()
    if client is not None:
        try:
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8000],
                dimensions=dim,
            )
            return resp.data[0].embedding
        except Exception:
            pass

    # Fallback: deterministic hash-based (no semantic similarity)
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(dim,)).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-9)
    return v.tolist()

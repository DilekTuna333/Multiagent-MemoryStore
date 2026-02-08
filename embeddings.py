import logging
import numpy as np
import hashlib
from typing import Optional

logger = logging.getLogger(__name__)

# Global client cache
_client = None  # OpenAI | AzureOpenAI
_embedding_model: Optional[str] = None


def _get_client():
    """Return a shared OpenAI/AzureOpenAI client for embeddings."""
    global _client, _embedding_model
    if _client is not None:
        return _client

    try:
        from .config import settings

        # Azure OpenAI (priority)
        if settings.use_azure:
            from openai import AzureOpenAI
            _client = AzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
            )
            _embedding_model = settings.azure_openai_embedding_deployment
            logger.info("Embeddings: using Azure OpenAI deployment=%s", _embedding_model)
            return _client

        # Standard OpenAI (fallback)
        if settings.openai_api_key:
            from openai import OpenAI
            _client = OpenAI(api_key=settings.openai_api_key)
            _embedding_model = "text-embedding-3-small"
            logger.info("Embeddings: using standard OpenAI model=%s", _embedding_model)
            return _client
    except Exception as e:
        logger.error("Failed to create embedding client: %s", e)

    return None


def embed_text(text: str, dim: int = 384) -> list[float]:
    """Embed text using OpenAI/Azure if available, else deterministic hash fallback."""
    text = (text or "").strip()
    if not text:
        text = "empty"

    # Try OpenAI/Azure embeddings first (semantic similarity)
    client = _get_client()
    if client is not None:
        try:
            resp = client.embeddings.create(
                model=_embedding_model or "text-embedding-3-small",
                input=text[:8000],
                dimensions=dim,
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.warning("Embedding API call failed, falling back to hash: %s", e)

    # Fallback: deterministic hash-based (no semantic similarity)
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(dim,)).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-9)
    return v.tolist()

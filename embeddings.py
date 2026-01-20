import numpy as np
import hashlib

def embed_text(text: str, dim: int = 384) -> list[float]:
    """Deterministic, keyless embedding for demo. Replace with real embeddings later."""
    text = (text or "").strip()
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(dim,)).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-9)
    return v.tolist()

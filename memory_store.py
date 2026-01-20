from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from .config import settings
from .embeddings import embed_text
import time
import uuid

@dataclass
class MemoryHit:
    id: str
    score: float
    text: str
    meta: dict[str, Any]

class MemoryStore:
    """Qdrant-backed memory store with per-scope collections."""
    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url)
        self.dim = settings.embed_dim

    def ensure_collection(self, name: str):
        existing = [c.name for c in self.client.get_collections().collections]
        if name in existing:
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
        )

    def upsert_text(
        self,
        collection: str,
        text: str,
        meta: Optional[dict[str, Any]] = None,
        point_id: Optional[str] = None,
    ) -> str:
        self.ensure_collection(collection)
        pid = point_id or str(uuid.uuid4())
        payload = {
            "text": text,
            "meta": meta or {},
            "ts": int(time.time()),
        }
        vec = embed_text(text, self.dim)
        self.client.upsert(
            collection_name=collection,
            points=[qm.PointStruct(id=pid, vector=vec, payload=payload)],
        )
        return pid

    def search(
        self,
        collection: str,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.2,
        filter_meta: Optional[dict[str, Any]] = None,
    ) -> list[MemoryHit]:
        self.ensure_collection(collection)
        vec = embed_text(query, self.dim)

        qfilter = None
        if filter_meta:
            conditions = []
            for k, v in filter_meta.items():
                conditions.append(
                    qm.FieldCondition(
                        key=f"meta.{k}",
                        match=qm.MatchValue(value=v),
                    )
                )
            qfilter = qm.Filter(must=conditions)

        res = self.client.search(
            collection_name=collection,
            query_vector=vec,
            limit=limit,
            with_payload=True,
            score_threshold=score_threshold,
            query_filter=qfilter,
        )
        hits: list[MemoryHit] = []
        for r in res:
            payload = r.payload or {}
            hits.append(
                MemoryHit(
                    id=str(r.id),
                    score=float(r.score),
                    text=str(payload.get("text", "")),
                    meta=dict(payload.get("meta", {})),
                )
            )
        return hits

    def list_recent(self, collection: str, limit: int = 20) -> list[dict[str, Any]]:
        self.ensure_collection(collection)
        points, _ = self.client.scroll(
            collection_name=collection,
            limit=limit,
            with_payload=True,
        )
        items = []
        for p in points:
            payload = p.payload or {}
            items.append(
                {
                    "id": str(p.id),
                    "text": payload.get("text", ""),
                    "meta": payload.get("meta", {}),
                    "ts": payload.get("ts", 0),
                }
            )
        items.sort(key=lambda x: x["ts"], reverse=True)
        return items[:limit]

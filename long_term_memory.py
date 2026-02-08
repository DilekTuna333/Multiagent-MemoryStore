"""
Long-Term Memory System with Procedural, Episodic, and Semantic categories.

Uses Qdrant as vector store with per-agent, per-category collections.
Integrates with OpenAI for memory categorization and embedding.

Memory Categories:
  - PROCEDURAL: How-to knowledge, workflows, step sequences, rules
  - EPISODIC: Specific conversation events, user requests, outcomes
  - SEMANTIC: Facts, definitions, entity relationships, domain knowledge

Retrieval Strategy (by category):
  - SEMANTIC + EPISODIC: filtered by user_id (personal data)
  - PROCEDURAL: NO user_id filter (universal domain knowledge)
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .config import settings
from .embeddings import embed_text
from .llm_utils import get_openai_client, chat_completion


class MemoryCategory(str, Enum):
    PROCEDURAL = "procedural"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


# Categories that contain personal/user-specific data → filter by user_id
_PERSONAL_CATEGORIES = {MemoryCategory.EPISODIC, MemoryCategory.SEMANTIC}


@dataclass
class LTMEntry:
    id: str
    text: str
    category: MemoryCategory
    agent: str
    session_id: str
    user_id: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: int = 0


CATEGORY_SYSTEM_PROMPT = """Sen bir hafıza sınıflandırma uzmanısın. Verilen konuşma parçasını analiz et ve uygun hafıza kategorisine sınıflandır.

Kategoriler:
1. PROCEDURAL - Nasıl yapılır bilgisi, iş akışları, adım dizileri, kurallar, prosedürler.
   Örnek: "İzin girişi yapmak için önce tarih aralığını belirle, sonra onay sürecini başlat"

2. EPISODIC - Belirli bir konuşma olayı, kullanıcının talebi, sonuçlar, deneyimler.
   Örnek: "Kullanıcı 5M TL kredi limiti sordu, DSCR analizi yapıldı, %1.8 çıktı"

3. SEMANTIC - Gerçekler, tanımlar, varlık ilişkileri, kişisel bilgiler, alan bilgisi, genel bilgi.
   Örnek: "Kullanıcının göz rengi mavi" veya "KOBİ segmenti için POS komisyon oranı %1.2'dir"

Sadece şu JSON formatında yanıt ver:
{"category": "PROCEDURAL|EPISODIC|SEMANTIC", "confidence": 0.0-1.0, "summary": "kısa özet"}
"""


class LongTermMemory:
    """Per-agent, per-category long-term memory backed by Qdrant."""

    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url)
        self.dim = settings.embed_dim
        self._openai = get_openai_client()

    def _collection_name(self, agent: str, category: MemoryCategory) -> str:
        return f"{settings.mem0_collection_prefix}_{agent}_{category.value}"

    def _ensure_collection(self, name: str):
        existing = [c.name for c in self.client.get_collections().collections]
        if name not in existing:
            self.client.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            )

    def categorize(self, text: str) -> tuple[MemoryCategory, float, str]:
        """Classify text into a memory category using LLM or fallback heuristics."""
        if self._openai:
            return self._categorize_with_llm(text)
        return self._categorize_heuristic(text)

    def _categorize_with_llm(self, text: str) -> tuple[MemoryCategory, float, str]:
        try:
            raw = chat_completion(
                messages=[
                    {"role": "system", "content": CATEGORY_SYSTEM_PROMPT},
                    {"role": "user", "content": text[:2000]},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            if "{" in raw:
                raw = raw[raw.index("{"):raw.rindex("}") + 1]
            parsed = json.loads(raw)
            cat_str = parsed.get("category", "EPISODIC").upper()
            cat = MemoryCategory(cat_str.lower())
            confidence = float(parsed.get("confidence", 0.7))
            summary = parsed.get("summary", text[:100])
            return cat, confidence, summary
        except Exception:
            return self._categorize_heuristic(text)

    def _categorize_heuristic(self, text: str) -> tuple[MemoryCategory, float, str]:
        """Rule-based fallback categorization."""
        t = text.lower()

        procedural_keywords = [
            "süreç", "prosedür", "nasıl yapılır", "yapılır", "kural",
            "workflow", "akış", "sıra ile", "onay ver", "giriş yap", "başvur",
            "hesapla", "formül", "yöntem", "talimat", "step", "adım adım",
        ]

        # Personal info patterns (Turkish morphology aware)
        personal_patterns = [
            "adım ", "adım,", "ismim", "benim adım", "benim ismim",
            "yaşım", "yaşındayım", "doğdum", "doğum",
            "gözlerim", "göz rengi", "gözüm", "saçım", "saç rengi", "boyum",
            "kilom", "memleketim", "şehrim", "ülkem",
            "mesleğim", "işim", "çalışıyorum",
            "evliyim", "bekarım", "eşim", "çocuğum",
            "numar", "adres", "mail", "e-posta",
            "favori", "sevdiğim", "tercih",
        ]

        semantic_keywords = [
            "oran", "komisyon", "limit", "tanım", "nedir", "anlamı",
            "kur", "faiz", "vade", "segment", "kategori", "tür",
            "politika", "mevzuat", "yönetmelik", "kanun",
            "göz", "renk", "isim", "yaş",
            "benim", "bana", "benimki", "kahverengi", "mavi", "yeşil", "siyah",
        ]

        # Check personal patterns first (strongest signal for SEMANTIC)
        personal_score = sum(1 for p in personal_patterns if p in t)
        if personal_score >= 1:
            return MemoryCategory.SEMANTIC, 0.8, text[:100]

        proc_score = sum(1 for k in procedural_keywords if k in t)
        sem_score = sum(1 for k in semantic_keywords if k in t)

        if proc_score > sem_score and proc_score >= 2:
            return MemoryCategory.PROCEDURAL, 0.6, text[:100]
        elif sem_score > proc_score and sem_score >= 1:
            return MemoryCategory.SEMANTIC, 0.6, text[:100]
        else:
            return MemoryCategory.EPISODIC, 0.5, text[:100]

    def insert(
        self,
        agent: str,
        session_id: str,
        text: str,
        category: Optional[MemoryCategory] = None,
        metadata: Optional[dict[str, Any]] = None,
        user_id: str = "",
    ) -> LTMEntry:
        """Insert a memory, auto-categorizing if no category provided."""
        if category is None:
            category, confidence, summary = self.categorize(text)
        else:
            confidence = 1.0
            summary = text[:100]

        col_name = self._collection_name(agent, category)
        self._ensure_collection(col_name)

        pid = str(uuid.uuid4())
        ts = int(time.time())

        payload = {
            "text": text,
            "summary": summary,
            "category": category.value,
            "agent": agent,
            "session_id": session_id,
            "user_id": user_id,
            "confidence": confidence,
            "ts": ts,
            "meta": metadata or {},
        }

        vec = embed_text(text, self.dim)
        self.client.upsert(
            collection_name=col_name,
            points=[qm.PointStruct(id=pid, vector=vec, payload=payload)],
        )

        return LTMEntry(
            id=pid,
            text=text,
            category=category,
            agent=agent,
            session_id=session_id,
            user_id=user_id,
            score=confidence,
            metadata=metadata or {},
            timestamp=ts,
        )

    def insert_conversation_turn(
        self,
        agent: str,
        session_id: str,
        user_message: str,
        agent_response: str,
        extra_meta: Optional[dict[str, Any]] = None,
        user_id: str = "",
    ) -> list[LTMEntry]:
        """Insert a full conversation turn - categorizes and stores in appropriate buckets."""
        entries = []

        # Store the episodic record (the event itself)
        episodic_text = f"Kullanıcı: {user_message}\nAjan yanıtı: {agent_response}"
        entry = self.insert(
            agent=agent,
            session_id=session_id,
            text=episodic_text,
            category=MemoryCategory.EPISODIC,
            metadata={"type": "conversation_turn", **(extra_meta or {})},
            user_id=user_id,
        )
        entries.append(entry)

        # Extract personal facts from user message and store as SEMANTIC
        personal_facts = self._extract_personal_facts(user_message)
        for fact in personal_facts:
            fact_entry = self.insert(
                agent=agent,
                session_id=session_id,
                text=fact,
                category=MemoryCategory.SEMANTIC,
                metadata={"type": "personal_fact", "source": "user_message", **(extra_meta or {})},
                user_id=user_id,
            )
            entries.append(fact_entry)

        # Auto-categorize combined text for additional non-episodic storage
        if not personal_facts:
            combined = f"{user_message} {agent_response}"
            cat, conf, summary = self.categorize(combined)
            if cat != MemoryCategory.EPISODIC:
                entry2 = self.insert(
                    agent=agent,
                    session_id=session_id,
                    text=combined,
                    category=cat,
                    metadata={"type": "auto_categorized", "source_category": cat.value, **(extra_meta or {})},
                    user_id=user_id,
                )
                entries.append(entry2)

        return entries

    def _extract_personal_facts(self, user_message: str) -> list[str]:
        """Extract personal facts from user message for SEMANTIC storage.

        Looks for patterns like 'adım X', 'yaşım X', 'gözlerim X' etc.
        Returns clean fact strings like 'Kullanıcının adı: Dilek Tuna'
        """
        facts = []
        t = user_message.strip()
        tl = t.lower()

        # Try LLM-based extraction first
        if self._openai:
            llm_facts = self._extract_facts_with_llm(t)
            if llm_facts:
                return llm_facts

        # Heuristic: pattern-based extraction
        # Name patterns
        name_patterns = [
            r"(?:benim\s+)?ad[ıi]m\s+(.+?)(?:\.|,|$)",
            r"(?:benim\s+)?ismim\s+(.+?)(?:\.|,|$)",
            r"ben\s+(.+?)(?:\.|,|$|\s+yaş)",
        ]
        for pat in name_patterns:
            m = re.search(pat, tl)
            if m:
                name_val = m.group(1).strip().rstrip(".,")
                # Use original case from the message
                orig_m = re.search(pat, t, re.IGNORECASE)
                if orig_m:
                    name_val = orig_m.group(1).strip().rstrip(".,")
                if len(name_val) > 1 and len(name_val) < 60:
                    facts.append(f"Kullanıcının adı: {name_val}")
                break

        # Age patterns
        age_patterns = [
            r"yaş[ıi]m\s+(\d+)",
            r"(\d+)\s+yaş[ıi]nd",
        ]
        for pat in age_patterns:
            m = re.search(pat, tl)
            if m:
                facts.append(f"Kullanıcının yaşı: {m.group(1)}")
                break

        # Eye color patterns — handle both "kahverengi gözlerim" and "gözlerim kahverengi"
        _COLORS = ["kahverengi", "mavi", "yeşil", "siyah", "ela", "gri", "kestane"]
        eye_found = False

        # Pattern 1: "<color> gözlerim var" / "<color> gözlü"
        for color in _COLORS:
            if color in tl and ("göz" in tl):
                facts.append(f"Kullanıcının göz rengi: {color}")
                eye_found = True
                break

        if not eye_found:
            # Pattern 2: "gözlerim <color>" / "göz rengim <color>"
            eye_patterns = [
                r"gözlerim\s+(\w+)",
                r"göz\s+reng[im]+\s+(\w+)",
            ]
            for pat in eye_patterns:
                m = re.search(pat, tl)
                if m:
                    color = m.group(1).strip().rstrip(".,")
                    if color not in ("var", "ne", "nasıl", "bir"):
                        facts.append(f"Kullanıcının göz rengi: {color}")
                        break

        # General "X var/dir" patterns for personal attributes
        attr_patterns = [
            (r"saçlar[ıi]m?\s+(.+?)(?:\.|,|$)", "saç rengi/tipi"),
            (r"boyum\s+(.+?)(?:\.|,|$)", "boyu"),
            (r"kilom\s+(.+?)(?:\.|,|$)", "kilosu"),
            (r"mesleğim\s+(.+?)(?:\.|,|$)", "mesleği"),
            (r"memleketim\s+(.+?)(?:\.|,|$)", "memleketi"),
        ]
        for pat, label in attr_patterns:
            m = re.search(pat, tl)
            if m:
                val = m.group(1).strip().rstrip(".,")
                if len(val) > 0 and len(val) < 60:
                    facts.append(f"Kullanıcının {label}: {val}")

        return facts

    def _extract_facts_with_llm(self, text: str) -> list[str]:
        """Use LLM to extract personal/semantic facts from text."""
        try:
            prompt = """Kullanıcının mesajından kişisel bilgileri ve gerçekleri çıkar.
Her bir bilgiyi ayrı satırda, "Kullanıcının X: Y" formatında yaz.
Eğer kişisel bilgi yoksa boş yanıt ver.

Örnekler:
"adım Ali, 25 yaşındayım" → Kullanıcının adı: Ali\nKullanıcının yaşı: 25
"kahverengi gözlerim var" → Kullanıcının göz rengi: kahverengi
"merhaba nasılsın" → (boş)

Sadece bilgileri listele, başka bir şey yazma."""

            raw = chat_completion(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text[:500]},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            raw = raw.strip()
            if not raw or raw == "(boş)" or len(raw) < 5:
                return []
            facts = [line.strip() for line in raw.split("\n") if line.strip() and ":" in line]
            return facts[:5]
        except Exception:
            return []

    def retrieve(
        self,
        agent: str,
        query: str,
        category: Optional[MemoryCategory] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 5,
        score_threshold: float = 0.15,
    ) -> list[LTMEntry]:
        """Retrieve memories by similarity.

        Smart filtering by category:
          - SEMANTIC/EPISODIC: filter by user_id (personal data)
          - PROCEDURAL: NO user_id filter (universal knowledge)
        """
        categories = [category] if category else list(MemoryCategory)
        all_hits: list[LTMEntry] = []

        vec = embed_text(query, self.dim)

        for cat in categories:
            col_name = self._collection_name(agent, cat)
            self._ensure_collection(col_name)

            # Build filter conditions
            conditions = []
            if session_id:
                conditions.append(
                    qm.FieldCondition(key="session_id", match=qm.MatchValue(value=session_id))
                )
            # Smart user_id filtering: only for personal categories
            if user_id and cat in _PERSONAL_CATEGORIES:
                conditions.append(
                    qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id))
                )

            qfilter = qm.Filter(must=conditions) if conditions else None

            try:
                res = self.client.search(
                    collection_name=col_name,
                    query_vector=vec,
                    limit=limit,
                    with_payload=True,
                    score_threshold=score_threshold,
                    query_filter=qfilter,
                )
            except Exception:
                continue

            for r in res:
                payload = r.payload or {}
                all_hits.append(LTMEntry(
                    id=str(r.id),
                    text=str(payload.get("text", "")),
                    category=cat,
                    agent=payload.get("agent", agent),
                    session_id=payload.get("session_id", ""),
                    user_id=payload.get("user_id", ""),
                    score=float(r.score),
                    metadata=dict(payload.get("meta", {})),
                    timestamp=payload.get("ts", 0),
                ))

        all_hits.sort(key=lambda x: x.score, reverse=True)
        return all_hits[:limit]

    def retrieve_by_category(
        self,
        agent: str,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit_per_category: int = 3,
    ) -> dict[str, list[LTMEntry]]:
        """Retrieve memories grouped by category with smart user_id filtering."""
        result = {}
        for cat in MemoryCategory:
            hits = self.retrieve(
                agent=agent,
                query=query,
                category=cat,
                session_id=session_id,
                user_id=user_id,
                limit=limit_per_category,
            )
            result[cat.value] = hits
        return result

    def list_all_for_agent(
        self,
        agent: str,
        limit_per_category: int = 20,
        user_id: Optional[str] = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """List all recent memories for an agent grouped by category."""
        result = {}
        for cat in MemoryCategory:
            col_name = self._collection_name(agent, cat)
            self._ensure_collection(col_name)

            scroll_filter = None
            if user_id and cat in _PERSONAL_CATEGORIES:
                scroll_filter = qm.Filter(must=[
                    qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id))
                ])

            try:
                points, _ = self.client.scroll(
                    collection_name=col_name,
                    limit=limit_per_category,
                    with_payload=True,
                    scroll_filter=scroll_filter,
                )
                items = []
                for p in points:
                    payload = p.payload or {}
                    items.append({
                        "id": str(p.id),
                        "text": payload.get("text", ""),
                        "summary": payload.get("summary", ""),
                        "category": cat.value,
                        "agent": payload.get("agent", agent),
                        "session_id": payload.get("session_id", ""),
                        "user_id": payload.get("user_id", ""),
                        "ts": payload.get("ts", 0),
                        "meta": payload.get("meta", {}),
                    })
                items.sort(key=lambda x: x["ts"], reverse=True)
                result[cat.value] = items[:limit_per_category]
            except Exception:
                result[cat.value] = []
        return result

    def format_context(self, entries: list[LTMEntry]) -> str:
        """Format retrieved memories into a context string for the agent."""
        if not entries:
            return ""

        grouped: dict[str, list[LTMEntry]] = {}
        for e in entries:
            grouped.setdefault(e.category.value, []).append(e)

        lines = ["=== Uzun Süreli Hafıza (Long-Term Memory) ==="]

        cat_labels = {
            "procedural": "Prosedürel (Nasıl Yapılır)",
            "episodic": "Episodik (Geçmiş Olaylar)",
            "semantic": "Semantik (Bilgi/Gerçekler)",
        }

        for cat_val, cat_entries in grouped.items():
            lines.append(f"\n{cat_labels.get(cat_val, cat_val)}:")
            for e in cat_entries:
                lines.append(f"  - [skor:{e.score:.2f}] {e.text[:200]}")

        return "\n".join(lines)

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI

from ..config import settings
from ..memory_store import MemoryStore
from ..long_term_memory import LongTermMemory, MemoryCategory

# Models that require max_completion_tokens instead of max_tokens
_NEW_TOKEN_MODELS = {"o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"}


def _token_param(n: int) -> dict:
    """Return the right token-limit kwarg for the configured model."""
    model = settings.openai_model.lower()
    if any(model.startswith(p) for p in _NEW_TOKEN_MODELS):
        return {"max_completion_tokens": n}
    return {"max_tokens": n}


@dataclass
class AgentResult:
    agent_name: str
    answer: str
    notes: dict[str, Any] = field(default_factory=dict)
    ltm_entries: list[dict[str, Any]] = field(default_factory=list)


class BaseAgent:
    name: str = "base"
    memory_collection: str = "base_memory"
    shared_collection: str = "shared_memory"
    system_prompt: str = "Sen yardımcı bir asistansın."

    def __init__(self, memory: MemoryStore, ltm: LongTermMemory):
        self.memory = memory
        self.ltm = ltm
        self._openai: Optional[OpenAI] = None
        if settings.openai_api_key:
            self._openai = OpenAI(api_key=settings.openai_api_key)

    # --- Short-term memory (Qdrant per-session) ---

    def retrieve_private_context(self, user_message: str, session_id: str) -> str:
        hits = self.memory.search(
            self.memory_collection,
            query=user_message,
            limit=5,
            score_threshold=0.2,
            filter_meta={"session_id": session_id},
        )
        if not hits:
            return ""
        lines = [f"- ({h.score:.2f}) {h.text}" for h in hits]
        return "Ajan kısa süreli hafızası:\n" + "\n".join(lines)

    def retrieve_shared_context(self, user_message: str, session_id: str) -> str:
        hits = self.memory.search(
            self.shared_collection,
            query=user_message,
            limit=5,
            score_threshold=0.2,
            filter_meta={"session_id": session_id},
        )
        if not hits:
            return ""
        lines = [f"- ({h.score:.2f}) {h.text}" for h in hits]
        return "Ortak hafızadan:\n" + "\n".join(lines)

    def write_private_memory(self, session_id: str, text: str, meta: dict[str, Any] | None = None):
        self.memory.upsert_text(
            self.memory_collection,
            text=text,
            meta={"session_id": session_id, **(meta or {})},
        )

    def write_structured_fact(self, session_id: str, fact: dict, meta: dict[str, Any] | None = None):
        text = "FACT_JSON: " + json.dumps(fact, ensure_ascii=False)
        self.memory.upsert_text(
            self.memory_collection,
            text=text,
            meta={
                "session_id": session_id,
                "type": "fact_structured",
                "fact": fact,
                **(meta or {}),
            },
        )

    # --- Long-term memory ---

    def retrieve_ltm_context(self, query: str, session_id: Optional[str] = None) -> str:
        """Retrieve long-term memories across all categories."""
        entries = self.ltm.retrieve(
            agent=self.name,
            query=query,
            session_id=None,  # LTM is cross-session
            limit=6,
        )
        return self.ltm.format_context(entries)

    def store_to_ltm(
        self,
        session_id: str,
        user_message: str,
        agent_response: str,
        extra_meta: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Store conversation turn into long-term memory with auto-categorization."""
        entries = self.ltm.insert_conversation_turn(
            agent=self.name,
            session_id=session_id,
            user_message=user_message,
            agent_response=agent_response,
            extra_meta=extra_meta,
        )
        return [{"id": e.id, "category": e.category.value, "text": e.text[:100]} for e in entries]

    # --- LLM call ---

    def call_llm(self, user_message: str, context: str = "") -> str:
        """Call OpenAI LLM with agent's system prompt + context. Falls back to template."""
        if not self._openai:
            return self._fallback_response(user_message)

        messages = [{"role": "system", "content": self.system_prompt}]
        if context:
            messages.append({"role": "system", "content": f"Bağlam bilgisi:\n{context}"})
        messages.append({"role": "user", "content": user_message})

        try:
            resp = self._openai.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                temperature=0.7,
                **_token_param(1024),
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return self._fallback_response(user_message) + f"\n(LLM hatası: {e})"

    def _fallback_response(self, user_message: str) -> str:
        return f"[{self.name}] Demo yanıt - LLM API anahtarı yapılandırılmamış."

    # --- Main run ---

    def run(self, user_message: str, session_id: str) -> AgentResult:
        raise NotImplementedError

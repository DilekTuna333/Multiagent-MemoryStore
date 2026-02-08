from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import settings
from ..memory_store import MemoryStore
from ..long_term_memory import LongTermMemory, MemoryCategory
from ..llm_utils import get_openai_client, chat_completion


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
        self._openai = get_openai_client()

    # --- Short-term memory (Qdrant per-session) ---

    def retrieve_private_context(
        self, user_message: str, session_id: str, user_id: str = ""
    ) -> str:
        filter_meta: dict[str, Any] = {"session_id": session_id}
        if user_id:
            filter_meta["user_id"] = user_id
        hits = self.memory.search(
            self.memory_collection,
            query=user_message,
            limit=5,
            score_threshold=0.2,
            filter_meta=filter_meta,
        )
        if not hits:
            return ""
        lines = [f"- ({h.score:.2f}) {h.text}" for h in hits]
        return "Ajan kısa süreli hafızası:\n" + "\n".join(lines)

    def retrieve_shared_context(
        self, user_message: str, session_id: str, user_id: str = ""
    ) -> str:
        filter_meta: dict[str, Any] = {"session_id": session_id}
        if user_id:
            filter_meta["user_id"] = user_id
        hits = self.memory.search(
            self.shared_collection,
            query=user_message,
            limit=5,
            score_threshold=0.2,
            filter_meta=filter_meta,
        )
        if not hits:
            return ""
        lines = [f"- ({h.score:.2f}) {h.text}" for h in hits]
        return "Ortak hafızadan:\n" + "\n".join(lines)

    def write_private_memory(
        self, session_id: str, text: str, user_id: str = "",
        meta: dict[str, Any] | None = None,
    ):
        base = {"session_id": session_id, **(meta or {})}
        if user_id:
            base["user_id"] = user_id
        self.memory.upsert_text(self.memory_collection, text=text, meta=base)

    def write_structured_fact(
        self, session_id: str, fact: dict, user_id: str = "",
        meta: dict[str, Any] | None = None,
    ):
        text = "FACT_JSON: " + json.dumps(fact, ensure_ascii=False)
        base = {
            "session_id": session_id,
            "type": "fact_structured",
            "fact": fact,
            **(meta or {}),
        }
        if user_id:
            base["user_id"] = user_id
        self.memory.upsert_text(self.memory_collection, text=text, meta=base)

    # --- Long-term memory ---

    def retrieve_ltm_context(self, query: str, user_id: str = "") -> str:
        """Retrieve LTM with smart user_id filtering.

        SEMANTIC/EPISODIC → filtered by user_id (personal data).
        PROCEDURAL → no user_id filter (universal knowledge).
        """
        entries = self.ltm.retrieve(
            agent=self.name,
            query=query,
            session_id=None,  # LTM is cross-session
            user_id=user_id or None,
            limit=6,
        )
        return self.ltm.format_context(entries)

    def store_to_ltm(
        self,
        session_id: str,
        user_message: str,
        agent_response: str,
        user_id: str = "",
        extra_meta: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        entries = self.ltm.insert_conversation_turn(
            agent=self.name,
            session_id=session_id,
            user_message=user_message,
            agent_response=agent_response,
            extra_meta=extra_meta,
            user_id=user_id,
        )
        return [{"id": e.id, "category": e.category.value, "text": e.text[:100]} for e in entries]

    # --- LLM call ---

    def call_llm(self, user_message: str, context: str = "") -> str:
        if not self._openai:
            return self._fallback_response(user_message)

        messages = [{"role": "system", "content": self.system_prompt}]
        if context:
            messages.append({"role": "system", "content": f"Bağlam bilgisi:\n{context}"})
        messages.append({"role": "user", "content": user_message})

        try:
            return chat_completion(messages, temperature=0.7, max_tokens=1024)
        except Exception as e:
            return self._fallback_response(user_message) + f"\n(LLM hatası: {e})"

    def _fallback_response(self, user_message: str) -> str:
        return f"[{self.name}] Demo yanıt - LLM API anahtarı yapılandırılmamış."

    # --- Main run ---

    def run(self, user_message: str, session_id: str, user_id: str = "") -> AgentResult:
        raise NotImplementedError

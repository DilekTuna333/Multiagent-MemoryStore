from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import json
from ..memory_store import MemoryStore

@dataclass
class AgentResult:
    agent_name: str
    answer: str
    notes: dict[str, Any]

class BaseAgent:
    name: str = "base"
    memory_collection: str = "base_memory"
    shared_collection: str = "shared_memory"

    def __init__(self, memory: MemoryStore):
        self.memory = memory

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
        return "Ajan private hafızasından ilgili anılar:\n" + "\n".join(lines)

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
        return "Ortak hafızadan ilgili anılar:\n" + "\n".join(lines)

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

    def run(self, user_message: str, session_id: str) -> AgentResult:
        raise NotImplementedError

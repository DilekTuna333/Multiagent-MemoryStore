from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import re
from .memory_store import MemoryStore
from .agents.ik_agent import IKAgent
from .agents.kredi_agent import TicariKrediAgent
from .agents.kampanya_agent import KampanyalarAgent

@dataclass
class SupervisorResponse:
    chosen_agent: str
    answer: str
    debug: dict[str, Any]

class SupervisorDeepAgent:
    """Planner + Execute with strict memory governance."""
    def __init__(self, memory: MemoryStore):
        self.memory = memory

        # Registry only for routing/execution
        self.agents = {
            "ik": IKAgent(memory),
            "ticari_kredi": TicariKrediAgent(memory),
            "kampanyalar": KampanyalarAgent(memory),
        }

        # Supervisor allowed scopes
        self.shared_collection = "shared_memory"
        self.supervisor_collection = "supervisor_memory"

    def plan(self, user_message: str) -> str:
        t = user_message.lower()
        if any(k in t for k in ["izin", "yıllık izin", "mazeret", "rapor", "izin giri"]):
            return "ik"
        if any(k in t for k in ["kredi", "limit", "teminat", "nakit akış", "dscr", "borç"]):
            return "ticari_kredi"
        if any(k in t for k in ["kampanya", "teklif", "paket", "indirim", "promosyon"]):
            return "kampanyalar"
        return "ticari_kredi"

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

    def retrieve_supervisor_context(self, user_message: str, session_id: str) -> str:
        hits = self.memory.search(
            self.supervisor_collection,
            query=user_message,
            limit=5,
            score_threshold=0.2,
            filter_meta={"session_id": session_id},
        )
        if not hits:
            return ""
        lines = [f"- ({h.score:.2f}) {h.text}" for h in hits]
        return "Supervisor private hafızasından ilgili notlar:\n" + "\n".join(lines)

    def write_routing_audit(self, session_id: str, user_message: str, chosen_agent: str, signals: dict[str, Any] | None = None):
        audit = {
            "event": "routing_decision",
            "chosen_agent": chosen_agent,
            "signals": signals or {},
            "user_message_preview": user_message[:240],
        }
        self.memory.upsert_text(
            self.supervisor_collection,
            text=f"AUDIT: routed_to={chosen_agent} preview={user_message[:120]}",
            meta={"session_id": session_id, "type": "audit", "audit": audit},
        )

    def summarize_turn(self, user_message: str, agent_answer: str) -> str:
        um = user_message.strip().replace("\n", " ")
        aa = agent_answer.strip().replace("\n", " ")
        return f"User: {um[:200]} | Agent: {aa[:240]}"

    def extract_memories_for_shared_and_supervisor(self, user_message: str, agent_answer: str) -> list[str]:
        text = f"{user_message}\n{agent_answer}"
        memories = []

        for m in re.findall(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s*[-–]\s*\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", text):
            memories.append(f"Tarih aralığı konuşuldu: {m}")

        for m in re.findall(r"\b(\d{1,3}(?:[.,]\d{3})+|\d+)\s*(tl|try|₺|usd|eur)?\b", text.lower()):
            num, cur = m
            if len(num) >= 4:
                memories.append(f"Tutar geçti: {num} {cur}".strip())

        kw = ["kalan izin", "limit önerisi", "teminat", "kampanya", "pos", "ticari kart"]
        for k in kw:
            if k in text.lower():
                memories.append(f"Ana konu: {k}")

        out, seen = [], set()
        for x in memories:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out[:8]

    def update_supervisor_and_shared_memories(self, session_id: str, agent_name: str, user_message: str, agent_answer: str):
        summary = self.summarize_turn(user_message, agent_answer)
        extracted = self.extract_memories_for_shared_and_supervisor(user_message, agent_answer)

        self.memory.upsert_text(
            self.shared_collection,
            text=f"SUMMARY: {summary}",
            meta={"session_id": session_id, "type": "summary", "agent": agent_name},
        )

        self.memory.upsert_text(
            self.supervisor_collection,
            text=f"SUMMARY: {summary}",
            meta={"session_id": session_id, "type": "summary", "agent": agent_name},
        )

        for mem in extracted:
            self.memory.upsert_text(
                self.shared_collection,
                text=f"MEM: {mem}",
                meta={"session_id": session_id, "type": "fact", "agent": agent_name},
            )
            self.memory.upsert_text(
                self.supervisor_collection,
                text=f"MEM: {mem}",
                meta={"session_id": session_id, "type": "fact", "agent": agent_name},
            )

    def handle(self, session_id: str, user_message: str) -> SupervisorResponse:
        chosen = self.plan(user_message)

        signals = {
            "matched_keywords": [k for k in ["izin","kredi","limit","teminat","kampanya","teklif"] if k in user_message.lower()],
        }
        self.write_routing_audit(session_id, user_message, chosen, signals=signals)

        shared_ctx = self.retrieve_shared_context(user_message, session_id)
        sup_ctx = self.retrieve_supervisor_context(user_message, session_id)

        routed_message = f"{shared_ctx}\n\n{sup_ctx}\n\n{user_message}".strip()

        agent = self.agents[chosen]
        result = agent.run(user_message=routed_message, session_id=session_id)

        self.update_supervisor_and_shared_memories(session_id, chosen, user_message, result.answer)

        return SupervisorResponse(
            chosen_agent=chosen,
            answer=result.answer,
            debug={
                "shared_ctx_present": bool(shared_ctx),
                "supervisor_ctx_present": bool(sup_ctx),
            },
        )

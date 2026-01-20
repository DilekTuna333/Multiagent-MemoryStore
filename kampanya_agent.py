from __future__ import annotations
from .base import BaseAgent, AgentResult

class KampanyalarAgent(BaseAgent):
    name = "kampanyalar"
    memory_collection = "kampanya_memory"

    def run(self, user_message: str, session_id: str) -> AgentResult:
        shared_ctx = self.retrieve_shared_context(user_message, session_id)
        priv_ctx = self.retrieve_private_context(user_message, session_id)

        answer = (
            f"{shared_ctx}\n\n{priv_ctx}\n\n"
            "Kampanyalar Agent yanıtı (demo):\n"
            "- Müşteri segmenti ve davranışına göre 3 kampanya önerisi üretir.\n"
            "Örnek:\n"
            "1) POS komisyon indirimi + ilk 3 ay ciro hedefi\n"
            "2) Ticari kartta taksit ve aidat muafiyeti\n"
            "3) KOBİ paket: EFT/havale ücretsiz + çek karnesi avantajı"
        ).strip()

        self.write_private_memory(
            session_id,
            f"TURN: {user_message[:200]} | ANSWER: {answer[:220]}",
            meta={"type": "turn"},
        )

        fact = {
            "domain": "kampanyalar",
            "entity": "campaign_request",
            "intent": "suggest_offers",
            "segment_hint": "kobi" if "kobi" in user_message.lower() else None,
        }
        self.write_structured_fact(session_id, fact, meta={"source": "rule_based"})

        return AgentResult(agent_name=self.name, answer=answer, notes={"intent": "kampanya"})

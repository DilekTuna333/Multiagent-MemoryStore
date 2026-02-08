from __future__ import annotations

from .base import BaseAgent, AgentResult


class KampanyalarAgent(BaseAgent):
    name = "kampanyalar"
    memory_collection = "kampanya_memory"
    system_prompt = (
        "Sen bankacılık kampanya ve ürün teklif uzmanısın. Müşteri segmentine göre "
        "uygun kampanya önerileri, POS komisyon indirimleri, ticari kart teklifleri, "
        "KOBİ paketleri ve promosyon fırsatları konusunda yardımcı olursun. Yanıtlarını Türkçe ver."
    )

    def _fallback_response(self, user_message: str) -> str:
        return (
            "Kampanyalar Agent yanıtı (demo):\n"
            "- Müşteri segmenti ve davranışına göre 3 kampanya önerisi üretir.\n"
            "Örnek:\n"
            "1) POS komisyon indirimi + ilk 3 ay ciro hedefi\n"
            "2) Ticari kartta taksit ve aidat muafiyeti\n"
            "3) KOBİ paket: EFT/havale ücretsiz + çek karnesi avantajı"
        )

    def run(self, user_message: str, session_id: str, user_id: str = "") -> AgentResult:
        shared_ctx = self.retrieve_shared_context(user_message, session_id, user_id)
        priv_ctx = self.retrieve_private_context(user_message, session_id, user_id)
        ltm_ctx = self.retrieve_ltm_context(user_message, user_id)

        context = "\n\n".join(filter(None, [shared_ctx, priv_ctx, ltm_ctx]))
        answer = self.call_llm(user_message, context)

        # Short-term memory
        self.write_private_memory(
            session_id,
            f"TURN: {user_message[:200]} | ANSWER: {answer[:220]}",
            user_id=user_id,
            meta={"type": "turn"},
        )

        fact = {
            "domain": "kampanyalar",
            "entity": "campaign_request",
            "intent": "suggest_offers",
            "segment_hint": "kobi" if "kobi" in user_message.lower() else None,
        }
        self.write_structured_fact(session_id, fact, user_id=user_id, meta={"source": "rule_based"})

        # Long-term memory
        ltm_entries = self.store_to_ltm(session_id, user_message, answer, user_id=user_id)

        return AgentResult(
            agent_name=self.name,
            answer=answer,
            notes={"intent": "kampanya"},
            ltm_entries=ltm_entries,
        )

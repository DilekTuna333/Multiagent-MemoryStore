from __future__ import annotations

from .base import BaseAgent, AgentResult


class IKAgent(BaseAgent):
    name = "ik"
    memory_collection = "ik_memory"
    system_prompt = (
        "Sen bir İnsan Kaynakları (İK) uzman asistanısın. Bankacılık sektöründe çalışan personelin "
        "izin yönetimi, mazeret izinleri, yıllık izin hesaplamaları, rapor girişleri ve İK mevzuatı "
        "konularında yardımcı olursun. Yanıtlarını Türkçe ver."
    )

    def _fallback_response(self, user_message: str) -> str:
        return (
            "IK Agent yanıtı:\n"
            "- İzin bakiyesi için: sistemde kalan izin günleri personel kartından ve yıllık izin kayıtlarından çekilir.\n"
            "- Demo modda: 'Kalan izin: 12 gün, bu yıl kullanılan: 8 gün' varsayımsal bilgi döndürüyorum.\n"
            "- Eğer 'izin girişi' istersen, başlangıç/bitiş tarihlerini yaz; kayda alacağım."
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

        # Structured fact
        fact = {
            "domain": "ik",
            "entity": "leave_inquiry",
            "intent": "balance_or_request",
            "requested_action": "query_balance",
        }
        self.write_structured_fact(session_id, fact, user_id=user_id, meta={"source": "rule_based"})

        # Long-term memory (auto-categorized)
        ltm_entries = self.store_to_ltm(session_id, user_message, answer, user_id=user_id)

        return AgentResult(
            agent_name=self.name,
            answer=answer,
            notes={"intent": "izin"},
            ltm_entries=ltm_entries,
        )

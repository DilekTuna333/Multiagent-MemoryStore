from __future__ import annotations

from .base import BaseAgent, AgentResult


class GenelAgent(BaseAgent):
    """General-purpose agent for chitchat, personal info, and unclassified queries."""
    name = "genel"
    memory_collection = "genel_memory"
    system_prompt = (
        "Sen yardımcı bir bankacılık asistanısın. Kullanıcıyla genel sohbet edebilir, "
        "kişisel bilgilerini hatırlayabilir ve sorularına yanıt verebilirsin. "
        "Eğer kullanıcı daha önce bir bilgi paylaştıysa (isim, göz rengi, tercihler vb.) "
        "ve bu bilgi sana bağlam olarak verilmişse, onu kullanarak yanıt ver. "
        "Yanıtlarını Türkçe ver."
    )

    def _fallback_response(self, user_message: str) -> str:
        return (
            "Merhaba! Ben bankacılık asistanınızım. Size izin, kredi veya kampanya "
            "konularında yardımcı olabilirim. Ayrıca genel sorularınızı da yanıtlayabilirim."
        )

    def run(self, user_message: str, session_id: str) -> AgentResult:
        shared_ctx = self.retrieve_shared_context(user_message, session_id)
        priv_ctx = self.retrieve_private_context(user_message, session_id)
        ltm_ctx = self.retrieve_ltm_context(user_message)

        context = "\n\n".join(filter(None, [shared_ctx, priv_ctx, ltm_ctx]))
        answer = self.call_llm(user_message, context)

        # Short-term memory
        self.write_private_memory(
            session_id,
            f"TURN: {user_message[:200]} | ANSWER: {answer[:220]}",
            meta={"type": "turn"},
        )

        # Long-term memory
        ltm_entries = self.store_to_ltm(session_id, user_message, answer)

        return AgentResult(
            agent_name=self.name,
            answer=answer,
            notes={"intent": "genel"},
            ltm_entries=ltm_entries,
        )

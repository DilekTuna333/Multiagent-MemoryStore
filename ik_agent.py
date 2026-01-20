from __future__ import annotations
from .base import BaseAgent, AgentResult

class IKAgent(BaseAgent):
    name = "ik"
    memory_collection = "ik_memory"

    def run(self, user_message: str, session_id: str) -> AgentResult:
        shared_ctx = self.retrieve_shared_context(user_message, session_id)
        priv_ctx = self.retrieve_private_context(user_message, session_id)

        answer = (
            f"{shared_ctx}\n\n{priv_ctx}\n\n"
            "IK Agent yanıtı:\n"
            "- İzin bakiyesi için: sistemde kalan izin günleri personel kartından ve yıllık izin kayıtlarından çekilir.\n"
            "- Demo modda: 'Kalan izin: 12 gün, bu yıl kullanılan: 8 gün' varsayımsal bilgi döndürüyorum.\n"
            "- Eğer 'izin girişi' istersen, başlangıç/bitiş tarihlerini yaz; kayda alacağım."
        ).strip()

        self.write_private_memory(
            session_id,
            f"TURN: {user_message[:200]} | ANSWER: {answer[:220]}",
            meta={"type": "turn"},
        )

        fact = {
            "domain": "ik",
            "entity": "leave_inquiry",
            "intent": "balance_or_request",
            "requested_action": "query_balance",
        }
        self.write_structured_fact(session_id, fact, meta={"source": "rule_based"})

        return AgentResult(agent_name=self.name, answer=answer, notes={"intent": "izin"})

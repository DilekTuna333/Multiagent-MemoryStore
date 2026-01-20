from __future__ import annotations
from .base import BaseAgent, AgentResult
import re

class TicariKrediAgent(BaseAgent):
    name = "ticari_kredi"
    memory_collection = "kredi_memory"

    def run(self, user_message: str, session_id: str) -> AgentResult:
        shared_ctx = self.retrieve_shared_context(user_message, session_id)
        priv_ctx = self.retrieve_private_context(user_message, session_id)

        answer = (
            f"{shared_ctx}\n\n{priv_ctx}\n\n"
            "Ticari Kredi Agent yanıtı (demo):\n"
            "1) RAG: Şirket/ürün/limit/teminat gibi alanları çıkarır.\n"
            "2) Dış kaynak taraması: Bu demoda web taraması yok; kullanıcıdan finansal özet (ciro, EBITDA, borç) ister.\n"
            "3) Ön analiz: kaldıraç, likidite, DSCR tahmini, limit önerisi ve follow-up soruları üretir.\n\n"
            "Follow-up soruları:\n"
            "- Son 12 ay ciro ve brüt kar?\n"
            "- Toplam banka borcu ve vade dağılımı?\n"
            "- Teminat yapısı (ipotek/çek/senet/garanti) var mı?"
        ).strip()

        self.write_private_memory(
            session_id,
            f"TURN: {user_message[:200]} | ANSWER: {answer[:220]}",
            meta={"type": "turn"},
        )

        # --- structured extraction (demo) ---
        text = user_message.lower()

        limit = None
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*(m|mn|milyon)\s*(tl|try|₺)", text)
        if m:
            val = float(m.group(1).replace(",", "."))
            limit = int(val * 1_000_000)
        else:
            m2 = re.search(r"(\d{4,})\s*(tl|try|₺)", text)
            if m2:
                limit = int(m2.group(1))

        teminat = None
        for t in ["ipotek", "çek", "senet", "garanti", "kefalet"]:
            if t in text:
                teminat = t
                break

        fact = {
            "domain": "ticari_kredi",
            "entity": "credit_request",
            "requested_limit_try": limit,
            "collateral_hint": teminat,
            "needs_followup": True,
        }
        self.write_structured_fact(session_id, fact, meta={"source": "heuristic_extractor"})

        return AgentResult(agent_name=self.name, answer=answer, notes={"intent": "kredi"})

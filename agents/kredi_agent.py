from __future__ import annotations

import re

from .base import BaseAgent, AgentResult


class TicariKrediAgent(BaseAgent):
    name = "ticari_kredi"
    memory_collection = "kredi_memory"
    system_prompt = (
        "Sen ticari bankacılık kredi analiz uzmanısın. Şirketlerin kredi limiti önerileri, "
        "teminat yapısı analizi, DSCR hesabı, nakit akış değerlendirmesi ve risk analizi "
        "konularında yardımcı olursun. Yanıtlarını Türkçe ver. Follow-up soruları sor."
    )

    def _fallback_response(self, user_message: str) -> str:
        return (
            "Ticari Kredi Agent yanıtı (demo):\n"
            "1) RAG: Şirket/ürün/limit/teminat gibi alanları çıkarır.\n"
            "2) Dış kaynak taraması: Bu demoda web taraması yok; kullanıcıdan finansal özet ister.\n"
            "3) Ön analiz: kaldıraç, likidite, DSCR tahmini, limit önerisi ve follow-up soruları üretir.\n\n"
            "Follow-up soruları:\n"
            "- Son 12 ay ciro ve brüt kar?\n"
            "- Toplam banka borcu ve vade dağılımı?\n"
            "- Teminat yapısı (ipotek/çek/senet/garanti) var mı?"
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

        # Structured extraction
        text = user_message.lower()
        limit_val = None
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*(m|mn|milyon)\s*(tl|try|₺)", text)
        if m:
            val = float(m.group(1).replace(",", "."))
            limit_val = int(val * 1_000_000)
        else:
            m2 = re.search(r"(\d{4,})\s*(tl|try|₺)", text)
            if m2:
                limit_val = int(m2.group(1))

        teminat = None
        for t in ["ipotek", "çek", "senet", "garanti", "kefalet"]:
            if t in text:
                teminat = t
                break

        fact = {
            "domain": "ticari_kredi",
            "entity": "credit_request",
            "requested_limit_try": limit_val,
            "collateral_hint": teminat,
            "needs_followup": True,
        }
        self.write_structured_fact(session_id, fact, meta={"source": "heuristic_extractor"})

        # Long-term memory
        ltm_entries = self.store_to_ltm(session_id, user_message, answer)

        return AgentResult(
            agent_name=self.name,
            answer=answer,
            notes={"intent": "kredi"},
            ltm_entries=ltm_entries,
        )

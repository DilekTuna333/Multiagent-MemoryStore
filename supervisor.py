"""
Supervisor Deep Agent Architecture with streaming step-by-step planning.

Flow:
  1. PLAN   - Analyze user intent, classify category, decide routing
  2. RECALL - Retrieve long-term memories (procedural/episodic/semantic)
  3. ROUTE  - Select and delegate to the appropriate agent
  4. EXECUTE- Run the agent with full context injection
  5. MEMORIZE- Store turn into LTM with auto-categorization
  6. RESPOND- Return final answer with debug info

Each step emits SSE events so the UI can display progress in real time.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

from openai import OpenAI

from .config import settings
from .memory_store import MemoryStore
from .long_term_memory import LongTermMemory, MemoryCategory
from .agents.ik_agent import IKAgent
from .agents.kredi_agent import TicariKrediAgent
from .agents.kampanya_agent import KampanyalarAgent
from .agents.genel_agent import GenelAgent

# Models that require max_completion_tokens instead of max_tokens
_NEW_TOKEN_MODELS = {"o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"}


def _token_param(n: int) -> dict:
    model = settings.openai_model.lower()
    if any(model.startswith(p) for p in _NEW_TOKEN_MODELS):
        return {"max_completion_tokens": n}
    return {"max_tokens": n}


@dataclass
class PlanStep:
    step: str
    status: str  # "pending" | "running" | "done" | "error"
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0


@dataclass
class SupervisorResponse:
    chosen_agent: str
    answer: str
    debug: dict[str, Any]
    plan_steps: list[dict[str, Any]] = field(default_factory=list)
    ltm_entries: list[dict[str, Any]] = field(default_factory=list)


ROUTING_SYSTEM_PROMPT = """Sen bir banka supervisor agentsın. Kullanıcının mesajını analiz et ve hangi ajana yönlendirileceğine karar ver.

Ajanlar:
1. ik - İnsan Kaynakları: İzin, mazeret, rapor, personel konuları
2. ticari_kredi - Ticari Kredi: Kredi, limit, teminat, nakit akış, DSCR, borç analizi
3. kampanyalar - Kampanyalar: Kampanya, teklif, paket, indirim, promosyon
4. genel - Genel Asistan: Selamlaşma, sohbet, kişisel bilgiler, sınıflandırılamayan konular

Eğer mesaj açıkça bir bankacılık konusuyla ilgili değilse "genel" seç.

JSON formatında yanıt ver:
{"agent": "ik|ticari_kredi|kampanyalar|genel", "confidence": 0.0-1.0, "reasoning": "kısa neden"}
"""


class SupervisorDeepAgent:
    """Deep Agent supervisor with step-by-step planning and streaming."""

    def __init__(self, memory: MemoryStore, ltm: LongTermMemory):
        self.memory = memory
        self.ltm = ltm
        self._openai: Optional[OpenAI] = None
        if settings.openai_api_key:
            self._openai = OpenAI(api_key=settings.openai_api_key)

        self.agents = {
            "ik": IKAgent(memory, ltm),
            "ticari_kredi": TicariKrediAgent(memory, ltm),
            "kampanyalar": KampanyalarAgent(memory, ltm),
            "genel": GenelAgent(memory, ltm),
        }

        self.shared_collection = "shared_memory"
        self.supervisor_collection = "supervisor_memory"

    # ─── STEP 1: PLAN ────────────────────────────────────────

    def plan(self, user_message: str) -> tuple[str, dict[str, Any]]:
        """Decide which agent to route to. Uses LLM if available, else keywords."""
        if self._openai:
            return self._plan_with_llm(user_message)
        return self._plan_with_keywords(user_message)

    def _plan_with_llm(self, user_message: str) -> tuple[str, dict[str, Any]]:
        try:
            resp = self._openai.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message[:1000]},
                ],
                temperature=0.1,
                **_token_param(200),
            )
            raw = resp.choices[0].message.content.strip()
            if "{" in raw:
                raw = raw[raw.index("{"):raw.rindex("}") + 1]
            parsed = json.loads(raw)
            agent = parsed.get("agent", "genel")
            if agent not in self.agents:
                agent = "genel"
            return agent, {
                "method": "llm",
                "confidence": parsed.get("confidence", 0.8),
                "reasoning": parsed.get("reasoning", ""),
            }
        except Exception:
            return self._plan_with_keywords(user_message)

    def _plan_with_keywords(self, user_message: str) -> tuple[str, dict[str, Any]]:
        t = user_message.lower()
        if any(k in t for k in ["izin", "yıllık izin", "mazeret", "rapor", "izin giri"]):
            return "ik", {"method": "keyword", "matched": "ik_keywords"}
        if any(k in t for k in ["kredi", "limit", "teminat", "nakit akış", "dscr", "borç"]):
            return "ticari_kredi", {"method": "keyword", "matched": "kredi_keywords"}
        if any(k in t for k in ["kampanya", "teklif", "paket", "indirim", "promosyon"]):
            return "kampanyalar", {"method": "keyword", "matched": "kampanya_keywords"}
        return "genel", {"method": "keyword", "matched": "default_general"}

    # ─── STEP 2: RECALL ──────────────────────────────────────

    def recall_memories(self, user_message: str, session_id: str, agent_name: str) -> dict[str, Any]:
        """Retrieve memories from all layers."""
        shared_ctx = self._retrieve_shared_context(user_message, session_id)
        sup_ctx = self._retrieve_supervisor_context(user_message, session_id)

        # Long-term memory retrieval grouped by category
        ltm_by_cat = self.ltm.retrieve_by_category(
            agent=agent_name,
            query=user_message,
            limit_per_category=3,
        )
        ltm_flat = []
        for cat, entries in ltm_by_cat.items():
            for e in entries:
                ltm_flat.append({"category": cat, "score": e.score, "text": e.text[:150]})

        # Also retrieve supervisor-level LTM
        sup_ltm = self.ltm.retrieve(
            agent="supervisor",
            query=user_message,
            limit=3,
        )

        ltm_ctx = self.ltm.format_context(
            [e for entries in ltm_by_cat.values() for e in entries] + sup_ltm
        )

        return {
            "shared_ctx": shared_ctx,
            "supervisor_ctx": sup_ctx,
            "ltm_ctx": ltm_ctx,
            "ltm_details": ltm_flat,
        }

    def _retrieve_shared_context(self, user_message: str, session_id: str) -> str:
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

    def _retrieve_supervisor_context(self, user_message: str, session_id: str) -> str:
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
        return "Supervisor hafızasından:\n" + "\n".join(lines)

    # ─── STEP 3: EXECUTE ─────────────────────────────────────

    def execute_agent(self, agent_name: str, user_message: str, session_id: str, context: str) -> dict[str, Any]:
        """Augment message with context and execute agent."""
        routed_message = f"{context}\n\n{user_message}".strip() if context else user_message
        agent = self.agents[agent_name]
        result = agent.run(user_message=routed_message, session_id=session_id)
        return {
            "answer": result.answer,
            "notes": result.notes,
            "ltm_entries": result.ltm_entries,
        }

    # ─── STEP 4: MEMORIZE ────────────────────────────────────

    def memorize(self, session_id: str, agent_name: str, user_message: str, answer: str) -> list[dict[str, Any]]:
        """Store memories at supervisor level."""
        # Short-term audit
        self._write_routing_audit(session_id, user_message, agent_name)

        # Short-term shared + supervisor
        summary = self._summarize_turn(user_message, answer)
        extracted = self._extract_facts(user_message, answer)

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

        # Long-term memory for supervisor
        sup_entries = self.ltm.insert_conversation_turn(
            agent="supervisor",
            session_id=session_id,
            user_message=user_message,
            agent_response=answer,
            extra_meta={"routed_to": agent_name},
        )

        return [{"id": e.id, "category": e.category.value} for e in sup_entries]

    def _write_routing_audit(self, session_id: str, user_message: str, chosen_agent: str):
        self.memory.upsert_text(
            self.supervisor_collection,
            text=f"AUDIT: routed_to={chosen_agent} preview={user_message[:120]}",
            meta={"session_id": session_id, "type": "audit", "chosen_agent": chosen_agent},
        )

    def _summarize_turn(self, user_message: str, agent_answer: str) -> str:
        um = user_message.strip().replace("\n", " ")
        aa = agent_answer.strip().replace("\n", " ")
        return f"User: {um[:200]} | Agent: {aa[:240]}"

    def _extract_facts(self, user_message: str, agent_answer: str) -> list[str]:
        text = f"{user_message}\n{agent_answer}"
        memories = []

        for m in re.findall(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s*[-–]\s*\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", text):
            memories.append(f"Tarih aralığı: {m}")

        for m in re.findall(r"\b(\d{1,3}(?:[.,]\d{3})+|\d+)\s*(tl|try|₺|usd|eur)?\b", text.lower()):
            num, cur = m
            if len(num) >= 4:
                memories.append(f"Tutar: {num} {cur}".strip())

        kw = ["kalan izin", "limit önerisi", "teminat", "kampanya", "pos", "ticari kart"]
        for k in kw:
            if k in text.lower():
                memories.append(f"Konu: {k}")

        out, seen = [], set()
        for x in memories:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out[:8]

    # ─── MAIN HANDLE (sync) ──────────────────────────────────

    def handle(self, session_id: str, user_message: str) -> SupervisorResponse:
        """Synchronous handler that returns the full response."""
        steps: list[dict[str, Any]] = []

        # Step 1: Plan
        t0 = time.time()
        chosen, plan_info = self.plan(user_message)
        steps.append({
            "step": "plan",
            "status": "done",
            "detail": f"Ajan seçildi: {chosen}",
            "data": plan_info,
            "duration_ms": int((time.time() - t0) * 1000),
        })

        # Step 2: Recall
        t0 = time.time()
        memories = self.recall_memories(user_message, session_id, chosen)
        steps.append({
            "step": "recall",
            "status": "done",
            "detail": f"Hafıza taraması tamamlandı (LTM: {len(memories['ltm_details'])} kayıt)",
            "data": {"ltm_count": len(memories["ltm_details"]), "ltm_details": memories["ltm_details"]},
            "duration_ms": int((time.time() - t0) * 1000),
        })

        # Step 3: Execute
        t0 = time.time()
        context = "\n\n".join(filter(None, [
            memories["shared_ctx"],
            memories["supervisor_ctx"],
            memories["ltm_ctx"],
        ]))
        exec_result = self.execute_agent(chosen, user_message, session_id, context)
        steps.append({
            "step": "execute",
            "status": "done",
            "detail": f"{chosen} ajanı yanıt üretti",
            "data": {"agent": chosen},
            "duration_ms": int((time.time() - t0) * 1000),
        })

        # Step 4: Memorize
        t0 = time.time()
        sup_ltm = self.memorize(session_id, chosen, user_message, exec_result["answer"])
        steps.append({
            "step": "memorize",
            "status": "done",
            "detail": f"Hafıza güncellendi ({len(sup_ltm)} LTM kaydı)",
            "data": {"supervisor_ltm": sup_ltm, "agent_ltm": exec_result.get("ltm_entries", [])},
            "duration_ms": int((time.time() - t0) * 1000),
        })

        return SupervisorResponse(
            chosen_agent=chosen,
            answer=exec_result["answer"],
            debug={
                "shared_ctx_present": bool(memories["shared_ctx"]),
                "supervisor_ctx_present": bool(memories["supervisor_ctx"]),
                "ltm_ctx_present": bool(memories["ltm_ctx"]),
                "plan_method": plan_info.get("method", "unknown"),
            },
            plan_steps=steps,
            ltm_entries=exec_result.get("ltm_entries", []) + sup_ltm,
        )

    # ─── STREAMING HANDLE (async generator for SSE) ──────────

    async def handle_stream(self, session_id: str, user_message: str) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator that yields step-by-step events for SSE streaming."""

        # Step 1: Plan
        yield {"event": "step_start", "step": "plan", "detail": "Kullanıcı mesajı analiz ediliyor..."}
        t0 = time.time()
        chosen, plan_info = self.plan(user_message)
        yield {
            "event": "step_done",
            "step": "plan",
            "detail": f"Ajan seçildi: {chosen}",
            "data": {**plan_info, "chosen_agent": chosen},
            "duration_ms": int((time.time() - t0) * 1000),
        }

        # Step 2: Recall
        yield {"event": "step_start", "step": "recall", "detail": "Uzun süreli hafıza taranıyor..."}
        t0 = time.time()
        memories = self.recall_memories(user_message, session_id, chosen)
        yield {
            "event": "step_done",
            "step": "recall",
            "detail": f"Hafıza taraması tamamlandı ({len(memories['ltm_details'])} LTM kaydı bulundu)",
            "data": {
                "ltm_count": len(memories["ltm_details"]),
                "ltm_details": memories["ltm_details"],
                "has_shared": bool(memories["shared_ctx"]),
                "has_supervisor": bool(memories["supervisor_ctx"]),
            },
            "duration_ms": int((time.time() - t0) * 1000),
        }

        # Step 3: Route
        yield {
            "event": "step_start",
            "step": "route",
            "detail": f"Mesaj {chosen} ajanına yönlendiriliyor...",
        }

        # Step 4: Execute
        yield {"event": "step_start", "step": "execute", "detail": f"{chosen} ajanı çalıştırılıyor..."}
        t0 = time.time()
        context = "\n\n".join(filter(None, [
            memories["shared_ctx"],
            memories["supervisor_ctx"],
            memories["ltm_ctx"],
        ]))
        exec_result = self.execute_agent(chosen, user_message, session_id, context)
        yield {
            "event": "step_done",
            "step": "execute",
            "detail": f"{chosen} ajanı yanıt üretti",
            "data": {"agent": chosen, "answer_preview": exec_result["answer"][:100]},
            "duration_ms": int((time.time() - t0) * 1000),
        }

        # Step 5: Memorize
        yield {"event": "step_start", "step": "memorize", "detail": "Hafıza güncelleniyor..."}
        t0 = time.time()
        sup_ltm = self.memorize(session_id, chosen, user_message, exec_result["answer"])
        yield {
            "event": "step_done",
            "step": "memorize",
            "detail": f"Hafıza güncellendi ({len(sup_ltm)} yeni LTM kaydı)",
            "data": {
                "supervisor_ltm": sup_ltm,
                "agent_ltm": exec_result.get("ltm_entries", []),
            },
            "duration_ms": int((time.time() - t0) * 1000),
        }

        # Final response
        yield {
            "event": "complete",
            "data": {
                "chosen_agent": chosen,
                "answer": exec_result["answer"],
                "ltm_entries": exec_result.get("ltm_entries", []) + sup_ltm,
                "debug": {
                    "shared_ctx_present": bool(memories["shared_ctx"]),
                    "supervisor_ctx_present": bool(memories["supervisor_ctx"]),
                    "ltm_ctx_present": bool(memories["ltm_ctx"]),
                    "plan_method": plan_info.get("method", "unknown"),
                },
            },
        }

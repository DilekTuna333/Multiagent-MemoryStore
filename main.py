from __future__ import annotations

import json
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .memory_store import MemoryStore
from .long_term_memory import LongTermMemory, MemoryCategory
from .supervisor import SupervisorDeepAgent

app = FastAPI(title="Supervisor Deep Agent + Long-Term Memory Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory = MemoryStore()
ltm = LongTermMemory()
supervisor = SupervisorDeepAgent(memory, ltm)


class ChatIn(BaseModel):
    session_id: str
    message: str


@app.get("/health")
def health():
    return {"ok": True}


# ─── Sync chat endpoint (backward compatible) ───────────────

@app.post("/chat")
def chat(payload: ChatIn):
    resp = supervisor.handle(session_id=payload.session_id, user_message=payload.message)
    return {
        "session_id": payload.session_id,
        "chosen_agent": resp.chosen_agent,
        "answer": resp.answer,
        "debug": resp.debug,
        "plan_steps": resp.plan_steps,
        "ltm_entries": resp.ltm_entries,
    }


# ─── SSE streaming chat endpoint ────────────────────────────

@app.post("/chat/stream")
async def chat_stream(payload: ChatIn):
    async def event_generator():
        async for event in supervisor.handle_stream(
            session_id=payload.session_id,
            user_message=payload.message,
        ):
            yield {
                "event": event.get("event", "message"),
                "data": json.dumps(event, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


# ─── Short-term memory endpoint ──────────────────────────────

@app.get("/memory/{session_id}")
def get_memories(session_id: str):
    return {
        "shared": memory.list_recent("shared_memory", limit=30),
        "supervisor": memory.list_recent("supervisor_memory", limit=30),
        "ik": memory.list_recent("ik_memory", limit=30),
        "ticari_kredi": memory.list_recent("kredi_memory", limit=30),
        "kampanyalar": memory.list_recent("kampanya_memory", limit=30),
        "genel": memory.list_recent("genel_memory", limit=30),
    }


# ─── Long-term memory endpoints ─────────────────────────────

@app.get("/ltm/{agent}")
def get_ltm(agent: str):
    """Get all long-term memories for an agent grouped by category."""
    return ltm.list_all_for_agent(agent, limit_per_category=20)


@app.get("/ltm/{agent}/search")
def search_ltm(
    agent: str,
    q: str,
    category: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 10,
):
    """Search long-term memories by query with optional category filter."""
    cat = MemoryCategory(category) if category else None
    entries = ltm.retrieve(
        agent=agent,
        query=q,
        category=cat,
        session_id=session_id,
        limit=limit,
    )
    return [
        {
            "id": e.id,
            "text": e.text,
            "category": e.category.value,
            "score": e.score,
            "agent": e.agent,
            "session_id": e.session_id,
            "timestamp": e.timestamp,
            "metadata": e.metadata,
        }
        for e in entries
    ]


@app.get("/ltm/all/summary")
def get_all_ltm_summary():
    """Get summary of all LTM across all agents."""
    agents = ["supervisor", "ik", "ticari_kredi", "kampanyalar", "genel"]
    result = {}
    for agent in agents:
        result[agent] = ltm.list_all_for_agent(agent, limit_per_category=10)
    return result

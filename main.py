from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .config import settings
from .memory_store import MemoryStore
from .long_term_memory import LongTermMemory, MemoryCategory
from .shared_board import SharedBoard, TaskStatus
from .supervisor import SupervisorDeepAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Supervisor Deep Agent + Long-Term Memory Demo")

# Startup log — show which provider is active
if settings.use_azure:
    logger.info("=== Azure OpenAI aktif ===")
    logger.info("  Endpoint: %s", settings.azure_openai_endpoint)
    logger.info("  LLM Deployment: %s", settings.azure_openai_deployment_name)
    logger.info("  Embedding Deployment: %s", settings.azure_openai_embedding_deployment)
    logger.info("  API Version: %s", settings.azure_openai_api_version)
elif settings.openai_api_key:
    logger.info("=== Standard OpenAI aktif ===")
    logger.info("  Model: %s", settings.openai_model)
else:
    logger.warning("=== UYARI: Ne Azure ne OpenAI API key tanımlı! LLM çağrıları çalışmayacak. ===")

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
board = supervisor.board


class ChatIn(BaseModel):
    session_id: str
    message: str
    user_id: str = ""


@app.get("/health")
def health():
    return {"ok": True}


# ─── Sync chat endpoint (backward compatible) ───────────────

@app.post("/chat")
def chat(payload: ChatIn):
    resp = supervisor.handle(
        session_id=payload.session_id,
        user_message=payload.message,
        user_id=payload.user_id,
    )
    return {
        "session_id": payload.session_id,
        "user_id": payload.user_id,
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
            user_id=payload.user_id,
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
def get_ltm(agent: str, user_id: Optional[str] = None):
    return ltm.list_all_for_agent(agent, limit_per_category=20, user_id=user_id)


@app.get("/ltm/{agent}/search")
def search_ltm(
    agent: str,
    q: str,
    category: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 10,
):
    cat = MemoryCategory(category) if category else None
    entries = ltm.retrieve(
        agent=agent,
        query=q,
        category=cat,
        session_id=session_id,
        user_id=user_id,
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
    agents = ["supervisor", "ik", "ticari_kredi", "kampanyalar", "genel", "shared"]
    result = {}
    for agent in agents:
        result[agent] = ltm.list_all_for_agent(agent, limit_per_category=10)
    return result


# ─── Shared Board endpoints ──────────────────────────────────

class TaskIn(BaseModel):
    session_id: str
    description: str
    assigned_to: str
    created_by: str = "supervisor"
    priority: int = 1


class TaskUpdateIn(BaseModel):
    task_id: str
    status: str  # pending, in_progress, done, failed
    result: str = ""
    session_id: str = ""


class AnnounceIn(BaseModel):
    session_id: str
    message: str
    created_by: str = "supervisor"
    target_agents: list[str] = []


class SharedFactIn(BaseModel):
    session_id: str
    text: str
    written_by: str
    category: Optional[str] = None


@app.post("/board/task")
def create_board_task(payload: TaskIn):
    task = board.create_task(
        session_id=payload.session_id,
        description=payload.description,
        assigned_to=payload.assigned_to,
        created_by=payload.created_by,
        priority=payload.priority,
    )
    return {"task_id": task.id, "status": task.status.value}


@app.post("/board/task/update")
def update_board_task(payload: TaskUpdateIn):
    board.update_task_status(
        task_id=payload.task_id,
        status=TaskStatus(payload.status),
        result=payload.result,
        session_id=payload.session_id,
    )
    return {"ok": True}


@app.get("/board/tasks/{session_id}")
def get_board_tasks(session_id: str):
    return board.get_all_tasks(session_id)


@app.get("/board/tasks/{session_id}/{agent_name}")
def get_agent_tasks(session_id: str, agent_name: str):
    return board.get_tasks_for_agent(agent_name, session_id)


@app.post("/board/announce")
def create_announcement(payload: AnnounceIn):
    ann = board.announce(
        session_id=payload.session_id,
        message=payload.message,
        created_by=payload.created_by,
        target_agents=payload.target_agents or None,
    )
    return {"ann_id": ann.id}


@app.get("/board/announcements/{session_id}")
def get_announcements(session_id: str, agent: Optional[str] = None):
    return board.get_announcements(session_id, agent)


@app.post("/board/fact")
def write_shared_fact(payload: SharedFactIn):
    cat = MemoryCategory(payload.category) if payload.category else None
    result = board.write_shared_fact(
        session_id=payload.session_id,
        text=payload.text,
        written_by=payload.written_by,
        category=cat,
    )
    return result


@app.get("/board/facts")
def get_shared_facts(q: Optional[str] = None, category: Optional[str] = None, top_k: int = 5):
    if q:
        cat = MemoryCategory(category) if category else None
        return board.search_shared_facts(q, category=cat, top_k=top_k)
    return board.get_all_shared_facts()

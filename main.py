from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .memory_store import MemoryStore
from .supervisor import SupervisorDeepAgent

app = FastAPI(title="Supervisor Deep Agent + Qdrant Memory Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory = MemoryStore()
supervisor = SupervisorDeepAgent(memory)

class ChatIn(BaseModel):
    session_id: str
    message: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(payload: ChatIn):
    resp = supervisor.handle(session_id=payload.session_id, user_message=payload.message)
    return {
        "session_id": payload.session_id,
        "chosen_agent": resp.chosen_agent,
        "answer": resp.answer,
        "debug": resp.debug,
    }

@app.get("/memory/{session_id}")
def get_memories(session_id: str):
    return {
        "shared": memory.list_recent("shared_memory", limit=30),
        "supervisor": memory.list_recent("supervisor_memory", limit=30),
        "ik": memory.list_recent("ik_memory", limit=30),
        "ticari_kredi": memory.list_recent("kredi_memory", limit=30),
        "kampanyalar": memory.list_recent("kampanya_memory", limit=30),
    }

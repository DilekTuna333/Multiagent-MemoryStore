"""
Shared Board Memory - Central communication hub for multi-agent collaboration.

Three layers:
  1. TASK BOARD (short-term, session-scoped)
     - Supervisor creates tasks, assigns to agents
     - Agents pick up tasks, update status, write results
     - Like a Kanban board: pending -> in_progress -> done

  2. SHARED FACTS (long-term, cross-session)
     - Any agent can write facts visible to all agents
     - "Müşteri X cirosu 10M TL" type knowledge
     - Stored in LTM with category tagging

  3. ANNOUNCEMENTS (short-term, session-scoped)
     - Supervisor broadcasts context to all agents
     - "Bu müşteri VIP", "Acil talep" etc.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .memory_store import MemoryStore
from .long_term_memory import LongTermMemory, MemoryCategory


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


@dataclass
class BoardTask:
    id: str
    description: str
    assigned_to: str  # agent name
    status: TaskStatus
    created_by: str  # "supervisor" or agent name
    session_id: str
    result: str = ""
    priority: int = 1  # 1=normal, 2=high, 3=urgent
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: int = 0
    updated_at: int = 0


@dataclass
class Announcement:
    id: str
    message: str
    created_by: str
    session_id: str
    target_agents: list[str] = field(default_factory=list)  # empty = all
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: int = 0


class SharedBoard:
    """Central shared board for inter-agent communication and task management."""

    TASK_COLLECTION = "board_tasks"
    ANNOUNCEMENT_COLLECTION = "board_announcements"
    SHARED_FACTS_AGENT = "shared"  # LTM agent name for shared facts

    def __init__(self, memory: MemoryStore, ltm: LongTermMemory):
        self.memory = memory
        self.ltm = ltm

    # ─── TASK BOARD ──────────────────────────────────────────

    def create_task(
        self,
        session_id: str,
        description: str,
        assigned_to: str,
        created_by: str = "supervisor",
        priority: int = 1,
        metadata: Optional[dict[str, Any]] = None,
    ) -> BoardTask:
        """Supervisor creates a task and assigns it to an agent."""
        task_id = str(uuid.uuid4())
        now = int(time.time())

        task = BoardTask(
            id=task_id,
            description=description,
            assigned_to=assigned_to,
            status=TaskStatus.PENDING,
            created_by=created_by,
            session_id=session_id,
            priority=priority,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )

        self.memory.upsert_text(
            self.TASK_COLLECTION,
            text=f"TASK [{task.status.value}] -> {assigned_to}: {description}",
            meta={
                "session_id": session_id,
                "task_id": task_id,
                "assigned_to": assigned_to,
                "status": task.status.value,
                "created_by": created_by,
                "priority": priority,
                "result": "",
                **(metadata or {}),
            },
            point_id=task_id,
        )

        return task

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: str = "",
        session_id: str = "",
    ):
        """Agent updates a task's status and optionally writes a result."""
        now = int(time.time())
        # Re-upsert with updated status
        self.memory.upsert_text(
            self.TASK_COLLECTION,
            text=f"TASK [{status.value}] result={result[:200]}",
            meta={
                "session_id": session_id,
                "task_id": task_id,
                "status": status.value,
                "result": result[:500],
                "updated_at": now,
            },
            point_id=task_id,
        )

    def get_tasks_for_agent(
        self,
        agent_name: str,
        session_id: str,
        status_filter: Optional[TaskStatus] = None,
    ) -> list[dict[str, Any]]:
        """Get all tasks assigned to an agent in a session."""
        points, _ = self.memory.client.scroll(
            collection_name=self.TASK_COLLECTION,
            scroll_filter=_build_filter({
                "meta.session_id": session_id,
                "meta.assigned_to": agent_name,
                **({"meta.status": status_filter.value} if status_filter else {}),
            }),
            limit=50,
            with_payload=True,
        )
        return _points_to_list(points)

    def get_all_tasks(self, session_id: str) -> list[dict[str, Any]]:
        """Get all tasks in a session (for supervisor/UI)."""
        self.memory.ensure_collection(self.TASK_COLLECTION)
        try:
            points, _ = self.memory.client.scroll(
                collection_name=self.TASK_COLLECTION,
                scroll_filter=_build_filter({"meta.session_id": session_id}),
                limit=100,
                with_payload=True,
            )
            return _points_to_list(points)
        except Exception:
            return []

    # ─── ANNOUNCEMENTS ───────────────────────────────────────

    def announce(
        self,
        session_id: str,
        message: str,
        created_by: str = "supervisor",
        target_agents: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Announcement:
        """Broadcast an announcement to all or specific agents."""
        ann_id = str(uuid.uuid4())
        now = int(time.time())

        ann = Announcement(
            id=ann_id,
            message=message,
            created_by=created_by,
            session_id=session_id,
            target_agents=target_agents or [],
            metadata=metadata or {},
            created_at=now,
        )

        self.memory.upsert_text(
            self.ANNOUNCEMENT_COLLECTION,
            text=f"ANNOUNCEMENT by {created_by}: {message}",
            meta={
                "session_id": session_id,
                "ann_id": ann_id,
                "created_by": created_by,
                "target_agents": target_agents or [],
                "type": "announcement",
                **(metadata or {}),
            },
            point_id=ann_id,
        )

        return ann

    def get_announcements(
        self,
        session_id: str,
        agent_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get announcements for a session, optionally filtered by target agent."""
        self.memory.ensure_collection(self.ANNOUNCEMENT_COLLECTION)
        try:
            points, _ = self.memory.client.scroll(
                collection_name=self.ANNOUNCEMENT_COLLECTION,
                scroll_filter=_build_filter({"meta.session_id": session_id}),
                limit=50,
                with_payload=True,
            )
            items = _points_to_list(points)
            if agent_name:
                items = [
                    it for it in items
                    if not it.get("meta", {}).get("target_agents")
                    or agent_name in it.get("meta", {}).get("target_agents", [])
                ]
            return items
        except Exception:
            return []

    # ─── SHARED FACTS (Long-term, cross-session) ─────────────

    def write_shared_fact(
        self,
        session_id: str,
        text: str,
        written_by: str,
        category: Optional[MemoryCategory] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Any agent can write a fact to the shared long-term knowledge base."""
        entry = self.ltm.insert(
            agent=self.SHARED_FACTS_AGENT,
            session_id=session_id,
            text=text,
            category=category,
            metadata={"written_by": written_by, **(metadata or {})},
        )
        return {"id": entry.id, "category": entry.category.value, "text": entry.text[:100]}

    def search_shared_facts(
        self,
        query: str,
        category: Optional[MemoryCategory] = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search shared facts across all sessions (long-term knowledge)."""
        entries = self.ltm.retrieve(
            agent=self.SHARED_FACTS_AGENT,
            query=query,
            category=category,
            session_id=None,  # cross-session
            limit=top_k,
        )
        return [
            {
                "id": e.id,
                "text": e.text,
                "category": e.category.value,
                "score": e.score,
                "metadata": e.metadata,
            }
            for e in entries
        ]

    def get_all_shared_facts(self, limit_per_category: int = 20) -> dict[str, list[dict[str, Any]]]:
        """Get all shared facts grouped by category."""
        return self.ltm.list_all_for_agent(self.SHARED_FACTS_AGENT, limit_per_category)

    # ─── BOARD CONTEXT (for injection into agents) ───────────

    def get_board_context(
        self,
        session_id: str,
        agent_name: str,
        query: str,
        top_k: int = 5,
    ) -> str:
        """Build a context string from the board for an agent."""
        parts = []

        # Pending tasks for this agent
        tasks = self.get_tasks_for_agent(agent_name, session_id, TaskStatus.PENDING)
        if tasks:
            lines = [f"  - [{t.get('meta',{}).get('priority',1)}] {t.get('text','')[:150]}" for t in tasks[:5]]
            parts.append("Bekleyen görevlerin:\n" + "\n".join(lines))

        # Announcements
        anns = self.get_announcements(session_id, agent_name)
        if anns:
            lines = [f"  - {a.get('text','')[:150]}" for a in anns[:3]]
            parts.append("Duyurular:\n" + "\n".join(lines))

        # Shared facts (semantic search)
        facts = self.search_shared_facts(query, top_k=top_k)
        if facts:
            lines = [f"  - [{f['category']}] {f['text'][:150]}" for f in facts]
            parts.append("Ortak bilgi tabanından:\n" + "\n".join(lines))

        if not parts:
            return ""
        return "=== Shared Board ===\n" + "\n\n".join(parts)


# ─── Helpers ─────────────────────────────────────────────────

def _build_filter(conditions: dict[str, Any]):
    """Build a Qdrant filter from key-value conditions."""
    from qdrant_client.http import models as qm

    must = []
    for key, value in conditions.items():
        if value is not None and value != "":
            must.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=value)))
    if not must:
        return None
    return qm.Filter(must=must)


def _points_to_list(points) -> list[dict[str, Any]]:
    """Convert Qdrant scroll results to dicts."""
    items = []
    for p in points:
        payload = p.payload or {}
        items.append({
            "id": str(p.id),
            "text": payload.get("text", ""),
            "meta": payload.get("meta", {}),
            "ts": payload.get("ts", 0),
        })
    items.sort(key=lambda x: x.get("ts", 0), reverse=True)
    return items

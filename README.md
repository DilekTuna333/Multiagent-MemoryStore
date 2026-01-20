# Supervisor Deep Agent + Qdrant (Mem0-ready) Demo

This repo is a working demo of:
- Supervisor (planner) + sub-agents (IK, Ticari Kredi, Kampanyalar)
- Memory governance:
  - Supervisor uses ONLY: shared_memory + supervisor_memory
  - Sub-agents use: shared_memory (read) + their own private memory (read/write)
- Qdrant as the vector DB (collections per memory scope)
- Audit routing decisions in supervisor_memory
- Structured facts written by sub-agents into their private memories

## Run

1) Start Qdrant
```bash
docker compose up -d
```

2) Create venv + install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

3) Start API
```bash
uvicorn app.main:app --reload --port 8000
```

4) Open UI
Open `web/index.html` in your browser.

API:
- POST /chat { "session_id": "...", "message": "..." }
- GET  /memory/{session_id}

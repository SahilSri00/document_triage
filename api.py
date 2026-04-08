"""
FastAPI wrapper for the Document Triage Environment.

Exposes the Gymnasium environment as a REST API so remote agents
can interact with it over HTTP.

Run:  uvicorn api:app --host 0.0.0.0 --port 7860 --reload
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.environment import DocumentTriageEnv

# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Document Triage Environment API",
    description="REST API for the Gymnasium-based Document Triage & Routing RL environment.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session store (in-memory) ─────────────────────────────────────────

sessions: Dict[str, DocumentTriageEnv] = {}

# ── Schemas ───────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    difficulty: Optional[str] = None
    max_steps: int = 15

class StepRequest(BaseModel):
    session_id: str
    action_type: int = Field(..., ge=0, le=7)
    parameter: str = ""

class ResetResponse(BaseModel):
    session_id: str
    observation: Dict[str, Any]
    info: Dict[str, Any]

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

# ── Endpoints ─────────────────────────────────────────────────────────

@app.post("/reset", response_model=ResetResponse)
def reset_env(req: ResetRequest):
    """Create a new session and reset the environment."""
    env = DocumentTriageEnv(tasks_path="tasks/tasks.json", max_steps=req.max_steps)
    options = {}
    if req.task_id:
        options["task_id"] = req.task_id
    elif req.difficulty:
        options["difficulty"] = req.difficulty

    try:
        obs, info = env.reset(options=options if options else None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    sid = str(uuid.uuid4())
    sessions[sid] = env
    return ResetResponse(session_id=sid, observation=obs, info=info)


@app.post("/step", response_model=StepResponse)
def step_env(req: StepRequest):
    """Take one step in the environment."""
    env = sessions.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")

    action = {"action_type": req.action_type, "parameter": req.parameter}
    obs, reward, terminated, truncated, info = env.step(action)

    # Clean up finished sessions
    if terminated or truncated:
        sessions.pop(req.session_id, None)

    return StepResponse(
        observation=obs, reward=reward,
        terminated=terminated, truncated=truncated, info=info,
    )


@app.get("/tasks")
def list_tasks():
    """List all available task IDs grouped by difficulty."""
    env = DocumentTriageEnv(tasks_path="tasks/tasks.json")
    return {
        "easy": env.get_tasks_by_difficulty("easy"),
        "medium": env.get_tasks_by_difficulty("medium"),
        "hard": env.get_tasks_by_difficulty("hard"),
        "total": len(env.tasks),
    }


@app.get("/actions")
def list_actions():
    """Return available actions and their descriptions."""
    env = DocumentTriageEnv(tasks_path="tasks/tasks.json")
    return {
        "actions": env.get_action_descriptions(),
        "departments": env.get_departments(),
    }


@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(sessions)}

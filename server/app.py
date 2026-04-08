"""
FastAPI application for Document Triage Environment.

Creates an HTTP server compatible with OpenEnv clients and the
pre-validation script (POST /reset returns 200).
"""

import json
import os
import sys
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment import DocumentTriageEnv

# ── App Setup ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Document Triage Environment",
    description="OpenEnv-compatible RL environment for document triage and routing",
    version="1.0.0",
)

# ── Session Storage ───────────────────────────────────────────────────

_sessions: Dict[str, DocumentTriageEnv] = {}
_default_env = DocumentTriageEnv(
    tasks_path=os.path.join(os.path.dirname(__file__), "..", "tasks", "tasks.json"),
    max_steps=15,
)

# ── Request / Response Models ─────────────────────────────────────────


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    difficulty: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action_type: int
    parameter: str


class ResetResponse(BaseModel):
    session_id: str
    observation: Dict[str, Any]
    info: Dict[str, Any]


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    done: bool
    info: Dict[str, Any]


# ── Endpoints ─────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok", "env": "document_triage", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest = None):
    """Reset the environment. OpenEnv pre-validation POSTs to /reset."""
    if req is None:
        req = ResetRequest()

    session_id = str(uuid4())
    env = DocumentTriageEnv(
        tasks_path=os.path.join(os.path.dirname(__file__), "..", "tasks", "tasks.json"),
        max_steps=15,
    )

    options = {}
    if req.task_id:
        options["task_id"] = req.task_id
    if req.difficulty:
        options["difficulty"] = req.difficulty

    obs, info = env.reset(seed=req.seed, options=options if options else None)
    _sessions[session_id] = env

    return ResetResponse(session_id=session_id, observation=obs, info=info)


@app.post("/step/{session_id}")
def step(session_id: str, req: StepRequest):
    """Execute one action in the environment."""
    if session_id not in _sessions:
        return {"error": f"Session '{session_id}' not found. Call /reset first."}

    env = _sessions[session_id]
    action = {"action_type": req.action_type, "parameter": req.parameter}
    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated
    if done:
        _sessions.pop(session_id, None)

    return StepResponse(
        observation=obs,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        done=done,
        info=info,
    )


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": _default_env.get_task_ids(),
        "by_difficulty": {
            d: _default_env.get_tasks_by_difficulty(d)
            for d in ["easy", "medium", "hard"]
        },
    }


@app.get("/actions")
def list_actions():
    return _default_env.get_action_descriptions()


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

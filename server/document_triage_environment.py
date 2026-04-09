"""
Document Triage — OpenEnv Server Environment.

Wraps the Gymnasium-based DocumentTriageEnv into an OpenEnv-compatible
Environment class with reset(), step(), state(), and get_metadata().
"""

import json
import os
import sys
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

# Ensure project root is importable inside Docker
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.models import (
    DocumentTriageAction,
    DocumentTriageObservation,
    DocumentTriageState,
)
from src.environment import DocumentTriageEnv as GymEnv


_TASKS_PATH = os.path.join(os.path.dirname(__file__), "..", "tasks", "tasks.json")


def _safe_parse(value: Any, fallback: Any = None) -> Any:
    """Parse a JSON string into a Python object, or return fallback."""
    if fallback is None:
        fallback = {} if isinstance(value, str) and value.strip().startswith("{") else []
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return fallback
    return fallback


class DocumentTriageEnvironment(Environment):
    """OpenEnv server-side Environment for document triage.

    Each instance wraps a fresh Gymnasium DocumentTriageEnv and exposes
    the standard OpenEnv protocol (reset / step / state / get_metadata).
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self.gym_env = GymEnv(tasks_path=_TASKS_PATH, max_steps=15)
        # Auto-reset so stateless HTTP calls (/step without /reset) don't crash
        self.gym_env.reset()
        self._state = DocumentTriageState(
            episode_id=str(uuid4()), step_count=0
        )
        self._done = False
        self._total_reward = 0.0

    # ── metadata (used by /metadata endpoint) ──────────────────────────

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="Document Triage",
            description=(
                "An RL environment where agents learn to classify, extract, "
                "validate, flag, and route office documents. Features 15 tasks "
                "across 3 difficulty levels with a 6-component weighted grader."
            ),
            version="1.0.0",
            author="Sahil Srivastava",
        )

    # ── reset ──────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DocumentTriageObservation:
        """Reset the environment and load a new task."""
        # Allow callers to pick a specific task or difficulty
        options = {}
        task_id = kwargs.get("task_id")
        difficulty = kwargs.get("difficulty")
        if task_id:
            options["task_id"] = task_id
        if difficulty:
            options["difficulty"] = difficulty

        obs, info = self.gym_env.reset(
            seed=seed, options=options if options else None
        )

        eid = episode_id or str(uuid4())
        tid = info.get("task_id", "")
        diff = info.get("difficulty", "")

        self._state = DocumentTriageState(
            episode_id=eid,
            step_count=0,
            task_id=tid,
            difficulty=diff,
            is_done=False,
            total_reward=0.0,
        )
        self._done = False
        self._total_reward = 0.0

        return DocumentTriageObservation(
            done=False,
            reward=0.0,
            document_text=obs.get("document_text", ""),
            document_metadata=_safe_parse(obs.get("metadata", "{}"), {}),
            classified_as="",
            extracted_fields={},
            validated_fields={},
            flagged_missing=[],
            flagged_inconsistencies=[],
            steps_taken=0,
            steps_remaining=self.gym_env.max_steps,
            task_id=tid,
            difficulty=diff,
            action_feedback="Environment reset. Begin triaging the document.",
        )

    # ── step ───────────────────────────────────────────────────────────

    def step(
        self,
        action: DocumentTriageAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DocumentTriageObservation:
        """Execute one action in the environment."""
        self._state.step_count += 1

        # Convert OpenEnv action → Gymnasium action dict
        if hasattr(action, "model_dump"):
            action_dict = action.model_dump()
        elif isinstance(action, dict):
            action_dict = action
        else:
            action_dict = {
                "action_type": getattr(action, "action_type", 6),
                "parameter": getattr(action, "parameter", "operations"),
            }

        # Remove OpenEnv-level metadata key if present
        action_dict.pop("metadata", None)

        obs, reward, terminated, truncated, info = self.gym_env.step(action_dict)
        done = terminated or truncated
        self._done = done
        self._total_reward += reward
        self._state.is_done = done
        self._state.total_reward = self._total_reward

        # Parse observation strings back to structured data safely
        extracted = _safe_parse(obs.get("extracted_fields", "{}"), {})
        validated = _safe_parse(obs.get("validated_fields", "{}"), {})
        missing = _safe_parse(obs.get("flagged_missing", "[]"), [])
        inconsistencies = _safe_parse(obs.get("flagged_inconsistencies", "[]"), [])
        doc_metadata = _safe_parse(obs.get("metadata", "{}"), {})

        result = DocumentTriageObservation(
            done=done,
            reward=float(reward),
            document_text=obs.get("document_text", ""),
            document_metadata=doc_metadata,
            classified_as=obs.get("classified_as", ""),
            extracted_fields=extracted if isinstance(extracted, dict) else {},
            validated_fields=validated if isinstance(validated, dict) else {},
            flagged_missing=missing if isinstance(missing, list) else [],
            flagged_inconsistencies=(
                inconsistencies if isinstance(inconsistencies, list) else []
            ),
            steps_taken=obs.get("steps_taken", self._state.step_count),
            steps_remaining=obs.get("steps_remaining", 0),
            task_id=self._state.task_id,
            difficulty=self._state.difficulty,
            action_feedback=info.get("action_feedback", ""),
        )

        # Attach grading info when episode ends
        if done:
            result.final_score = info.get("final_score")
            result.grade = info.get("grade")
            result.score_breakdown = info.get("score_breakdown")

        return result

    # ── state ──────────────────────────────────────────────────────────

    @property
    def state(self) -> DocumentTriageState:
        return self._state

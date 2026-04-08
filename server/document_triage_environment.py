"""
Document Triage — OpenEnv Server Environment.

Wraps the Gymnasium-based DocumentTriageEnv into an OpenEnv-compatible
Environment class with reset(), step(), and state().
"""

import json
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.types import Action, Observation, State
from openenv.core.env_server.environment import Environment

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment import DocumentTriageEnv as GymEnv


class DocumentTriageEnvironment(Environment):
    """OpenEnv server-side Environment for document triage."""

    def __init__(self):
        self.gym_env = GymEnv(
            tasks_path=os.path.join(os.path.dirname(__file__), "..", "tasks", "tasks.json"),
            max_steps=15,
        )
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False
        self._last_obs = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset the environment and load a new task."""
        options = kwargs.get("options", None)
        obs, info = self.gym_env.reset(seed=seed, options=options)

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._done = False
        self._last_obs = obs

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "document_text": obs["document_text"],
                "metadata": obs["metadata"],
                "classified_as": "",
                "extracted_fields": "{}",
                "validated_fields": "{}",
                "flagged_missing": "[]",
                "flagged_inconsistencies": "[]",
                "steps_taken": 0,
                "steps_remaining": self.gym_env.max_steps,
                "task_id": info.get("task_id", ""),
                "difficulty": info.get("difficulty", ""),
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute one action in the environment."""
        self._state.step_count += 1

        # Parse action from OpenEnv action dict
        action_dict = {}
        if hasattr(action, "model_dump"):
            action_dict = action.model_dump()
        elif isinstance(action, dict):
            action_dict = action
        else:
            # Try to extract from generic Action
            action_dict = {
                "action_type": getattr(action, "action_type", 6),
                "parameter": getattr(action, "parameter", "operations"),
            }

        obs, reward, terminated, truncated, info = self.gym_env.step(action_dict)
        done = terminated or truncated
        self._done = done
        self._last_obs = obs

        meta = {
            "document_text": obs["document_text"],
            "metadata": obs["metadata"],
            "classified_as": obs["classified_as"],
            "extracted_fields": obs["extracted_fields"],
            "validated_fields": obs["validated_fields"],
            "flagged_missing": obs["flagged_missing"],
            "flagged_inconsistencies": obs["flagged_inconsistencies"],
            "steps_taken": obs["steps_taken"],
            "steps_remaining": obs["steps_remaining"],
            "action": info.get("action", ""),
            "parameter": info.get("parameter", ""),
            "step_reward": info.get("step_reward", 0),
            "total_reward": info.get("total_reward", 0),
        }

        if done:
            meta["final_score"] = info.get("final_score", 0)
            meta["grade"] = info.get("grade", "F")
            meta["score_breakdown"] = info.get("score_breakdown", {})

        return Observation(
            done=done,
            reward=reward,
            metadata=meta,
        )

    @property
    def state(self) -> State:
        return self._state

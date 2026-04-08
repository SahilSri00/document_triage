"""
Document Triage & Routing Environment
======================================

A Gymnasium-based RL environment where an AI agent processes documents by
classifying, extracting fields, flagging issues, and routing to the correct
department. Designed for evaluating LLM-based agents on realistic office work.

Design Principles:
    - PENALIZE, don't BLOCK: Every action is accepted, bad ones score poorly.
    - CASCADE EFFECT: Every action changes multiple state variables.
    - STATE ≠ OBSERVATION: The agent never sees the answer key.
    - TERMINAL actions end the episode immediately.

Actions:
    0: classify               - Classify the document type
    1: extract                 - Extract a named field value
    2: validate                - Validate a previously extracted field
    3: flag_missing            - Flag a field as missing from the document
    4: flag_inconsistency      - Flag an inconsistency in the document
    5: request_clarification   - Request clarification on ambiguous content
    6: route_to                - Route to a department  (TERMINAL)
    7: escalate                - Escalate the document   (TERMINAL)

Grader (6 weighted components):
    Classification    → 15%
    Field Extraction  → 30%
    Missing Detection → 15%
    Inconsistency     → 10%
    Routing           → 20%
    Efficiency        → 10%
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DocumentTriageEnv(gym.Env):
    """Gymnasium environment for document triage and routing.

    The agent receives a document and must:
      1. Classify its type.
      2. Extract key fields.
      3. Validate extracted information.
      4. Flag missing or inconsistent data.
      5. Route the document to the correct department (or escalate).

    The episode ends when the agent takes a TERMINAL action (route_to / escalate)
    or runs out of steps (truncation).
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    # ── Action Constants ──────────────────────────────────────────────
    CLASSIFY = 0
    EXTRACT = 1
    VALIDATE = 2
    FLAG_MISSING = 3
    FLAG_INCONSISTENCY = 4
    REQUEST_CLARIFICATION = 5
    ROUTE_TO = 6
    ESCALATE = 7

    ACTION_NAMES = [
        "classify",
        "extract",
        "validate",
        "flag_missing",
        "flag_inconsistency",
        "request_clarification",
        "route_to",
        "escalate",
    ]

    WORK_ACTIONS = {0, 1, 2, 3, 4, 5}  # Non-terminal
    TERMINAL_ACTIONS = {6, 7}           # End the episode

    # ── Valid Departments & Document Types ─────────────────────────────
    DEPARTMENTS = [
        "finance", "hr", "legal", "operations",
        "it", "compliance", "executive",
    ]

    DOCUMENT_TYPES = [
        "invoice", "contract", "memo", "report", "letter",
        "form", "receipt", "proposal", "notice", "complaint",
    ]

    # Common field name aliases → canonical name
    FIELD_ALIASES = {
        "total_amount": "amount", "total_due": "amount", "total": "amount",
        "invoice_amount": "amount", "payment_amount": "amount",
        "recipient": "vendor", "from": "vendor", "sender_company": "vendor",
        "company": "vendor", "company_name": "vendor", "supplier": "vendor",
        "invoice_date": "date", "document_date": "date", "report_date": "date",
        "payment_due_date": "due_date", "payment_deadline": "due_date",
        "name": "employee_name", "complainant": "employee_name",
        "emp_id": "employee_id", "id": "employee_id",
        "role": "position", "job_title": "position", "title": "position",
        "annual_salary": "salary", "compensation": "salary", "pay": "salary",
        "joining_date": "start_date", "commencement_date": "start_date",
        "effective_date": "start_date",
        "dept": "department", "team": "department",
        "order_number": "po_number", "purchase_order": "po_number",
        "total_cost": "total_amount", "estimated_cost": "total_cost",
        "auditor": "auditor", "prepared_by": "auditor",
        "expiry_date": "valid_until", "expiration": "valid_until",
        "cost": "monthly_cost", "retainer": "monthly_cost",
        "term": "contract_term", "contract_duration": "contract_term",
        "incident_number": "incident_id",
        "records": "records_affected", "affected_records": "records_affected",
        "vendor_name": "vendor_involved", "third_party": "vendor_involved",
        "period": "report_period", "quarter": "report_period",
        "revenue": "total_revenue", "expenses": "total_expenses",
        "profit": "net_income", "net_profit": "net_income",
        "margin": "net_margin", "profit_margin": "net_margin",
        "number_of_incidents": "incidents_count",
        "type": "complaint_type", "subject": "complaint_type",
        "filed_date": "date_filed", "filing_date": "date_filed",
        "detection_date": "date_detected",
        "probation": "probation_period",
    }

    # ── Reward Constants ──────────────────────────────────────────────
    REWARD_CORRECT_CLASSIFY = 0.5
    REWARD_WRONG_CLASSIFY = -0.3
    REWARD_RECLASSIFY = -0.2
    REWARD_CORRECT_EXTRACT = 0.3
    REWARD_WRONG_EXTRACT = -0.2
    REWARD_DUPLICATE_EXTRACT = -0.15
    REWARD_CORRECT_VALIDATE = 0.1
    REWARD_WRONG_VALIDATE = -0.1
    REWARD_VALIDATE_UNEXTRACTED = -0.2
    REWARD_CORRECT_FLAG_MISSING = 0.3
    REWARD_WRONG_FLAG_MISSING = -0.2
    REWARD_DUPLICATE_FLAG = -0.15
    REWARD_CORRECT_FLAG_INCONSISTENCY = 0.3
    REWARD_WRONG_FLAG_INCONSISTENCY = -0.2
    REWARD_USEFUL_CLARIFICATION = 0.1
    REWARD_USELESS_CLARIFICATION = -0.1
    REWARD_CORRECT_ROUTE = 1.0
    REWARD_WRONG_ROUTE = -1.0
    REWARD_CORRECT_ESCALATE = 0.8
    REWARD_WRONG_ESCALATE = -0.8
    REWARD_STEP_PENALTY = -0.05
    REWARD_TRUNCATION_PENALTY = -0.5
    REWARD_INVALID_ACTION = -0.3

    # ── Grader Weights ────────────────────────────────────────────────
    WEIGHT_CLASSIFICATION = 0.15
    WEIGHT_EXTRACTION = 0.30
    WEIGHT_MISSING = 0.15
    WEIGHT_INCONSISTENCY = 0.10
    WEIGHT_ROUTING = 0.20
    WEIGHT_EFFICIENCY = 0.10

    # ──────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────

    def __init__(
        self,
        tasks_path: str = "tasks/tasks.json",
        max_steps: int = 15,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        # ── Load task data ────────────────────────────────────────────
        tasks_file = Path(tasks_path)
        if not tasks_file.is_absolute():
            tasks_file = Path(os.path.dirname(os.path.abspath(__file__))).parent / tasks_file
        with open(tasks_file, "r", encoding="utf-8") as f:
            self.tasks: List[Dict[str, Any]] = json.load(f)

        # ── Define spaces ─────────────────────────────────────────────
        self.action_space = spaces.Dict(
            {
                "action_type": spaces.Discrete(8),
                "parameter": spaces.Text(max_length=256),
            }
        )

        self.observation_space = spaces.Dict(
            {
                "document_text": spaces.Text(max_length=5000),
                "metadata": spaces.Text(max_length=1000),
                "classified_as": spaces.Text(max_length=64),
                "extracted_fields": spaces.Text(max_length=2000),
                "validated_fields": spaces.Text(max_length=1000),
                "flagged_missing": spaces.Text(max_length=1000),
                "flagged_inconsistencies": spaces.Text(max_length=1000),
                "clarifications_requested": spaces.Text(max_length=1000),
                "steps_taken": spaces.Discrete(max_steps + 1),
                "steps_remaining": spaces.Discrete(max_steps + 1),
                "action_history": spaces.Text(max_length=3000),
            }
        )

        # ── Initialise mutable state ─────────────────────────────────
        self._init_state()

    # ──────────────────────────────────────────────────────────────────
    # State helpers
    # ──────────────────────────────────────────────────────────────────

    def _init_state(self) -> None:
        """Reset every mutable state variable to its default."""
        # Ground-truth (hidden from agent)
        self.current_task: Optional[Dict[str, Any]] = None
        self.answer_key: Optional[Dict[str, Any]] = None

        # Observable document data
        self.document_text: str = ""
        self.document_metadata: Dict[str, str] = {}

        # Agent workspace
        self.classified_as: str = ""
        self.extracted_fields: Dict[str, str] = {}
        self.validated_fields: Dict[str, bool] = {}
        self.flagged_missing: List[str] = []
        self.flagged_inconsistencies: List[str] = []
        self.clarifications_requested: List[str] = []
        self.routed_to: str = ""
        self.escalated: bool = False

        # Progress tracking
        self.steps_taken: int = 0
        self.steps_remaining: int = self.max_steps
        self.action_history: List[Dict[str, Any]] = []
        self.total_reward: float = 0.0

    # ──────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment and load a new (or specified) task."""
        super().reset(seed=seed)
        self._init_state()

        # Select task
        if options and "task_id" in options:
            matches = [t for t in self.tasks if t["task_id"] == options["task_id"]]
            if not matches:
                raise ValueError(f"task_id '{options['task_id']}' not found")
            self.current_task = matches[0]
        elif options and "difficulty" in options:
            pool = [t for t in self.tasks if t["difficulty"] == options["difficulty"]]
            if not pool:
                raise ValueError(f"No tasks with difficulty '{options['difficulty']}'")
            self.current_task = pool[self.np_random.integers(0, len(pool))]
        else:
            self.current_task = self.tasks[self.np_random.integers(0, len(self.tasks))]

        # Populate state from task
        self.answer_key = self.current_task["answer_key"]
        self.document_text = self.current_task["document_text"]
        self.document_metadata = self.current_task.get("metadata", {})
        self.steps_remaining = self.max_steps

        observation = self._build_observation()
        info = {
            "task_id": self.current_task["task_id"],
            "difficulty": self.current_task.get("difficulty", "unknown"),
            "max_steps": self.max_steps,
        }
        return observation, info

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one action and return (obs, reward, terminated, truncated, info)."""

        # ── Parse action ──────────────────────────────────────────────
        action_type, parameter = self._parse_action(action)

        # ── Guard: episode already over ───────────────────────────────
        if self.steps_remaining <= 0:
            obs = self._build_observation()
            return obs, 0.0, True, False, {"error": "Episode already ended"}

        # ── Cascade: update counters ──────────────────────────────────
        self.steps_taken += 1
        self.steps_remaining -= 1

        # Small per-step cost to encourage efficiency
        reward: float = self.REWARD_STEP_PENALTY

        # Record action
        action_name = self.ACTION_NAMES[action_type] if 0 <= action_type < 8 else "invalid"
        self.action_history.append(
            {"step": self.steps_taken, "action": action_name, "parameter": parameter}
        )

        # ── Dispatch to handler ───────────────────────────────────────
        terminated = False
        info: Dict[str, Any] = {}

        if action_type == self.CLASSIFY:
            reward += self._handle_classify(parameter)
        elif action_type == self.EXTRACT:
            reward += self._handle_extract(parameter)
        elif action_type == self.VALIDATE:
            reward += self._handle_validate(parameter)
        elif action_type == self.FLAG_MISSING:
            reward += self._handle_flag_missing(parameter)
        elif action_type == self.FLAG_INCONSISTENCY:
            reward += self._handle_flag_inconsistency(parameter)
        elif action_type == self.REQUEST_CLARIFICATION:
            reward += self._handle_request_clarification(parameter)
        elif action_type == self.ROUTE_TO:
            reward += self._handle_route_to(parameter)
            terminated = True
        elif action_type == self.ESCALATE:
            reward += self._handle_escalate(parameter)
            terminated = True
        else:
            reward += self._handle_invalid_action(action_type, parameter)

        self.total_reward += reward

        # ── Check truncation (ran out of steps without terminal) ─────
        truncated = False
        if self.steps_remaining <= 0 and not terminated:
            truncated = True
            trunc_penalty = self.REWARD_TRUNCATION_PENALTY
            reward += trunc_penalty
            self.total_reward += trunc_penalty

        # ── Final score if episode ended ──────────────────────────────
        if terminated or truncated:
            final_score = self._calculate_final_score()
            info["final_score"] = final_score
            info["score_breakdown"] = self._get_score_breakdown()
            info["grade"] = self._letter_grade(final_score)

        info["step_reward"] = reward
        info["total_reward"] = self.total_reward
        info["action"] = action_name
        info["parameter"] = parameter

        observation = self._build_observation()
        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """Render the current environment state."""
        if self.current_task is None:
            return None

        lines = [
            "=" * 60,
            "  DOCUMENT TRIAGE ENVIRONMENT",
            "=" * 60,
            f"  Task:       {self.current_task['task_id']}",
            f"  Difficulty: {self.current_task.get('difficulty', '?')}",
            f"  Steps:      {self.steps_taken}/{self.max_steps}  (remaining: {self.steps_remaining})",
            f"  Reward:     {self.total_reward:+.2f}",
            "-" * 60,
            "  DOCUMENT (first 500 chars):",
            f"  {self.document_text[:500]}",
            "-" * 60,
            "  AGENT WORKSPACE:",
            f"    Classified as:       {self.classified_as or '(not yet)'}",
            f"    Extracted fields:    {json.dumps(self.extracted_fields) if self.extracted_fields else '(none)'}",
            f"    Validated fields:    {json.dumps(self.validated_fields) if self.validated_fields else '(none)'}",
            f"    Flagged missing:     {self.flagged_missing or '(none)'}",
            f"    Flagged inconsist.:  {self.flagged_inconsistencies or '(none)'}",
            f"    Clarifications:      {self.clarifications_requested or '(none)'}",
            f"    Routed to:           {self.routed_to or '(not yet)'}",
            f"    Escalated:           {self.escalated}",
            "-" * 60,
            "  ACTION HISTORY:",
        ]
        for entry in self.action_history[-10:]:
            lines.append(f"    Step {entry['step']}: {entry['action']}({entry['parameter']})")
        lines.append("=" * 60)

        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output

    # ──────────────────────────────────────────────────────────────────
    # Observation builder
    # ──────────────────────────────────────────────────────────────────

    def _build_observation(self) -> Dict[str, Any]:
        """Build the observation dict visible to the agent.

        NOTE: The answer_key is NEVER included — state ≠ observation.
        """
        return {
            "document_text": self.document_text,
            "metadata": json.dumps(self.document_metadata),
            "classified_as": self.classified_as,
            "extracted_fields": json.dumps(self.extracted_fields),
            "validated_fields": json.dumps(self.validated_fields),
            "flagged_missing": json.dumps(self.flagged_missing),
            "flagged_inconsistencies": json.dumps(self.flagged_inconsistencies),
            "clarifications_requested": json.dumps(self.clarifications_requested),
            "steps_taken": self.steps_taken,
            "steps_remaining": self.steps_remaining,
            "action_history": json.dumps(self.action_history),
        }

    # ──────────────────────────────────────────────────────────────────
    # Action parser
    # ──────────────────────────────────────────────────────────────────

    def _parse_action(self, action: Any) -> Tuple[int, str]:
        """Normalise various action formats into (action_type, parameter).

        Accepted formats:
            - dict:  {"action_type": int, "parameter": str}
            - tuple: (int, str)
            - str:   "action_name parameter_text"
        """
        if isinstance(action, dict):
            action_type = int(action.get("action_type", -1))
            parameter = str(action.get("parameter", ""))
        elif isinstance(action, (list, tuple)) and len(action) == 2:
            action_type = int(action[0])
            parameter = str(action[1])
        elif isinstance(action, str):
            parts = action.strip().split(maxsplit=1)
            name = parts[0].lower()
            parameter = parts[1] if len(parts) > 1 else ""
            if name in self.ACTION_NAMES:
                action_type = self.ACTION_NAMES.index(name)
            else:
                action_type = -1
        else:
            action_type = -1
            parameter = ""

        return action_type, parameter

    # ──────────────────────────────────────────────────────────────────
    # Action handlers  (each returns an incremental reward)
    # ──────────────────────────────────────────────────────────────────

    def _handle_classify(self, parameter: str) -> float:
        """Handle 'classify' action.  Parameter = proposed document type."""
        proposed = parameter.strip().lower()
        correct_type = self.answer_key["correct_type"].strip().lower()

        # Already classified? Penalise re-classify but still allow it.
        if self.classified_as:
            self.classified_as = proposed          # overwrite
            return self.REWARD_RECLASSIFY

        self.classified_as = proposed
        if proposed == correct_type:
            return self.REWARD_CORRECT_CLASSIFY
        return self.REWARD_WRONG_CLASSIFY

    def _resolve_field_name(self, field_name: str) -> str:
        """Resolve a field name through aliases to its canonical form."""
        canonical = self.FIELD_ALIASES.get(field_name, field_name)
        return canonical

    def _handle_extract(self, parameter: str) -> float:
        """Handle 'extract' action.  Parameter = 'field_name:value'."""
        if ":" not in parameter:
            return self.REWARD_INVALID_ACTION

        field_name, _, value = parameter.partition(":")
        field_name = field_name.strip().lower()
        value = value.strip()

        if not field_name or not value:
            return self.REWARD_INVALID_ACTION

        # Resolve alias (e.g. total_amount → amount)
        canonical = self._resolve_field_name(field_name)

        # Store under both original and canonical name for scorer
        # Duplicate extraction?
        if canonical in self.extracted_fields or field_name in self.extracted_fields:
            self.extracted_fields[canonical] = value  # overwrite
            return self.REWARD_DUPLICATE_EXTRACT

        self.extracted_fields[canonical] = value

        correct_fields = {
            k.lower(): v for k, v in self.answer_key.get("extractable_fields", {}).items()
        }

        if canonical in correct_fields:
            if self._fuzzy_match(value, correct_fields[canonical]):
                return self.REWARD_CORRECT_EXTRACT
            return self.REWARD_WRONG_EXTRACT
        # Extracted a field that doesn't exist in answer key
        return self.REWARD_WRONG_EXTRACT

    def _handle_validate(self, parameter: str) -> float:
        """Handle 'validate' action.  Parameter = field_name to validate."""
        field_name = parameter.strip().lower()

        if field_name not in self.extracted_fields:
            return self.REWARD_VALIDATE_UNEXTRACTED

        extracted_value = self.extracted_fields[field_name]
        correct_fields = {
            k.lower(): v for k, v in self.answer_key.get("extractable_fields", {}).items()
        }

        if field_name in correct_fields:
            is_correct = self._fuzzy_match(extracted_value, correct_fields[field_name])
        else:
            is_correct = False

        self.validated_fields[field_name] = is_correct
        return self.REWARD_CORRECT_VALIDATE if is_correct else self.REWARD_WRONG_VALIDATE

    def _handle_flag_missing(self, parameter: str) -> float:
        """Handle 'flag_missing' action.  Parameter = field name that's missing."""
        field_name = parameter.strip().lower()

        if not field_name:
            return self.REWARD_INVALID_ACTION

        if field_name in self.flagged_missing:
            return self.REWARD_DUPLICATE_FLAG

        self.flagged_missing.append(field_name)

        actual_missing = [f.lower() for f in self.answer_key.get("missing_fields", [])]
        if field_name in actual_missing:
            return self.REWARD_CORRECT_FLAG_MISSING
        return self.REWARD_WRONG_FLAG_MISSING

    def _handle_flag_inconsistency(self, parameter: str) -> float:
        """Handle 'flag_inconsistency'.  Parameter = description of inconsistency."""
        description = parameter.strip().lower()

        if not description:
            return self.REWARD_INVALID_ACTION

        if description in [d.lower() for d in self.flagged_inconsistencies]:
            return self.REWARD_DUPLICATE_FLAG

        self.flagged_inconsistencies.append(parameter.strip())

        actual_inconsistencies = [
            inc.lower() for inc in self.answer_key.get("inconsistencies", [])
        ]

        # Check if the flagged description is a reasonable match
        for actual in actual_inconsistencies:
            if self._fuzzy_match(description, actual):
                return self.REWARD_CORRECT_FLAG_INCONSISTENCY

        return self.REWARD_WRONG_FLAG_INCONSISTENCY

    def _handle_request_clarification(self, parameter: str) -> float:
        """Handle 'request_clarification'.  Parameter = the question / topic."""
        question = parameter.strip()

        if not question:
            return self.REWARD_INVALID_ACTION

        self.clarifications_requested.append(question)

        # If the task has known ambiguities, a clarification is useful
        has_ambiguities = bool(
            self.answer_key.get("missing_fields")
            or self.answer_key.get("inconsistencies")
            or self.answer_key.get("ambiguities")
        )
        return self.REWARD_USEFUL_CLARIFICATION if has_ambiguities else self.REWARD_USELESS_CLARIFICATION

    def _handle_route_to(self, parameter: str) -> float:
        """Handle 'route_to' (TERMINAL).  Parameter = department name."""
        department = parameter.strip().lower()
        self.routed_to = department

        correct_dept = self.answer_key["correct_department"].strip().lower()
        should_escalate = self.answer_key.get("should_escalate", False)

        # Agent routed but should have escalated
        if should_escalate:
            return self.REWARD_WRONG_ROUTE

        if department == correct_dept:
            return self.REWARD_CORRECT_ROUTE
        return self.REWARD_WRONG_ROUTE

    def _handle_escalate(self, parameter: str) -> float:
        """Handle 'escalate' (TERMINAL).  Parameter = reason for escalation."""
        self.escalated = True
        self.routed_to = "escalated"

        should_escalate = self.answer_key.get("should_escalate", False)
        if should_escalate:
            return self.REWARD_CORRECT_ESCALATE
        return self.REWARD_WRONG_ESCALATE

    def _handle_invalid_action(self, action_type: int, parameter: str) -> float:
        """Handle an unrecognised action type."""
        return self.REWARD_INVALID_ACTION

    # ──────────────────────────────────────────────────────────────────
    # Grader / Scorer
    # ──────────────────────────────────────────────────────────────────

    def _calculate_final_score(self) -> float:
        """Calculate the final weighted score (0.0 – 1.0)."""
        breakdown = self._get_score_breakdown()
        total = (
            breakdown["classification"] * self.WEIGHT_CLASSIFICATION
            + breakdown["extraction"] * self.WEIGHT_EXTRACTION
            + breakdown["missing_detection"] * self.WEIGHT_MISSING
            + breakdown["inconsistency_detection"] * self.WEIGHT_INCONSISTENCY
            + breakdown["routing"] * self.WEIGHT_ROUTING
            + breakdown["efficiency"] * self.WEIGHT_EFFICIENCY
        )
        return round(min(max(total, 0.0), 1.0), 4)

    def _get_score_breakdown(self) -> Dict[str, float]:
        """Return a dict of per-component scores (each 0.0 – 1.0)."""
        return {
            "classification": self._score_classification(),
            "extraction": self._score_extraction(),
            "missing_detection": self._score_missing_detection(),
            "inconsistency_detection": self._score_inconsistency_detection(),
            "routing": self._score_routing(),
            "efficiency": self._score_efficiency(),
        }

    # ── Individual scoring functions ──────────────────────────────────

    def _score_classification(self) -> float:
        """1.0 if correctly classified, 0.0 otherwise."""
        if not self.classified_as:
            return 0.0
        correct = self.answer_key["correct_type"].strip().lower()
        return 1.0 if self.classified_as.strip().lower() == correct else 0.0

    def _score_extraction(self) -> float:
        """Fraction of correctly extracted fields."""
        correct_fields = self.answer_key.get("extractable_fields", {})
        if not correct_fields:
            return 1.0  # Nothing to extract → full marks

        correct_count = 0
        total = len(correct_fields)

        for field_name, correct_value in correct_fields.items():
            fn_lower = field_name.lower()
            # Try exact match first, then check all extracted fields via aliases
            extracted = self.extracted_fields.get(fn_lower)
            if not extracted:
                # Check if agent used a different name that aliases to this
                for ext_name, ext_val in self.extracted_fields.items():
                    if self._resolve_field_name(ext_name) == fn_lower:
                        extracted = ext_val
                        break
            if extracted and self._fuzzy_match(extracted, correct_value):
                correct_count += 1

        return correct_count / total

    def _score_missing_detection(self) -> float:
        """Score based on correctly flagged missing fields vs false positives."""
        actual_missing = [f.lower() for f in self.answer_key.get("missing_fields", [])]

        if not actual_missing and not self.flagged_missing:
            return 1.0  # Nothing missing & agent didn't flag anything → perfect
        if not actual_missing and self.flagged_missing:
            return 0.0  # False positives

        correct_flags = sum(1 for f in self.flagged_missing if f.lower() in actual_missing)
        false_positives = len(self.flagged_missing) - correct_flags

        # Recall with false-positive penalty
        recall = correct_flags / len(actual_missing)
        penalty = min(false_positives * 0.2, 0.5)  # Cap penalty at -0.5
        return max(recall - penalty, 0.0)

    def _score_inconsistency_detection(self) -> float:
        """Score based on correctly flagged inconsistencies."""
        actual = self.answer_key.get("inconsistencies", [])

        if not actual and not self.flagged_inconsistencies:
            return 1.0
        if not actual and self.flagged_inconsistencies:
            return 0.0

        correct_flags = 0
        for flagged in self.flagged_inconsistencies:
            for actual_inc in actual:
                if self._fuzzy_match(flagged.lower(), actual_inc.lower()):
                    correct_flags += 1
                    break

        false_positives = len(self.flagged_inconsistencies) - correct_flags
        recall = correct_flags / len(actual)
        penalty = min(false_positives * 0.25, 0.5)
        return max(recall - penalty, 0.0)

    def _score_routing(self) -> float:
        """1.0 for correct routing/escalation, 0.0 otherwise."""
        should_escalate = self.answer_key.get("should_escalate", False)

        if should_escalate:
            return 1.0 if self.escalated else 0.0

        if not self.routed_to or self.routed_to == "escalated":
            return 0.0

        correct_dept = self.answer_key["correct_department"].strip().lower()
        return 1.0 if self.routed_to.strip().lower() == correct_dept else 0.0

    def _score_efficiency(self) -> float:
        """Score based on how efficiently the agent used its steps.

        Optimal = number of extractable fields + 2 (classify + route).
        Using fewer or equal steps → 1.0.
        Each extra step reduces the score gradually.
        """
        n_fields = len(self.answer_key.get("extractable_fields", {}))
        n_missing = len(self.answer_key.get("missing_fields", []))
        n_inconsistencies = len(self.answer_key.get("inconsistencies", []))

        # Optimal: classify + extract_all + flag_missing + flag_inconsistencies + route
        optimal = 1 + n_fields + n_missing + n_inconsistencies + 1

        if self.steps_taken <= optimal:
            return 1.0

        extra = self.steps_taken - optimal
        # Lose 0.1 per extra step, floor at 0.0
        return max(1.0 - extra * 0.1, 0.0)

    # ──────────────────────────────────────────────────────────────────
    # Utility: letter grade
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _letter_grade(score: float) -> str:
        if score >= 0.95:
            return "A+"
        if score >= 0.90:
            return "A"
        if score >= 0.85:
            return "A-"
        if score >= 0.80:
            return "B+"
        if score >= 0.75:
            return "B"
        if score >= 0.70:
            return "B-"
        if score >= 0.65:
            return "C+"
        if score >= 0.60:
            return "C"
        if score >= 0.55:
            return "C-"
        if score >= 0.50:
            return "D"
        return "F"

    # ──────────────────────────────────────────────────────────────────
    # Utility: fuzzy string matching
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _fuzzy_match(a: str, b: str, threshold: float = 0.80) -> bool:
        """Simple fuzzy string comparison.

        Returns True if the normalised strings are similar enough.
        Uses token-overlap ratio to allow minor wording differences
        (e.g. "$4,250.00" vs "4250.00", "Acme Corp" vs "Acme Corporation").
        """
        a_norm = a.strip().lower().replace(",", "").replace("$", "").replace("₹", "")
        b_norm = b.strip().lower().replace(",", "").replace("$", "").replace("₹", "")

        # Exact match after normalisation
        if a_norm == b_norm:
            return True

        # Token-level overlap
        tokens_a = set(a_norm.split())
        tokens_b = set(b_norm.split())

        if not tokens_a or not tokens_b:
            return False

        overlap = len(tokens_a & tokens_b)
        max_len = max(len(tokens_a), len(tokens_b))
        return (overlap / max_len) >= threshold

    # ──────────────────────────────────────────────────────────────────
    # Convenience helpers for agents
    # ──────────────────────────────────────────────────────────────────

    def get_available_actions(self) -> List[str]:
        """Return human-readable list of available actions."""
        return list(self.ACTION_NAMES)

    def get_action_descriptions(self) -> Dict[str, str]:
        """Return a mapping of action name → description for agent prompts."""
        return {
            "classify": "Classify the document type.  Parameter: document_type  (e.g. 'invoice')",
            "extract": "Extract a field value.  Parameter: field_name:value  (e.g. 'company_name:Acme Corp')",
            "validate": "Validate a previously extracted field.  Parameter: field_name",
            "flag_missing": "Flag a field as missing from the document.  Parameter: field_name",
            "flag_inconsistency": "Flag an inconsistency.  Parameter: description of the inconsistency",
            "request_clarification": "Request clarification.  Parameter: your question",
            "route_to": "Route document to a department (TERMINAL).  Parameter: department name",
            "escalate": "Escalate the document (TERMINAL).  Parameter: reason for escalation",
        }

    def get_departments(self) -> List[str]:
        """Return valid department names."""
        return list(self.DEPARTMENTS)

    def get_task_ids(self) -> List[str]:
        """Return all available task IDs."""
        return [t["task_id"] for t in self.tasks]

    def get_tasks_by_difficulty(self, difficulty: str) -> List[str]:
        """Return task IDs filtered by difficulty level."""
        return [t["task_id"] for t in self.tasks if t.get("difficulty") == difficulty]

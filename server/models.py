"""
OpenEnv-compatible Pydantic models for Document Triage Environment.

These extend the OpenEnv base Action/Observation types so that
create_app() can auto-generate schemas and the validator can
introspect the environment properly.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class DocumentTriageAction(Action):
    """Action for the Document Triage environment.

    Agents send one of 8 action types with a string parameter:
      0 = classify, 1 = extract, 2 = validate, 3 = flag_missing,
      4 = flag_inconsistency, 5 = request_clarification,
      6 = route_to (terminal), 7 = escalate (terminal).
    """

    action_type: int = Field(
        ...,
        ge=0,
        le=7,
        description=(
            "Action ID: 0=classify, 1=extract, 2=validate, 3=flag_missing, "
            "4=flag_inconsistency, 5=request_clarification, "
            "6=route_to (terminal), 7=escalate (terminal)"
        ),
    )
    parameter: str = Field(
        ...,
        description=(
            "Action parameter. Meaning depends on action_type: "
            "classify → document_type, extract → field_name:value, "
            "validate → field_name, flag_missing → field_name, "
            "flag_inconsistency → description, "
            "request_clarification → question, "
            "route_to → department, escalate → reason"
        ),
    )


class DocumentTriageObservation(Observation):
    """Observation returned by the Document Triage environment.

    Contains the current document, agent progress, and grading info.
    """

    document_text: str = Field(
        default="", description="Full text of the document being triaged"
    )
    document_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Structured metadata about the document"
    )
    classified_as: str = Field(
        default="", description="Current document classification (if set)"
    )
    extracted_fields: Dict[str, str] = Field(
        default_factory=dict, description="Fields extracted so far"
    )
    validated_fields: Dict[str, bool] = Field(
        default_factory=dict, description="Fields that have been validated"
    )
    flagged_missing: List[str] = Field(
        default_factory=list, description="Fields flagged as missing"
    )
    flagged_inconsistencies: List[str] = Field(
        default_factory=list, description="Inconsistencies flagged"
    )
    steps_taken: int = Field(default=0, description="Steps taken so far")
    steps_remaining: int = Field(default=15, description="Steps remaining")
    task_id: str = Field(default="", description="Current task ID")
    difficulty: str = Field(default="", description="Task difficulty (easy/medium/hard)")
    action_feedback: str = Field(
        default="", description="Feedback from the last action taken"
    )
    final_score: Optional[float] = Field(
        default=None, description="Final score (0.0-1.0), set when episode ends"
    )
    grade: Optional[str] = Field(
        default=None, description="Letter grade (A+/A/B/C/D/F), set when episode ends"
    )
    score_breakdown: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detailed grading breakdown by component, set when episode ends",
    )


class DocumentTriageState(State):
    """Extended state for the Document Triage environment."""

    task_id: str = Field(default="", description="Current task ID")
    difficulty: str = Field(default="", description="Task difficulty")
    is_done: bool = Field(default=False, description="Whether the episode has ended")
    total_reward: float = Field(default=0.0, description="Accumulated reward")

"""
OpenEnv-compatible Action and Observation models for Document Triage.

These pydantic models define the typed interface between the agent and environment.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentTriageAction(BaseModel):
    """An action the agent can take in the Document Triage environment.

    Attributes:
        action_type: Integer 0-7 mapping to action name.
            0=classify, 1=extract, 2=validate, 3=flag_missing,
            4=flag_inconsistency, 5=request_clarification,
            6=route_to (TERMINAL), 7=escalate (TERMINAL)
        parameter: Action-specific string parameter.
    """

    action_type: int = Field(..., ge=0, le=7, description="Action type (0-7)")
    parameter: str = Field(..., description="Action parameter string")


class DocumentTriageObservation(BaseModel):
    """The observation returned after each step.

    Attributes:
        document_text: The full text of the document to triage.
        metadata: JSON string of document metadata.
        classified_as: Current classification (empty if not yet classified).
        extracted_fields: JSON string of extracted field dict.
        validated_fields: JSON string of validated fields dict.
        flagged_missing: JSON string of flagged missing fields list.
        flagged_inconsistencies: JSON string of flagged inconsistency list.
        steps_taken: Steps taken so far.
        steps_remaining: Steps left before truncation.
        reward: Reward for the last action.
        done: Whether the episode is over.
        score: Final score (only present when done=True).
        grade: Letter grade (only present when done=True).
        score_breakdown: Per-component scores (only present when done=True).
        error: Error message if action was invalid.
    """

    document_text: str = ""
    metadata: str = "{}"
    classified_as: str = ""
    extracted_fields: str = "{}"
    validated_fields: str = "{}"
    flagged_missing: str = "[]"
    flagged_inconsistencies: str = "[]"
    steps_taken: int = 0
    steps_remaining: int = 15
    reward: float = 0.0
    done: bool = False
    score: Optional[float] = None
    grade: Optional[str] = None
    score_breakdown: Optional[Dict[str, float]] = None
    error: Optional[str] = None

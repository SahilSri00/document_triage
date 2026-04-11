"""
Formal OpenEnv Rubric classes for Document Triage Environment.

Uses the OpenEnv Rubric framework (WeightedSum, Gate, base Rubric)
to implement composable, introspectable reward computation.

Architecture:
    DocumentTriageRubric (WeightedSum)
    ├── ClassificationRubric    (15%)
    ├── ExtractionRubric        (30%)
    ├── MissingDetectionRubric  (15%)
    ├── InconsistencyRubric     (10%)
    ├── RoutingRubric           (20%)
    └── EfficiencyRubric        (10%)

Each child rubric computes its component score independently.
The parent WeightedSum aggregates them into a single 0.0–1.0 score.
"""

from typing import Any, Dict, List, Optional, Set

from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import WeightedSum


# ── Utility ──────────────────────────────────────────────────────────


def _fuzzy_match(a: str, b: str, threshold: float = 0.80) -> bool:
    """Simple fuzzy string comparison using token-overlap ratio.

    Handles minor wording differences (e.g. "$4,250.00" vs "4250.00",
    "Acme Corp" vs "Acme Corporation").
    """
    a_norm = a.strip().lower().replace(",", "").replace("$", "").replace("₹", "")
    b_norm = b.strip().lower().replace(",", "").replace("$", "").replace("₹", "")

    if a_norm == b_norm:
        return True

    tokens_a = set(a_norm.split())
    tokens_b = set(b_norm.split())

    if not tokens_a or not tokens_b:
        return False

    overlap = len(tokens_a & tokens_b)
    max_len = max(len(tokens_a), len(tokens_b))
    return (overlap / max_len) >= threshold


# Field name aliases → canonical name (same as environment.py)
FIELD_ALIASES: Dict[str, str] = {
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


def _resolve_field(name: str) -> str:
    """Resolve a field name through aliases to its canonical form."""
    return FIELD_ALIASES.get(name, name)


# ── Component Rubrics ────────────────────────────────────────────────


class ClassificationRubric(Rubric):
    """Score document classification (1.0 if correct, 0.0 otherwise)."""

    def forward(self, action: Any, observation: Any) -> float:
        answer_key = observation.get("answer_key", {})
        classified_as = observation.get("classified_as", "")

        if not classified_as:
            return 0.0

        correct = answer_key.get("correct_type", "").strip().lower()
        return 1.0 if classified_as.strip().lower() == correct else 0.0


class ExtractionRubric(Rubric):
    """Score field extraction as fraction of correctly extracted fields."""

    def forward(self, action: Any, observation: Any) -> float:
        answer_key = observation.get("answer_key", {})
        extracted_fields = observation.get("extracted_fields", {})
        correct_fields = answer_key.get("extractable_fields", {})

        if not correct_fields:
            return 1.0  # Nothing to extract → full marks

        correct_count = 0
        total = len(correct_fields)

        for field_name, correct_value in correct_fields.items():
            fn_lower = field_name.lower()
            extracted = extracted_fields.get(fn_lower)
            if not extracted:
                # Check aliases
                for ext_name, ext_val in extracted_fields.items():
                    if _resolve_field(ext_name) == fn_lower:
                        extracted = ext_val
                        break
            if extracted and _fuzzy_match(extracted, correct_value):
                correct_count += 1

        return correct_count / total


class MissingDetectionRubric(Rubric):
    """Score missing field detection: recall with false-positive penalty."""

    def forward(self, action: Any, observation: Any) -> float:
        answer_key = observation.get("answer_key", {})
        flagged_missing = observation.get("flagged_missing", [])
        actual_missing = [f.lower() for f in answer_key.get("missing_fields", [])]

        if not actual_missing and not flagged_missing:
            return 1.0  # Nothing missing & agent didn't flag → perfect
        if not actual_missing and flagged_missing:
            return 0.0  # False positives only

        correct_flags = sum(1 for f in flagged_missing if f.lower() in actual_missing)
        false_positives = len(flagged_missing) - correct_flags

        recall = correct_flags / len(actual_missing)
        penalty = min(false_positives * 0.2, 0.5)
        return max(recall - penalty, 0.0)


class InconsistencyRubric(Rubric):
    """Score inconsistency detection using fuzzy matching."""

    def forward(self, action: Any, observation: Any) -> float:
        answer_key = observation.get("answer_key", {})
        flagged = observation.get("flagged_inconsistencies", [])
        actual = answer_key.get("inconsistencies", [])

        if not actual and not flagged:
            return 1.0
        if not actual and flagged:
            return 0.0

        correct_flags = 0
        for flag in flagged:
            for actual_inc in actual:
                if _fuzzy_match(flag.lower(), actual_inc.lower()):
                    correct_flags += 1
                    break

        false_positives = len(flagged) - correct_flags
        recall = correct_flags / len(actual)
        penalty = min(false_positives * 0.25, 0.5)
        return max(recall - penalty, 0.0)


class RoutingRubric(Rubric):
    """Score routing decision (1.0 for correct, 0.0 otherwise)."""

    def forward(self, action: Any, observation: Any) -> float:
        answer_key = observation.get("answer_key", {})
        routed_to = observation.get("routed_to", "")
        escalated = observation.get("escalated", False)
        should_escalate = answer_key.get("should_escalate", False)

        if should_escalate:
            return 1.0 if escalated else 0.0

        if not routed_to or routed_to == "escalated":
            return 0.0

        correct_dept = answer_key.get("correct_department", "").strip().lower()
        return 1.0 if routed_to.strip().lower() == correct_dept else 0.0


class EfficiencyRubric(Rubric):
    """Score step efficiency: optimal steps = classify + extract_all + flags + route."""

    def forward(self, action: Any, observation: Any) -> float:
        answer_key = observation.get("answer_key", {})
        steps_taken = observation.get("steps_taken", 0)

        n_fields = len(answer_key.get("extractable_fields", {}))
        n_missing = len(answer_key.get("missing_fields", []))
        n_inconsistencies = len(answer_key.get("inconsistencies", []))

        optimal = 1 + n_fields + n_missing + n_inconsistencies + 1

        if steps_taken <= optimal:
            return 1.0

        extra = steps_taken - optimal
        return max(1.0 - extra * 0.1, 0.0)


# ── Top-Level Composed Rubric ────────────────────────────────────────


class DocumentTriageRubric(WeightedSum):
    """Top-level rubric for the Document Triage environment.

    Composes 6 component rubrics with the following weights:
        Classification:    15%
        Field Extraction:  30%
        Missing Detection: 15%
        Inconsistency:     10%
        Routing:           20%
        Efficiency:        10%

    Usage:
        rubric = DocumentTriageRubric()
        score = rubric(action=None, observation=episode_data)
        # score is 0.0 – 1.0

    Introspection:
        for name, child in rubric.named_children():
            print(f"{name}: {child.last_score}")
    """

    WEIGHTS = [0.15, 0.30, 0.15, 0.10, 0.20, 0.10]

    def __init__(self):
        components = [
            ClassificationRubric(),
            ExtractionRubric(),
            MissingDetectionRubric(),
            InconsistencyRubric(),
            RoutingRubric(),
            EfficiencyRubric(),
        ]
        super().__init__(rubrics=components, weights=self.WEIGHTS)

        # Assign named attributes for introspection
        self.classification = components[0]
        self.extraction = components[1]
        self.missing_detection = components[2]
        self.inconsistency = components[3]
        self.routing = components[4]
        self.efficiency = components[5]

    def score_episode(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience method: score an episode and return detailed breakdown.

        Args:
            episode_data: Dict containing answer_key, classified_as,
                          extracted_fields, flagged_missing, etc.

        Returns:
            Dict with total score, letter grade, and per-component breakdown.
        """
        total = self(action=None, observation=episode_data)

        breakdown = {
            "classification": self.classification.last_score,
            "extraction": self.extraction.last_score,
            "missing_detection": self.missing_detection.last_score,
            "inconsistency_detection": self.inconsistency.last_score,
            "routing": self.routing.last_score,
            "efficiency": self.efficiency.last_score,
        }

        return {
            "total_score": round(min(max(total, 0.0), 1.0), 4),
            "grade": _letter_grade(total),
            "breakdown": breakdown,
        }


def _letter_grade(score: float) -> str:
    """Convert a 0.0–1.0 score to a letter grade."""
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

"""
Baseline Agent for Document Triage Environment
================================================

A rule-based heuristic agent that demonstrates how to interact with the
environment. Uses keyword matching to classify, extract fields, detect
missing data, and route documents. Provides a performance baseline that
RL/LLM agents should beat.

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --difficulty hard
    python scripts/run_baseline.py --task easy_001
    python scripts/run_baseline.py --all
"""

import argparse
import json
import re
import sys
import os
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment import DocumentTriageEnv


# ── Keyword Lookup Tables ────────────────────────────────────────────

TYPE_KEYWORDS = {
    "invoice": ["invoice", "inv-", "amount due", "payment terms", "bill to", "factura"],
    "contract": ["agreement", "contract", "nda", "non-disclosure", "terms and conditions", "herein"],
    "memo": ["memo", "memorandum", "internal communication"],
    "report": ["report", "findings", "summary", "executive summary", "audit", "quarterly"],
    "letter": ["dear", "sincerely", "we are pleased", "appointment", "onboarding"],
    "form": ["purchase order", "po number", "po-", "authorization form", "requisition"],
    "receipt": ["receipt", "reimbursement", "expense", "payment method"],
    "proposal": ["proposal", "scope of services", "pricing", "valid until"],
    "notice": ["notice", "notification", "advisory", "alert"],
    "complaint": ["complaint", "harassment", "grievance", "hostile"],
}

DEPT_KEYWORDS = {
    "finance": ["invoice", "payment", "expense", "budget", "revenue", "financial", "receipt", "reimbursement", "cost", "amount"],
    "hr": ["employee", "onboarding", "salary", "leave", "performance", "hiring", "termination", "complaint", "harassment"],
    "legal": ["agreement", "contract", "nda", "non-disclosure", "legal", "litigation", "patent", "compliance"],
    "operations": ["purchase order", "warehouse", "delivery", "logistics", "inventory", "manufacturing", "proposal"],
    "it": ["security", "software", "cyber", "vulnerability", "system", "server", "database", "api"],
    "compliance": ["audit", "regulation", "compliance", "breach", "gdpr", "soc", "pci"],
    "executive": ["confidential", "board", "merger", "acquisition", "ceo", "executive"],
}

ESCALATION_KEYWORDS = [
    "fraud", "fraudulent", "forged", "suspected", "breach", "data breach",
    "harassment", "discrimination", "whistleblower", "retaliation",
    "material misstatement", "related party", "confidential",
]

# ── Field Extraction Patterns ────────────────────────────────────────

FIELD_PATTERNS = {
    "invoice_number": [r"[Ii]nvoice\s*(?:[Nn]umber|#|No\.?)\s*:?\s*([\w\-]+)"],
    "date": [r"[Dd]ate\s*:?\s*(\w+ \d{1,2},?\s*\d{4})"],
    "amount": [r"[Tt]otal\s*(?:[Aa]mount|[Dd]ue)?\s*:?\s*(\$[\d,]+\.?\d*)"],
    "vendor": [r"[Ff]rom\s*:?\s*(.+?)(?:\n|$)"],
    "due_date": [r"[Dd]ue\s*[Dd]ate\s*:?\s*(\w+ \d{1,2},?\s*\d{4})"],
    "employee_name": [r"[Dd]ear\s+(.+?)(?:,|\n)", r"[Ee]mployee\s*:?\s*(.+?)(?:\n|$)"],
    "po_number": [r"PO\s*(?:[Nn]umber|#)\s*:?\s*([\w\-]+)"],
    "report_id": [r"[Rr]eport\s*(?:ID|#)\s*:?\s*([\w\-]+)"],
    "proposal_number": [r"[Pp]roposal\s*(?:#|[Nn]umber)\s*:?\s*([\w\-]+)"],
}


class BaselineAgent:
    """Rule-based heuristic agent for benchmarking."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def act(self, observation: Dict[str, Any], step: int) -> Dict[str, Any]:
        """Decide next action based on current observation."""
        doc_text = observation["document_text"].lower()
        classified = observation["classified_as"]
        extracted = json.loads(observation["extracted_fields"]) if isinstance(observation["extracted_fields"], str) else observation["extracted_fields"]
        steps_remaining = observation["steps_remaining"]

        # Step 1: Classify if not done
        if not classified:
            doc_type = self._classify(doc_text)
            return {"action_type": 0, "parameter": doc_type}

        # Step 2: Extract fields
        field = self._find_next_field(doc_text, extracted)
        if field and steps_remaining > 2:
            name, value = field
            return {"action_type": 1, "parameter": f"{name}:{value}"}

        # Step 3: Check for escalation triggers
        if self._should_escalate(doc_text) and steps_remaining > 0:
            return {"action_type": 7, "parameter": "flagged by baseline heuristics"}

        # Step 4: Route to department
        dept = self._pick_department(doc_text, classified)
        return {"action_type": 6, "parameter": dept}

    def _classify(self, text: str) -> str:
        scores = {}
        for doc_type, keywords in TYPE_KEYWORDS.items():
            scores[doc_type] = sum(1 for kw in keywords if kw in text)
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "report"

    def _find_next_field(self, text: str, extracted: dict) -> Tuple[str, str] | None:
        original_text = text  # text is already lowered, need original for values
        for field_name, patterns in FIELD_PATTERNS.items():
            if field_name in extracted:
                continue
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return field_name, match.group(1).strip()
        return None

    def _should_escalate(self, text: str) -> bool:
        hits = sum(1 for kw in ESCALATION_KEYWORDS if kw in text)
        return hits >= 2

    def _pick_department(self, text: str, doc_type: str) -> str:
        scores = {}
        for dept, keywords in DEPT_KEYWORDS.items():
            scores[dept] = sum(1 for kw in keywords if kw in text)
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "operations"


def run_episode(env: DocumentTriageEnv, agent: BaselineAgent, task_id: str = None, difficulty: str = None) -> Dict[str, Any]:
    """Run one full episode and return results."""
    opts = {}
    if task_id:
        opts["task_id"] = task_id
    elif difficulty:
        opts["difficulty"] = difficulty

    obs, info = env.reset(options=opts if opts else None)
    task = info["task_id"]
    diff = info["difficulty"]

    step = 0
    terminated = truncated = False

    while not terminated and not truncated:
        action = agent.act(obs, step)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

    return {
        "task_id": task,
        "difficulty": diff,
        "steps": step,
        "final_score": info.get("final_score", 0),
        "grade": info.get("grade", "?"),
        "breakdown": info.get("score_breakdown", {}),
        "total_reward": info.get("total_reward", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Run baseline agent on Document Triage Environment")
    parser.add_argument("--task", type=str, help="Run a specific task ID")
    parser.add_argument("--difficulty", type=str, choices=["easy", "medium", "hard"], help="Filter by difficulty")
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step output")
    args = parser.parse_args()

    env = DocumentTriageEnv(tasks_path="tasks/tasks.json", max_steps=15)
    agent = BaselineAgent(verbose=not args.quiet)

    if args.task:
        result = run_episode(env, agent, task_id=args.task)
        _print_result(result)
    elif args.all:
        _run_all(env, agent)
    elif args.difficulty:
        tasks = env.get_tasks_by_difficulty(args.difficulty)
        results = [run_episode(env, agent, task_id=t) for t in tasks]
        _print_table(results)
    else:
        _run_all(env, agent)


def _run_all(env, agent):
    results = [run_episode(env, agent, task_id=t["task_id"]) for t in env.tasks]
    _print_table(results)

    # Summary by difficulty
    print("\n" + "=" * 60)
    print("  SUMMARY BY DIFFICULTY")
    print("=" * 60)
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            avg = sum(r["final_score"] for r in subset) / len(subset)
            print(f"  {diff:>8s}:  avg={avg:.2%}   n={len(subset)}")
    overall = sum(r["final_score"] for r in results) / len(results)
    print(f"  {'overall':>8s}:  avg={overall:.2%}   n={len(results)}")


def _print_result(r):
    print(f"\n📄 {r['task_id']}  ({r['difficulty']})")
    print(f"   Score: {r['final_score']:.2%}  Grade: {r['grade']}  Steps: {r['steps']}  Reward: {r['total_reward']:+.2f}")
    for k, v in r["breakdown"].items():
        print(f"     {k:.<28s} {v:.2%}")


def _print_table(results):
    print(f"\n{'Task':<14} {'Diff':<8} {'Score':>7} {'Grade':>6} {'Steps':>6} {'Reward':>8}")
    print("-" * 55)
    for r in results:
        print(f"{r['task_id']:<14} {r['difficulty']:<8} {r['final_score']:>6.1%} {r['grade']:>6} {r['steps']:>6} {r['total_reward']:>+7.2f}")


if __name__ == "__main__":
    main()

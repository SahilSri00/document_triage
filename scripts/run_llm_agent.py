"""
Gemini-Powered LLM Agent for Document Triage Environment
==========================================================

Uses Google Gemini Flash to intelligently process documents:
classify, extract fields, flag issues, and route/escalate.

Reads API key from GEMINI_API_KEY environment variable.

Usage:
    set GEMINI_API_KEY=your_key_here
    python scripts/run_llm_agent.py
    python scripts/run_llm_agent.py --task hard_001
    python scripts/run_llm_agent.py --difficulty hard
    python scripts/run_llm_agent.py --all
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import google.generativeai as genai

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment import DocumentTriageEnv

# ── Gemini Setup ──────────────────────────────────────────────────────

def get_model(model_name: str = "gemini-2.0-flash"):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set.")
        print("   Set it:  set GEMINI_API_KEY=your_key_here  (Windows)")
        print("            export GEMINI_API_KEY=your_key_here (Linux/Mac)")
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    print(f"🤖 Using model: {model_name}")
    return model


# ── Prompt Builder ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert document triage agent working in an office.
You receive a document and must process it by taking a sequence of actions.

AVAILABLE ACTIONS (return as JSON array):

1. {"action_type": 0, "parameter": "<doc_type>"}
   CLASSIFY - Identify the document type.
   Types: invoice, contract, memo, report, letter, form, receipt, proposal, notice, complaint

2. {"action_type": 1, "parameter": "<field_name>:<value>"}
   EXTRACT - Extract a specific field from the document. Use exact values from the text.

3. {"action_type": 2, "parameter": "<field_name>"}
   VALIDATE - Validate a field you already extracted.

4. {"action_type": 3, "parameter": "<field_name>"}
   FLAG MISSING - Flag a field that SHOULD be in this document but is NOT present.

5. {"action_type": 4, "parameter": "<description>"}
   FLAG INCONSISTENCY - Flag contradictions, mismatches, or suspicious discrepancies.

6. {"action_type": 5, "parameter": "<question>"}
   REQUEST CLARIFICATION - Ask about something ambiguous.

7. {"action_type": 6, "parameter": "<department>"}
   ROUTE TO - Send to final department. THIS ENDS THE EPISODE.
   Departments: finance, hr, legal, operations, it, compliance, executive

8. {"action_type": 7, "parameter": "<reason>"}
   ESCALATE - Escalate for urgent review. THIS ENDS THE EPISODE.
   Use ONLY for: fraud, harassment, data breaches, legal liability, executive matters.

RULES:
- ALWAYS classify first.
- Extract ALL key fields you can find (dates, amounts, names, IDs, etc.).
- Use SHORT, STANDARD field names for extraction:
  * For invoices: invoice_number, date, vendor, amount, due_date
  * For contracts: agreement_type, effective_date, parties, duration
  * For HR docs: employee_name, position, salary, start_date, department
  * For reports: report_id, date, total_vulnerabilities, remediation_deadline
  * For forms: po_number, date, vendor, total_amount, delivery_date
  * For receipts: employee_name, employee_id, amount, expense_type, date
  * For proposals: proposal_number, vendor, monthly_cost, contract_term, valid_until
  * Use snake_case for all field names.
- Flag any field that SHOULD exist but is MISSING from the document.
- Flag any INCONSISTENCIES (mismatched numbers, contradictions, suspicious details).
- End with EXACTLY ONE terminal action: either route_to OR escalate.
- Escalate ONLY when genuinely warranted (fraud, safety, legal risk, harassment, data breach).
- Keep total actions under 12 for efficiency.

Return ONLY a JSON array of actions. No explanation, no markdown, no code fences."""


def build_analysis_prompt(observation: Dict[str, Any]) -> str:
    metadata = observation["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    prompt = f"""{SYSTEM_PROMPT}

DOCUMENT TO PROCESS:
---
{observation['document_text']}
---

DOCUMENT METADATA:
{json.dumps(metadata, indent=2)}

Analyze this document carefully. Return your actions as a JSON array."""

    return prompt


# ── Response Parser ───────────────────────────────────────────────────

def parse_actions(response_text: str) -> List[Dict[str, Any]]:
    """Parse Gemini's response into a list of action dicts."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        actions = json.loads(text)
        if isinstance(actions, list):
            return actions
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the response
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    print(f"  ⚠️ Could not parse response: {text[:200]}")
    return []


# ── Episode Runner ────────────────────────────────────────────────────

def run_episode(
    env: DocumentTriageEnv,
    model: genai.GenerativeModel,
    task_id: str = None,
    difficulty: str = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one complete episode with the Gemini agent."""

    opts = {}
    if task_id:
        opts["task_id"] = task_id
    elif difficulty:
        opts["difficulty"] = difficulty

    obs, info = env.reset(options=opts if opts else None)
    tid = info["task_id"]
    diff = info["difficulty"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  📄 Task: {tid}  ({diff})")
        print(f"{'='*60}")

    # Ask Gemini to analyze and produce actions
    prompt = build_analysis_prompt(obs)

    try:
        response = model.generate_content(prompt)
        actions = parse_actions(response.text)
    except Exception as e:
        print(f"  ❌ Gemini API error: {e}")
        actions = [{"action_type": 6, "parameter": "operations"}]  # fallback

    if not actions:
        actions = [{"action_type": 6, "parameter": "operations"}]

    # Execute each action
    terminated = truncated = False
    step_count = 0

    for action in actions:
        if terminated or truncated:
            break

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        action_name = env.ACTION_NAMES[action.get("action_type", -1)] if 0 <= action.get("action_type", -1) < 8 else "?"

        if verbose:
            param = action.get("parameter", "")
            print(f"  Step {step_count}: {action_name}({param})  →  reward={reward:+.2f}")

    # If Gemini forgot the terminal action, force a route
    if not terminated and not truncated:
        if verbose:
            print(f"  ⚠️ No terminal action — forcing route to operations")
        _, _, terminated, truncated, info = env.step({"action_type": 6, "parameter": "operations"})
        step_count += 1

    final_score = info.get("final_score", 0)
    grade = info.get("grade", "?")
    breakdown = info.get("score_breakdown", {})

    if verbose:
        print(f"\n  🏆 Score: {final_score:.0%}  Grade: {grade}  Steps: {step_count}")
        for k, v in breakdown.items():
            print(f"     {k:.<30s} {v:.0%}")

    return {
        "task_id": tid,
        "difficulty": diff,
        "steps": step_count,
        "final_score": final_score,
        "grade": grade,
        "breakdown": breakdown,
        "total_reward": info.get("total_reward", 0),
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gemini LLM Agent for Document Triage")
    parser.add_argument("--task", type=str, help="Specific task ID")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"])
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash",
                        help="Gemini model name (e.g. gemini-1.5-flash, gemini-2.0-flash)")
    args = parser.parse_args()

    model = get_model(args.model)
    env = DocumentTriageEnv(tasks_path="tasks/tasks.json", max_steps=15)

    if args.task:
        run_episode(env, model, task_id=args.task, verbose=not args.quiet)
    elif args.difficulty:
        tasks = env.get_tasks_by_difficulty(args.difficulty)
        results = []
        for tid in tasks:
            results.append(run_episode(env, model, task_id=tid, verbose=not args.quiet))
            time.sleep(4)  # respect rate limits
        _print_table(results)
    elif args.all:
        _run_all(env, model, args.quiet)
    else:
        # Default: run one random task
        run_episode(env, model, verbose=True)


def _run_all(env, model, quiet):
    results = []
    for task in env.tasks:
        results.append(run_episode(env, model, task_id=task["task_id"], verbose=not quiet))
        time.sleep(4)  # 15 RPM = 1 every 4 seconds

    _print_table(results)

    # Comparison table
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            avg = sum(r["final_score"] for r in subset) / len(subset)
            print(f"  {diff:>8s}:  avg={avg:.1%}   n={len(subset)}")
    overall = sum(r["final_score"] for r in results) / len(results)
    print(f"  {'overall':>8s}:  avg={overall:.1%}   n={len(results)}")

    print(f"\n  📊 COMPARISON: Baseline=63.2%  →  Gemini Agent={overall:.1%}")


def _print_table(results):
    print(f"\n{'Task':<14} {'Diff':<8} {'Score':>7} {'Grade':>6} {'Steps':>6} {'Reward':>8}")
    print("-" * 55)
    for r in results:
        print(f"{r['task_id']:<14} {r['difficulty']:<8} {r['final_score']:>6.1%} {r['grade']:>6} {r['steps']:>6} {r['total_reward']:>+7.2f}")


if __name__ == "__main__":
    main()

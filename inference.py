import argparse
import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.environment import DocumentTriageEnv

# ── Configuration ─────────────────────────────────────────────────────

# Required environment variables (per submission checklist)
API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
HF_TOKEN = os.getenv("HF_TOKEN")

# Fallback for local testing
API_KEY = HF_TOKEN or os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")

BENCHMARK = "document_triage"
MAX_STEPS = 15
TEMPERATURE = 0.3
MAX_TOKENS = 2048

# ── Logging (mandatory stdout format) ─────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── System Prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert document triage agent working in an office.
You receive a document and must process it by taking a sequence of actions.

AVAILABLE ACTIONS (return as JSON array):

1. {"action_type": 0, "parameter": "<doc_type>"}
   CLASSIFY - Identify the document type.
   Types: invoice, contract, memo, report, letter, form, receipt, proposal, notice, complaint

2. {"action_type": 1, "parameter": "<field_name>:<value>"}
   EXTRACT - Extract a specific field. Use exact values from the text.

3. {"action_type": 2, "parameter": "<field_name>"}
   VALIDATE - Validate a previously extracted field.

4. {"action_type": 3, "parameter": "<field_name>"}
   FLAG MISSING - Flag a field that SHOULD be in this document but is NOT present.

5. {"action_type": 4, "parameter": "<description>"}
   FLAG INCONSISTENCIES - Flag contradictions, mismatches, or suspicious discrepancies.

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

Return ONLY a JSON array of actions. No explanation, no markdown, no code fences.
""").strip()


# ── LLM Call ──────────────────────────────────────────────────────────


def get_action_plan(client: OpenAI, document_text: str, metadata: str) -> List[Dict]:
    """Ask the LLM to analyze the document and return a list of actions."""
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            pass

    user_prompt = f"""DOCUMENT TO PROCESS:
---
{document_text}
---

DOCUMENT METADATA:
{json.dumps(metadata, indent=2) if isinstance(metadata, dict) else metadata}

Analyze this document carefully. Return your actions as a JSON array."""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return _parse_actions(text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return [{"action_type": 6, "parameter": "operations"}]


def _parse_actions(text: str) -> List[Dict]:
    """Parse LLM response into a list of action dicts."""
    # Strip markdown fences
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

    # Try to find JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return [{"action_type": 6, "parameter": "operations"}]


# ── Episode Runner ────────────────────────────────────────────────────


def run_episode(env: DocumentTriageEnv, client: OpenAI, task_id: str = None) -> Dict[str, Any]:
    """Run one complete episode and emit [START]/[STEP]/[END] logs."""
    options = {"task_id": task_id} if task_id else None
    obs, info = env.reset(options=options)
    tid = info["task_id"]

    log_start(task=tid, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        # Get action plan from LLM
        actions = get_action_plan(client, obs["document_text"], obs["metadata"])

        if not actions:
            actions = [{"action_type": 6, "parameter": "operations"}]

        terminated = truncated = False

        for action in actions:
            if terminated or truncated:
                break

            obs, reward, terminated, truncated, step_info = env.step(action)
            steps_taken += 1
            done = terminated or truncated

            action_name = env.ACTION_NAMES[action.get("action_type", -1)] \
                if 0 <= action.get("action_type", -1) < 8 else "unknown"
            param = action.get("parameter", "")
            action_str = f"{action_name}({param})"

            error = step_info.get("error", None)
            rewards.append(reward)

            log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=error)

        # Force terminal if LLM forgot
        if not terminated and not truncated:
            fallback = {"action_type": 6, "parameter": "operations"}
            obs, reward, terminated, truncated, step_info = env.step(fallback)
            steps_taken += 1
            rewards.append(reward)
            log_step(
                step=steps_taken,
                action="route_to(operations)",
                reward=reward,
                done=True,
                error=None,
            )

        score = step_info.get("final_score", 0.0)
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.5

    except Exception as e:
        print(f"[DEBUG] Episode failed: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": tid,
        "steps": steps_taken,
        "score": score,
        "success": success,
        "rewards": rewards,
        "grade": info.get("grade", "F") if 'step_info' not in locals() else step_info.get("grade", "F"),
    }


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Document Triage Inference Script")
    parser.add_argument("--task", type=str, help="Specific task ID")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"])
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    args = parser.parse_args()

    # The hackathon evaluator requires the LLM client to be properly set,
    # but if it fails to set HF_TOKEN we shouldn't completely crash because the validation 
    # needs to see the [START] and [END] logs to detect "tasks with graders".
    # So we'll pass a dummy key if nothing is set, so the logic flows and we get 0.0 score logging.
    actual_api_key = API_KEY or "dummy_key_for_validation"

    client = OpenAI(base_url=API_BASE_URL, api_key=actual_api_key)
    env = DocumentTriageEnv(tasks_path="tasks/tasks.json", max_steps=MAX_STEPS)

    if args.task:
        run_episode(env, client, task_id=args.task)
    elif args.difficulty:
        for tid in env.get_tasks_by_difficulty(args.difficulty):
            run_episode(env, client, task_id=tid)
            time.sleep(1)
    else:
        # Default: run ALL 15 tasks (easy, medium, hard)
        # --all flag also triggers this path for backward compat
        all_task_ids = [t["task_id"] for t in env.tasks]
        print(f"[INFO] Running all {len(all_task_ids)} tasks: {all_task_ids}", flush=True)
        results = []
        for tid in all_task_ids:
            r = run_episode(env, client, task_id=tid)
            results.append(r)
            time.sleep(1)

        # Summary table
        print(f"\n{'Task':<14} {'Score':>7} {'Grade':>6} {'Steps':>6}")
        print("-" * 40)
        for r in results:
            print(f"{r['task_id']:<14} {r['score']:>6.1%} {r['grade']:>6} {r['steps']:>6}")
        avg = sum(r["score"] for r in results) / len(results)
        print(f"\nOverall: {avg:.1%} across {len(results)} tasks")

if __name__ == "__main__":
    main()


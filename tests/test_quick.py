"""Quick smoke test — run this to verify the environment works."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import DocumentTriageEnv


def main():
    env = DocumentTriageEnv(tasks_path="tasks/tasks.json", max_steps=15)
    print(f"✅ Environment created  |  {len(env.tasks)} tasks loaded")
    print(f"   Difficulties: { {d: len(env.get_tasks_by_difficulty(d)) for d in ['easy','medium','hard']} }")

    obs, info = env.reset(options={"task_id": "easy_001"})
    print(f"\n📄 Task: {info['task_id']}  ({info['difficulty']})")
    print(f"   Doc preview: {obs['document_text'][:80]}...")

    # Run a simple sequence
    actions = [
        {"action_type": 0, "parameter": "invoice"},
        {"action_type": 1, "parameter": "invoice_number:INV-2024-0847"},
        {"action_type": 1, "parameter": "vendor:Acme Corporation"},
        {"action_type": 1, "parameter": "amount:$4,250.00"},
        {"action_type": 1, "parameter": "date:March 15, 2024"},
        {"action_type": 1, "parameter": "due_date:April 14, 2024"},
        {"action_type": 6, "parameter": "finance"},
    ]

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        name = env.ACTION_NAMES[action["action_type"]]
        print(f"   Step {env.steps_taken}: {name}({action['parameter']})  →  reward={reward:+.2f}")
        if terminated or truncated:
            break

    print(f"\n🏆 FINAL SCORE: {info['final_score']:.2%}  (Grade: {info['grade']})")
    breakdown = info["score_breakdown"]
    for k, v in breakdown.items():
        print(f"   {k:.<30s} {v:.2%}")

    env.render()
    print("\n✅ All checks passed!")


if __name__ == "__main__":
    main()

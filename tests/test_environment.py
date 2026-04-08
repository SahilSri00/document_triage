"""Comprehensive pytest tests for DocumentTriageEnv."""

import json
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment import DocumentTriageEnv


@pytest.fixture
def env():
    return DocumentTriageEnv(tasks_path="tasks/tasks.json", max_steps=15)


@pytest.fixture
def env_easy(env):
    env.reset(options={"task_id": "easy_001"})
    return env


# ── Construction & Reset ─────────────────────────────────────────────

class TestConstruction:
    def test_loads_tasks(self, env):
        assert len(env.tasks) >= 10

    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert "document_text" in obs
        assert "task_id" in info

    def test_reset_specific_task(self, env):
        obs, info = env.reset(options={"task_id": "easy_001"})
        assert info["task_id"] == "easy_001"
        assert info["difficulty"] == "easy"

    def test_reset_by_difficulty(self, env):
        _, info = env.reset(options={"difficulty": "hard"})
        assert info["difficulty"] == "hard"

    def test_reset_clears_state(self, env):
        env.reset(options={"task_id": "easy_001"})
        env.step({"action_type": 0, "parameter": "invoice"})
        env.reset(options={"task_id": "easy_002"})
        assert env.classified_as == ""
        assert env.steps_taken == 0

    def test_invalid_task_id(self, env):
        with pytest.raises(ValueError):
            env.reset(options={"task_id": "nonexistent_999"})


# ── Observation ──────────────────────────────────────────────────────

class TestObservation:
    def test_obs_has_all_keys(self, env_easy):
        obs = env_easy._build_observation()
        expected = {
            "document_text", "metadata", "classified_as", "extracted_fields",
            "validated_fields", "flagged_missing", "flagged_inconsistencies",
            "clarifications_requested", "steps_taken", "steps_remaining",
            "action_history",
        }
        assert set(obs.keys()) == expected

    def test_obs_does_not_leak_answer_key(self, env_easy):
        obs = env_easy._build_observation()
        obs_str = json.dumps(obs)
        assert "answer_key" not in obs_str
        assert "correct_department" not in obs_str


# ── Action Parsing ───────────────────────────────────────────────────

class TestActionParsing:
    def test_dict_format(self, env):
        t, p = env._parse_action({"action_type": 0, "parameter": "invoice"})
        assert t == 0 and p == "invoice"

    def test_tuple_format(self, env):
        t, p = env._parse_action((1, "amount:$100"))
        assert t == 1 and p == "amount:$100"

    def test_string_format(self, env):
        t, p = env._parse_action("classify invoice")
        assert t == 0 and p == "invoice"

    def test_invalid_string(self, env):
        t, _ = env._parse_action("gibberish blah")
        assert t == -1


# ── Work Actions ─────────────────────────────────────────────────────

class TestClassify:
    def test_correct(self, env_easy):
        _, r, *_ = env_easy.step({"action_type": 0, "parameter": "invoice"})
        assert r > 0

    def test_wrong(self, env_easy):
        _, r, *_ = env_easy.step({"action_type": 0, "parameter": "memo"})
        assert r < 0

    def test_reclassify_penalty(self, env_easy):
        env_easy.step({"action_type": 0, "parameter": "invoice"})
        _, r, *_ = env_easy.step({"action_type": 0, "parameter": "invoice"})
        assert r < 0  # reclassify penalty + step penalty


class TestExtract:
    def test_correct_field(self, env_easy):
        _, r, *_ = env_easy.step({"action_type": 1, "parameter": "vendor:Acme Corporation"})
        assert r > 0

    def test_wrong_value(self, env_easy):
        _, r, *_ = env_easy.step({"action_type": 1, "parameter": "vendor:Wrong Company"})
        assert r < 0

    def test_missing_colon(self, env_easy):
        _, r, *_ = env_easy.step({"action_type": 1, "parameter": "vendor"})
        assert r < 0

    def test_duplicate_extract(self, env_easy):
        env_easy.step({"action_type": 1, "parameter": "vendor:Acme Corporation"})
        _, r, *_ = env_easy.step({"action_type": 1, "parameter": "vendor:Acme Corporation"})
        assert r < 0


class TestValidate:
    def test_correct_extraction_validates(self, env_easy):
        env_easy.step({"action_type": 1, "parameter": "vendor:Acme Corporation"})
        _, r, *_ = env_easy.step({"action_type": 2, "parameter": "vendor"})
        assert r > 0
        assert env_easy.validated_fields["vendor"] is True

    def test_wrong_extraction_fails(self, env_easy):
        env_easy.step({"action_type": 1, "parameter": "vendor:Wrong"})
        _, r, *_ = env_easy.step({"action_type": 2, "parameter": "vendor"})
        assert env_easy.validated_fields["vendor"] is False

    def test_validate_unextracted(self, env_easy):
        _, r, *_ = env_easy.step({"action_type": 2, "parameter": "vendor"})
        assert r < 0


class TestFlagMissing:
    def test_correct_flag(self, env):
        env.reset(options={"task_id": "med_001"})  # has missing PO number
        _, r, *_ = env.step({"action_type": 3, "parameter": "purchase_order_number"})
        assert r > 0

    def test_wrong_flag(self, env_easy):
        _, r, *_ = env_easy.step({"action_type": 3, "parameter": "invoice_number"})
        assert r < 0


class TestFlagInconsistency:
    def test_correct_flag(self, env):
        env.reset(options={"task_id": "med_002"})  # has salary mismatch
        _, r, *_ = env.step({"action_type": 4, "parameter": "Salary mismatch between offer letter ($140,000) and contract ($145,000)"})
        assert r > 0

    def test_wrong_flag(self, env_easy):
        _, r, *_ = env_easy.step({"action_type": 4, "parameter": "random inconsistency"})
        assert r < 0


# ── Terminal Actions ─────────────────────────────────────────────────

class TestRouting:
    def test_correct_route_terminates(self, env_easy):
        _, _, terminated, _, info = env_easy.step({"action_type": 6, "parameter": "finance"})
        assert terminated is True
        assert "final_score" in info

    def test_wrong_route(self, env_easy):
        _, r, terminated, _, _ = env_easy.step({"action_type": 6, "parameter": "hr"})
        assert terminated is True
        assert r < 0

    def test_route_when_should_escalate(self, env):
        env.reset(options={"task_id": "hard_001"})  # should escalate
        _, r, terminated, _, _ = env.step({"action_type": 6, "parameter": "finance"})
        assert terminated
        assert r < 0


class TestEscalation:
    def test_correct_escalation(self, env):
        env.reset(options={"task_id": "hard_001"})
        _, r, terminated, _, _ = env.step({"action_type": 7, "parameter": "suspected fraud"})
        assert terminated
        assert r > 0

    def test_wrong_escalation(self, env_easy):
        _, r, terminated, _, _ = env_easy.step({"action_type": 7, "parameter": "no reason"})
        assert terminated
        assert r < 0


# ── Episode Flow ─────────────────────────────────────────────────────

class TestEpisodeFlow:
    def test_truncation_on_max_steps(self, env):
        env_short = DocumentTriageEnv(tasks_path="tasks/tasks.json", max_steps=3)
        env_short.reset(options={"task_id": "easy_001"})
        for i in range(3):
            _, _, term, trunc, info = env_short.step({"action_type": 1, "parameter": f"field{i}:val{i}"})
        assert trunc is True
        assert "final_score" in info

    def test_cascade_step_counters(self, env_easy):
        assert env_easy.steps_taken == 0
        assert env_easy.steps_remaining == 15
        env_easy.step({"action_type": 0, "parameter": "invoice"})
        assert env_easy.steps_taken == 1
        assert env_easy.steps_remaining == 14

    def test_action_history_recorded(self, env_easy):
        env_easy.step({"action_type": 0, "parameter": "invoice"})
        assert len(env_easy.action_history) == 1
        assert env_easy.action_history[0]["action"] == "classify"


# ── Grader ───────────────────────────────────────────────────────────

class TestGrader:
    def test_perfect_score(self, env):
        env.reset(options={"task_id": "easy_001"})
        env.step({"action_type": 0, "parameter": "invoice"})
        env.step({"action_type": 1, "parameter": "invoice_number:INV-2024-0847"})
        env.step({"action_type": 1, "parameter": "vendor:Acme Corporation"})
        env.step({"action_type": 1, "parameter": "amount:$4,250.00"})
        env.step({"action_type": 1, "parameter": "date:March 15, 2024"})
        env.step({"action_type": 1, "parameter": "due_date:April 14, 2024"})
        _, _, _, _, info = env.step({"action_type": 6, "parameter": "finance"})
        assert info["final_score"] >= 0.95
        assert info["grade"] in ("A+", "A")

    def test_zero_effort_score(self, env):
        env.reset(options={"task_id": "easy_001"})
        _, _, _, _, info = env.step({"action_type": 6, "parameter": "wrong_dept"})
        assert info["final_score"] < 0.5

    def test_score_between_zero_and_one(self, env_easy):
        _, _, _, _, info = env_easy.step({"action_type": 6, "parameter": "finance"})
        assert 0.0 <= info["final_score"] <= 1.0

    def test_render_works(self, env_easy):
        output = env_easy.render()
        assert "DOCUMENT TRIAGE ENVIRONMENT" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

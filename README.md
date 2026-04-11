---
title: Document Triage
emoji: 📄
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# 📄 Document Triage & Routing Environment

A **Gymnasium-based reinforcement learning environment** built on the [OpenEnv](https://github.com/pytorch/openenv) framework, where AI agents learn to process, classify, and route office documents — simulating a real-world document triage workflow.

> Built for the **Meta PyTorch × Scaler Hackathon** using OpenEnv's standard API.

---

## 🎯 Overview

An AI agent receives a document and must perform a multi-step triage workflow:

1. **Classify** its type (invoice, contract, report, complaint, etc.)
2. **Extract** key fields (amounts, dates, names, IDs)
3. **Validate** extracted information against the document
4. **Flag** missing fields or inconsistencies
5. **Route** to the correct department — or **escalate** when warranted

The environment features **15 curated tasks** across 3 difficulty levels (easy, medium, hard), a **6-component weighted grader** built using OpenEnv's formal Rubric system, and full OpenEnv HTTP API compliance.

---

## 🏗️ Architecture

```
document_triage/
├── src/
│   ├── environment.py          # DocumentTriageEnv (Gymnasium core)
│   └── rubrics.py              # OpenEnv Rubric classes (WeightedSum)
├── server/
│   ├── app.py                  # OpenEnv create_app() entry point
│   ├── document_triage_environment.py  # OpenEnv Environment wrapper
│   └── models.py               # Pydantic Action/Observation/State
├── tasks/
│   └── tasks.json              # 15 tasks (5 easy / 5 medium / 5 hard)
├── tests/
│   ├── test_quick.py           # Smoke test
│   └── test_environment.py     # Full pytest suite (38 tests)
├── inference.py                # LLM inference agent (OpenAI client)
├── openenv.yaml                # OpenEnv manifest (15 tasks + graders)
├── Dockerfile                  # Docker deployment
├── pyproject.toml              # Project metadata + dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Install

```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run tests

```bash
pytest tests/ -v
```

### 3. Start the OpenEnv server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 4. Run the LLM inference agent

```bash
export GEMINI_API_KEY=your_key  # or HF_TOKEN
python inference.py              # Runs all 15 tasks
python inference.py --task hard_001  # Run a specific task
python inference.py --difficulty easy  # Run all easy tasks
```

---

## 🎮 Action Space

| ID  | Action                  | Parameter                                        | Type         |
| --- | ----------------------- | ------------------------------------------------ | ------------ |
| 0   | `classify`              | document type (e.g. `"invoice"`)                 | Work         |
| 1   | `extract`               | `"field_name:value"` (e.g. `"amount:$4,250.00"`) | Work         |
| 2   | `validate`              | field name (e.g. `"amount"`)                     | Work         |
| 3   | `flag_missing`          | field name (e.g. `"po_number"`)                  | Work         |
| 4   | `flag_inconsistency`    | description (e.g. `"salary mismatch"`)           | Work         |
| 5   | `request_clarification` | question text                                    | Work         |
| 6   | `route_to`              | department (e.g. `"finance"`)                    | **Terminal** |
| 7   | `escalate`              | reason text                                      | **Terminal** |

**Terminal actions** end the episode immediately. **Work actions** continue the episode.

### Action Formats

```python
# Dict (standard)
env.step({"action_type": 0, "parameter": "invoice"})

# Tuple
env.step((0, "invoice"))

# String
env.step("classify invoice")
```

---

## 📊 Grading System (OpenEnv Rubric)

The environment uses OpenEnv's formal **Rubric** framework (`WeightedSum` composition) for introspectable evaluation:

```
DocumentTriageRubric (WeightedSum)
├── ClassificationRubric    (15%)  — correct document type?
├── ExtractionRubric        (30%)  — fraction of fields correctly extracted
├── MissingDetectionRubric  (15%)  — correctly flagged missing fields
├── InconsistencyRubric     (10%)  — correctly flagged inconsistencies
├── RoutingRubric           (20%)  — correct department / escalation
└── EfficiencyRubric        (10%)  — steps used vs. optimal
```

**Introspection** (after scoring):

```python
from src.rubrics import DocumentTriageRubric

rubric = DocumentTriageRubric()
result = rubric.score_episode(episode_data)
# result = {"total_score": 0.94, "grade": "A", "breakdown": {...}}

for name, child in rubric.named_children():
    print(f"{name}: {child.last_score}")
```

The final score produces a letter grade: **A+ → F**.

---

## 📋 Task Catalogue (15 Tasks)

| ID        | Difficulty | Document Type       | Key Challenge                            |
| --------- | ---------- | ------------------- | ---------------------------------------- |
| easy_001  | Easy       | Invoice             | Standard classification + extraction     |
| easy_002  | Easy       | Onboarding Letter   | HR document routing                      |
| easy_003  | Easy       | Purchase Order      | Line item extraction                     |
| easy_004  | Easy       | Expense Receipt     | Employee reimbursement                   |
| easy_005  | Easy       | NDA                 | Legal contract identification            |
| med_001   | Medium     | Invoice             | Missing PO reference                     |
| med_002   | Medium     | Employment Contract | Salary inconsistency with offer letter   |
| med_003   | Medium     | Vendor Proposal     | Missing start date + payment terms       |
| med_004   | Medium     | Travel Auth Form    | Personal travel inflating company costs  |
| med_005   | Medium     | Security Audit      | Critical CVEs, missing remediation owner |
| hard_001  | Hard       | Fraudulent Invoice  | Forged signature, offshore payment       |
| hard_002  | Hard       | Harassment Complaint| Prior HR inaction, requires escalation   |
| hard_003  | Hard       | Bilingual Invoice   | Spanish/English, currency conversion     |
| hard_004  | Hard       | Data Breach Notice  | 23K records, GDPR, multi-dept response   |
| hard_005  | Hard       | Financial Report    | Revenue discrepancy, related-party txn   |

---

## 🌐 OpenEnv HTTP API

The server exposes all standard OpenEnv endpoints via `create_app()`:

| Method | Path           | Description                        |
| ------ | -------------- | ---------------------------------- |
| `GET`  | `/health`      | Health check (`{"status":"healthy"}`) |
| `GET`  | `/metadata`    | Environment name + description     |
| `GET`  | `/schema`      | Action/Observation/State schemas   |
| `GET`  | `/state`       | Current episode state              |
| `POST` | `/reset`       | Start a new episode                |
| `POST` | `/step`        | Take an action                     |
| `POST` | `/mcp`         | JSON-RPC MCP endpoint              |
| `GET`  | `/openapi.json`| Full OpenAPI specification         |

### Validation

```bash
openenv validate .                                  # Local structure check
openenv validate --url http://localhost:7860         # Runtime API test (6/6 criteria)
```

---

## 🐳 Docker

```bash
docker build -t document-triage .
docker run -p 7860:7860 document-triage
```

---

## 🧠 Design Principles

1. **Penalize, Don't Block** — Every action is accepted. Bad actions score poorly rather than crashing. This creates a meaningful and continuous learning signal.
2. **Cascade Effect** — Every action changes multiple state variables (step counters, workspace, action history).
3. **State ≠ Observation** — The answer key is hidden. The agent never sees the correct answers directly.
4. **Difficulty Scaling** — Easy (clean docs) → Medium (missing fields, inconsistencies) → Hard (fraud, multilingual, escalation-required).
5. **Framework-Native** — Uses OpenEnv's Rubric system (`WeightedSum`, `Rubric`) for composable, introspectable evaluation.

---

## 📜 License

MIT

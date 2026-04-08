---
title: Document Triage
emoji: 👀
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: "3.50.2"
app_file: app.py
pinned: false
---

# 📄 Document Triage & Routing Environment

A **Gymnasium-based reinforcement learning environment** where AI agents learn to process, classify, and route office documents — simulating a real-world document triage workflow.

---

## 🎯 Overview

An AI agent receives a document and must:

1. **Classify** its type (invoice, contract, report, etc.)
2. **Extract** key fields (amounts, dates, names)
3. **Validate** extracted information against the document
4. **Flag** missing fields or inconsistencies
5. **Route** to the correct department — or **escalate** when needed

The environment follows the **Gymnasium API** (`reset` → `step` → `reward` → `done`) and includes a **6-component grader** that produces a final score from 0–100%.

---

## 🏗️ Architecture

```
document_triage/
├── src/
│   ├── __init__.py
│   └── environment.py          # DocumentTriageEnv (core)
├── tasks/
│   └── tasks.json              # 15 tasks (5 easy / 5 medium / 5 hard)
├── scripts/
│   └── run_baseline.py         # Rule-based baseline agent
├── tests/
│   ├── test_quick.py           # Smoke test
│   └── test_environment.py     # Full pytest suite
├── api.py                      # FastAPI REST wrapper
├── Dockerfile                  # Docker deployment
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🚀 Quick Start

### 1. Create virtual environment & install dependencies

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the smoke test

```bash
python tests/test_quick.py
```

### 3. Run the baseline agent

```bash
python scripts/run_baseline.py --all
```

### 4. Run the full test suite

```bash
pytest tests/ -v
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

## 📊 Grader (6 Components)

| Component               | Weight | Description                                                    |
| ----------------------- | ------ | -------------------------------------------------------------- |
| Classification          | 15%    | Correct document type?                                         |
| Field Extraction        | 30%    | Fraction of fields correctly extracted                         |
| Missing Detection       | 15%    | Correctly flagged missing fields (with false-positive penalty) |
| Inconsistency Detection | 10%    | Correctly flagged inconsistencies                              |
| Routing                 | 20%    | Correct department / escalation decision                       |
| Efficiency              | 10%    | Steps used vs. optimal                                         |

The final score produces a letter grade: **A+ → F**.

---

## 🌐 REST API

Start the server:

```bash
uvicorn api:app --host 0.0.0.0 --port 7860 --reload
```

### Endpoints

| Method | Path       | Description                |
| ------ | ---------- | -------------------------- |
| `POST` | `/reset`   | Start a new episode        |
| `POST` | `/step`    | Take an action             |
| `GET`  | `/tasks`   | List available tasks       |
| `GET`  | `/actions` | List actions & departments |
| `GET`  | `/health`  | Health check               |

### Example

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_001"}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "...", "action_type": 0, "parameter": "invoice"}'
```

---

## 🐳 Docker

```bash
docker build -t document-triage .
docker run -p 7860:7860 document-triage
```

---

## 🧠 Design Principles

1. **Penalize, Don't Block** — Every action is accepted. Bad actions score poorly. This creates a meaningful learning signal.
2. **Cascade Effect** — Every action changes multiple state variables (step counters, workspace, history).
3. **State ≠ Observation** — The answer key is hidden. The agent never sees the correct answers directly.
4. **Difficulty Scaling** — Easy (clean docs) → Medium (missing fields, inconsistencies) → Hard (fraud, multilingual, escalation-required).

---

## 📈 Extending

### Add more tasks

Edit `tasks/tasks.json`. Each task needs:

```json
{
  "task_id": "easy_006",
  "difficulty": "easy",
  "document_text": "...",
  "metadata": { "priority": "normal", "source": "email", ... },
  "answer_key": {
    "correct_type": "invoice",
    "extractable_fields": { "field": "value", ... },
    "missing_fields": [],
    "inconsistencies": [],
    "correct_department": "finance",
    "should_escalate": false
  }
}
```

### Build a custom agent

```python
from src.environment import DocumentTriageEnv

env = DocumentTriageEnv(tasks_path="tasks/tasks.json")
obs, info = env.reset()

# Your agent logic here
action = {"action_type": 0, "parameter": "invoice"}
obs, reward, terminated, truncated, info = env.step(action)
```

---

## 📜 License

MIT

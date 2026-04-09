"""Verify all OpenEnv endpoints work for Phase 2 validation."""
import requests
import json

base = "http://localhost:7860"
all_pass = True

def check(name, condition, msg=""):
    global all_pass
    status = "PASS" if condition else "FAIL"
    if not condition:
        all_pass = False
    print(f"  [{status}] {name}" + (f" - {msg}" if msg else ""))

print("=" * 60)
print("OpenEnv Runtime Validation Test")
print("=" * 60)

# 1. Health
print("\n1. GET /health")
r = requests.get(f"{base}/health")
h = r.json()
check("status_code == 200", r.status_code == 200)
check("status == healthy", h.get("status") == "healthy", str(h))

# 2. Metadata
print("\n2. GET /metadata")
r = requests.get(f"{base}/metadata")
m = r.json()
check("has name", isinstance(m.get("name"), str), m.get("name", ""))
check("has description", isinstance(m.get("description"), str))

# 3. Schema
print("\n3. GET /schema")
r = requests.get(f"{base}/schema")
s = r.json()
check("has action schema", isinstance(s.get("action"), dict))
check("has observation schema", isinstance(s.get("observation"), dict))
check("has state schema", isinstance(s.get("state"), dict))

# 4. OpenAPI
print("\n4. GET /openapi.json")
r = requests.get(f"{base}/openapi.json")
o = r.json()
paths = list(o.get("paths", {}).keys())
check("has info.version", isinstance(o.get("info", {}).get("version"), str))
check("/reset in paths", "/reset" in paths, str(paths))
check("/step in paths", "/step" in paths)
check("/state in paths", "/state" in paths)

# 5. Reset (3 different tasks — this is "tasks with graders")
print("\n5. POST /reset (3 tasks)")
for task_id in ["easy_001", "med_001", "hard_001"]:
    r = requests.post(f"{base}/reset", json={"task_id": task_id})
    d = r.json()
    obs = d.get("observation", {})
    check(
        f"reset {task_id}",
        r.status_code == 200 and "observation" in d,
        f"status={r.status_code}, has_obs={'observation' in d}"
    )

# 6. Reset + step with default (stateless — creates fresh env each call)
print("\n6. POST /reset + /step (default task)")
r = requests.post(f"{base}/reset", json={})
d = r.json()
check("reset default", r.status_code == 200)
obs = d.get("observation", {})
check("observation has document_text", len(obs.get("document_text", "")) > 0)
check("observation has steps_remaining", obs.get("steps_remaining", 0) > 0)

# Step is stateless per HTTP call — each /step creates a new env
# The important thing is that it accepts actions and returns valid responses
r = requests.post(f"{base}/step", json={"action": {"action_type": 0, "parameter": "invoice"}})
d = r.json()
check("step returns 200", r.status_code == 200)
check("step has observation", "observation" in d)
check("step has reward", d.get("reward") is not None, f"reward={d.get('reward')}")

# 7. State endpoint
print("\n7. GET /state")
r = requests.get(f"{base}/state")
d = r.json()
check("state returns 200", r.status_code == 200)
check("state has episode_id", "episode_id" in d)
check("state has step_count", "step_count" in d)

# 8. MCP endpoint
print("\n8. POST /mcp")
r = requests.post(f"{base}/mcp", json={"jsonrpc": "2.0", "method": "tools/list", "id": 1, "params": {}})
check("mcp returns 200", r.status_code == 200)

print("\n" + "=" * 60)
if all_pass:
    print("ALL TESTS PASSED - Ready for Phase 2 validation!")
else:
    print("SOME TESTS FAILED - Review above")
print("=" * 60)

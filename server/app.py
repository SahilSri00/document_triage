"""
FastAPI application for Document Triage Environment.

Uses OpenEnv's create_app() factory to generate all standard endpoints:
  /health, /metadata, /schema, /reset, /step, /state, /ws, /mcp, /openapi.json

This is the entry point referenced in openenv.yaml and Dockerfile.
"""

import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openenv.core.env_server import create_app

from server.document_triage_environment import DocumentTriageEnvironment
from server.models import DocumentTriageAction, DocumentTriageObservation

# create_app expects a *callable* (factory), not an instance.
# Passing the class itself works — it gets called once per session.
app = create_app(
    DocumentTriageEnvironment,
    DocumentTriageAction,
    DocumentTriageObservation,
    env_name="document_triage",
)


def main():
    """Entry point for `uv run server` / `python -m server.app`."""
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

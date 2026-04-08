"""
Document Triage Environment — OpenEnv-compatible client.

Usage:
    from document_triage import DocumentTriageEnv, DocumentTriageAction

    async with DocumentTriageEnv(base_url="https://your-space.hf.space") as env:
        result = await env.reset()
        result = await env.step(DocumentTriageAction(action_type=0, parameter="invoice"))
"""

from openenv.core.env_client import EnvClient

from .models import DocumentTriageAction, DocumentTriageObservation


class DocumentTriageClient(EnvClient):
    """Client for the Document Triage Environment.

    Connects to a running Document Triage server (HF Space or local Docker)
    and provides async reset()/step() interface.
    """

    pass  # EnvClient provides all needed functionality

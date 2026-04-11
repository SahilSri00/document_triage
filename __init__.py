"""
Document Triage Environment — OpenEnv Package.

Exports the core Gymnasium environment for direct use.
For OpenEnv server models, see server/models.py.
"""

from src.environment import DocumentTriageEnv

__all__ = [
    "DocumentTriageEnv",
]

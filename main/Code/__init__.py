"""Compatibility package exposing the FastAPI app from :mod:`main`.

Keeps older imports (``from main.Code import api``) working even after moving
the backend entrypoint to the repository root.
"""

from .main import api

__all__ = ["api"]


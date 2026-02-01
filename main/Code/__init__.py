"""
Python package initializer for the frontend/backend bundle.

We expose the FastAPI `api` object defined in `server.py` so tooling such as
Uvicorn or Vercel can import `main.Code` (or simply `main`) and locate the ASGI
application without tightly coupling to file paths.
"""

from .server import api

__all__ = ["api"]

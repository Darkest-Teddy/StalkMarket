# Expose the FastAPI application at package level for uvicorn/vercel
from .Code import api  # noqa: F401

__all__ = ["api"]

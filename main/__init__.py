# Expose the FastAPI application at package level for uvicorn
from .Code.main import api  # noqa: F401

__all__ = ["api"]


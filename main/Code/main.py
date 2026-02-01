"""FastAPI entrypoint exposed as ``api`` for uvicorn."""

from server import api

__all__ = ["api"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:api", host="0.0.0.0", port=8000, reload=True)

"""Compatibility shim for the FastAPI backend.

The real backend entrypoint now lives at the repository root (``../main.py``),
but many workflows (and documentation) still import ``main`` from inside
``main/Code``.  This shim loads the root module and re-exports its ``api``
object so commands like ``uvicorn main:api`` continue to work from this
directory.
"""

from __future__ import annotations

from importlib import util
from pathlib import Path
import sys

ROOT_MODULE_NAME = "_stalkmarket_backend_main"
ROOT_MAIN_PATH = Path(__file__).resolve().parents[2] / "main.py"


def _load_root_module():
    if ROOT_MODULE_NAME in sys.modules:
        return sys.modules[ROOT_MODULE_NAME]
    spec = util.spec_from_file_location(ROOT_MODULE_NAME, ROOT_MAIN_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load backend entrypoint at {ROOT_MAIN_PATH}")
    module = util.module_from_spec(spec)
    sys.modules[ROOT_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_root = _load_root_module()
api = _root.api

__all__ = ["api"]

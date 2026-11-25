"""
tests/conftest.py
──────────────────
Shared pytest fixtures for the FastAPI integration test suite.

The core challenge: FastAPI's TestClient runs the app's startup handler
which calls init_db() against the production SQLite engine.  When tests
override get_db with an in-memory session, the ORM queries hit an engine
that has the tables — but the session we yield is bound to a *different*
in-memory engine that may not.

Solution: patch app.database.engine at test time so init_db() (and all
ORM operations that internally bind to that engine) use the test engine.
Each test class gets a fresh engine so there is no cross-test state.
"""

from __future__ import annotations

import app.models.db_models  # noqa: F401 — register all ORM models with Base

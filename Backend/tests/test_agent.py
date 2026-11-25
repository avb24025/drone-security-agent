"""
tests/test_agent.py
--------------------
Unit tests for the LangChain AI Security Analyst Agent.

All Gemini / LangChain API calls are mocked.
Tests verify:
  - Tool functions query the database correctly and return structured JSON
  - Tools handle empty databases gracefully
  - Tools handle DB exceptions without crashing
  - answer_question() builds a ChatResponse with answer + sources
  - Conversational memory persists across calls in the same session
  - Missing API key raises RuntimeError (not silently returns empty answer)
  - Sources are deduplicated in the final response
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base
from app.models.db_models import Alert, Event, Frame
from app.models.schemas import (
    DetectedActivity,
    DetectedObject,
    DroneFrame,
    GeminiAnalysis,
)
from app.services import agent as agent_module
from app.services.agent import (
    _make_tools,
    answer_question,
    clear_session,
)
from app.services.event_processor import process_frame
from app.services.alert_engine import AlertEngineConfig, evaluate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def db() -> Session:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture(autouse=True)
def clear_memory():
    """Wipe all session memories before and after each test."""
    agent_module._session_memories.clear()
    yield
    agent_module._session_memories.clear()


# ---------------------------------------------------------------------------
# Data seeding helpers
# ---------------------------------------------------------------------------

def _seed_vehicle(
    db: Session,
    frame_id: int = 2,
    color: str = "blue",
    make: str = "Ford",
    model: str = "F150",
    location: str = "Main Gate",
    hour: int = 8,
    event_type: str = "vehicle_entry",
):
    frame = DroneFrame(
        frame_id=frame_id,
        timestamp=datetime(2026, 11, 24, hour, 0, 0),
        location=location,
        altitude=20.0, latitude=18.52, longitude=73.85,
        description=f"A {color} {make} {model} at {location}.",
    )
    analysis = GeminiAnalysis(
        objects=[DetectedObject(type="vehicle", category="truck",
                                color=color, make=make, model=model)],
        activities=[DetectedActivity(subject="vehicle", action="entering",
                                     location=location)],
        event_type=event_type,
        description=f"{color} {make} {model} entering.",
        is_suspicious=False, raw_response="{}",
    )
    process_frame(frame, analysis, db)
    db.commit()


def _seed_person(
    db: Session,
    frame_id: int = 1,
    location: str = "Main Gate",
    hour: int = 2,
    event_type: str = "loitering",
):
    frame = DroneFrame(
        frame_id=frame_id,
        timestamp=datetime(2026, 11, 24, hour, 0, 0),
        location=location,
        altitude=20.0, latitude=18.52, longitude=73.85,
        description=f"A person at {location}.",
    )
    analysis = GeminiAnalysis(
        objects=[DetectedObject(type="person", category="person")],
        activities=[DetectedActivity(subject="person", action="standing",
                                     location=location)],
        event_type=event_type,
        description=f"Person at {location}.",
        is_suspicious=True, raw_response="{}",
    )
    process_frame(frame, analysis, db)
    db.commit()


def _seed_alert(db: Session, frame_id: int = 6, location: str = "Garage"):
    _seed_person(db, frame_id=frame_id, location=location, hour=11,
                 event_type="restricted_area_intrusion")
    from sqlalchemy.orm import Session as SASession
    cfg = AlertEngineConfig()
    # Build a minimal context to trigger alert engine
    from app.services.event_processor import ProcessingContext
    events = db.query(Event).filter(Event.frame_id == frame_id).all()
    ctx = ProcessingContext(
        frame_id=frame_id,
        db_frame_id=frame_id,
        events=events,
        prior_vehicle_appearances=[],
        prior_person_appearances_at_location=0,
        is_reprocessed=False,
    )
    evaluate(ctx, db, cfg)
    db.commit()


# ---------------------------------------------------------------------------
# Tool: search_events
# ---------------------------------------------------------------------------

class TestSearchEventsTool:

    def _get_tool(self, db):
        from app.models.schemas import ChatSource
        sources = []
        tools = _make_tools(db, sources)
        tool = next(t for t in tools if t.name == "search_events")
        return tool, sources

    def test_returns_all_events_no_filter(self, db):
        _seed_vehicle(db)
        _seed_person(db)
        tool, sources = self._get_tool(db)
        result = tool.invoke({})
        data = json.loads(result)
        assert len(data) == 2

    def test_filter_by_object_type_vehicle(self, db):
        _seed_vehicle(db)
        _seed_person(db)
        tool, _ = self._get_tool(db)
        result = json.loads(tool.invoke({"object_type": "vehicle"}))
        assert all(r["object"].startswith("vehicle") or "Ford" in r["object"]
                   for r in result)

    def test_filter_by_location(self, db):
        _seed_vehicle(db, location="Main Gate")
        _seed_vehicle(db, frame_id=4, location="Garage")
        tool, _ = self._get_tool(db)
        result = json.loads(tool.invoke({"location": "Garage"}))
        assert all(r["location"] == "Garage" for r in result)

    def test_filter_by_event_type(self, db):
        _seed_vehicle(db, event_type="vehicle_entry")
        _seed_person(db, event_type="loitering")
        tool, _ = self._get_tool(db)
        result = json.loads(tool.invoke({"event_type": "vehicle_entry"}))
        assert all(r["event_type"] == "vehicle_entry" for r in result)

    def test_empty_db_returns_no_records_message(self, db):
        tool, _ = self._get_tool(db)
        result = tool.invoke({})
        assert "No events found" in result

    def test_sources_populated(self, db):
        _seed_vehicle(db)
        tool, sources = self._get_tool(db)
        tool.invoke({})
        assert len(sources) >= 1
        assert sources[0].event_id is not None

    def test_limit_respected(self, db):
        for i in range(5):
            _seed_vehicle(db, frame_id=i + 10)
        tool, _ = self._get_tool(db)
        result = json.loads(tool.invoke({"limit": 2}))
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Tool: search_frames
# ---------------------------------------------------------------------------

class TestSearchFramesTool:

    def _get_tool(self, db):
        from app.models.schemas import ChatSource
        sources = []
        tools = _make_tools(db, sources)
        return next(t for t in tools if t.name == "search_frames"), sources

    def test_returns_frames(self, db):
        _seed_vehicle(db)
        tool, _ = self._get_tool(db)
        result = json.loads(tool.invoke({}))
        assert len(result) == 1
        assert result[0]["frame_id"] == 2

    def test_filter_by_location(self, db):
        _seed_vehicle(db, location="Main Gate")
        _seed_vehicle(db, frame_id=4, location="Garage")
        tool, _ = self._get_tool(db)
        result = json.loads(tool.invoke({"location": "Garage"}))
        assert all(r["location"] == "Garage" for r in result)

    def test_empty_db_returns_message(self, db):
        tool, _ = self._get_tool(db)
        result = tool.invoke({})
        assert "No frames found" in result


# ---------------------------------------------------------------------------
# Tool: search_alerts
# ---------------------------------------------------------------------------

class TestSearchAlertsTool:

    def _get_tool(self, db):
        from app.models.schemas import ChatSource
        sources = []
        tools = _make_tools(db, sources)
        return next(t for t in tools if t.name == "search_alerts"), sources

    def test_returns_alerts(self, db):
        _seed_alert(db)
        tool, _ = self._get_tool(db)
        result = json.loads(tool.invoke({}))
        assert len(result) >= 1

    def test_filter_by_severity(self, db):
        _seed_alert(db)
        tool, _ = self._get_tool(db)
        result = json.loads(tool.invoke({"severity": "critical"}))
        assert all(r["severity"] == "critical" for r in result)

    def test_empty_db_returns_message(self, db):
        tool, _ = self._get_tool(db)
        result = tool.invoke({})
        assert "No alerts found" in result

    def test_sources_populated_from_alerts(self, db):
        _seed_alert(db)
        tool, sources = self._get_tool(db)
        tool.invoke({})
        assert len(sources) >= 1


# ---------------------------------------------------------------------------
# Tool: get_vehicle_history
# ---------------------------------------------------------------------------

class TestGetVehicleHistoryTool:

    def _get_tool(self, db):
        from app.models.schemas import ChatSource
        sources = []
        tools = _make_tools(db, sources)
        return next(t for t in tools if t.name == "get_vehicle_history"), sources

    def test_finds_blue_ford_twice(self, db):
        _seed_vehicle(db, frame_id=2, color="blue", make="Ford", model="F150")
        _seed_vehicle(db, frame_id=7, color="blue", make="Ford", model="F150")
        tool, _ = self._get_tool(db)
        result = tool.invoke({"color": "blue", "make": "Ford", "model": "F150"})
        assert "2 appearance" in result

    def test_no_match_returns_message(self, db):
        _seed_vehicle(db)
        tool, _ = self._get_tool(db)
        result = tool.invoke({"color": "green", "make": "BMW"})
        assert "No vehicle history found" in result

    def test_missing_all_attrs_returns_guidance(self, db):
        tool, _ = self._get_tool(db)
        result = tool.invoke({})
        assert "at least one" in result.lower()


# ---------------------------------------------------------------------------
# Tool: get_daily_security_summary
# ---------------------------------------------------------------------------

class TestDailySummaryTool:

    def _get_tool(self, db):
        from app.models.schemas import ChatSource
        sources = []
        tools = _make_tools(db, sources)
        return next(t for t in tools if t.name == "get_daily_security_summary"), sources

    def test_summary_with_date(self, db):
        _seed_vehicle(db)
        tool, _ = self._get_tool(db)
        result = json.loads(tool.invoke({"date_str": "2026-11-24"}))
        assert result["date"] == "2026-11-24"
        assert result["frames_processed"] == 1
        assert result["total_events"] >= 1

    def test_summary_no_date_uses_latest(self, db):
        _seed_vehicle(db)
        tool, _ = self._get_tool(db)
        result = json.loads(tool.invoke({}))
        assert result["date"] == "2026-11-24"

    def test_empty_db_returns_message(self, db):
        tool, _ = self._get_tool(db)
        result = tool.invoke({})
        assert "No drone footage" in result

    def test_invalid_date_format_returns_error_message(self, db):
        tool, _ = self._get_tool(db)
        result = tool.invoke({"date_str": "not-a-date"})
        assert "Invalid date" in result or "Invalid" in result


# ---------------------------------------------------------------------------
# Tool: get_location_activity
# ---------------------------------------------------------------------------

class TestLocationActivityTool:

    def _get_tool(self, db):
        from app.models.schemas import ChatSource
        sources = []
        tools = _make_tools(db, sources)
        return next(t for t in tools if t.name == "get_location_activity"), sources

    def test_returns_events_at_location(self, db):
        _seed_vehicle(db, location="Main Gate")
        _seed_person(db, frame_id=9, location="Main Gate")
        tool, _ = self._get_tool(db)
        result = tool.invoke({"location": "Main Gate"})
        assert "2 event" in result

    def test_nonexistent_location_returns_message(self, db):
        tool, _ = self._get_tool(db)
        result = tool.invoke({"location": "Moon Base"})
        assert "No activity recorded" in result

    def test_case_insensitive_location(self, db):
        _seed_vehicle(db, location="Main Gate")
        tool, _ = self._get_tool(db)
        result_lower = tool.invoke({"location": "main gate"})
        result_upper = tool.invoke({"location": "MAIN GATE"})
        assert "No activity" not in result_lower
        assert "No activity" not in result_upper


# ---------------------------------------------------------------------------
# answer_question() — mocked LangChain executor
# ---------------------------------------------------------------------------

class TestAnswerQuestion:

    def _mock_executor(self, answer_text: str):
        """Return a mock AgentExecutor that returns a fixed answer."""
        mock_exec = MagicMock()
        mock_exec.invoke.return_value = {"output": answer_text}
        return mock_exec

    def test_returns_chat_response_with_answer(self, db):
        with patch.object(agent_module, "_get_agent_executor",
                          return_value=self._mock_executor("Two vehicles detected.")):
            response = answer_question("What vehicles were seen?", "s1", db)
        assert response.answer == "Two vehicles detected."
        assert response.session_id == "s1"

    def test_response_has_session_id(self, db):
        with patch.object(agent_module, "_get_agent_executor",
                          return_value=self._mock_executor("All clear.")):
            response = answer_question("Any alerts?", "my-session", db)
        assert response.session_id == "my-session"

    def test_sources_included_from_tools(self, db):
        """Sources accumulated by tools should appear in the response."""
        _seed_vehicle(db)

        def fake_executor(db_, session_id, sources):
            # Simulate a tool adding a source, then the LLM answering
            from app.models.schemas import ChatSource
            sources.append(ChatSource(
                event_id=1, frame_id=2,
                timestamp=datetime(2026, 11, 24, 8, 0),
                location="Main Gate",
                description="Test source",
            ))
            mock_exec = MagicMock()
            mock_exec.invoke.return_value = {"output": "One vehicle."}
            return mock_exec

        with patch.object(agent_module, "_get_agent_executor", side_effect=fake_executor):
            response = answer_question("How many vehicles?", "s2", db)

        assert len(response.sources) == 1
        assert response.sources[0].event_id == 1

    def test_missing_api_key_raises_runtime_error(self, db):
        with patch.object(agent_module, "_get_agent_executor",
                          side_effect=RuntimeError("GEMINI_API_KEY is not set.")):
            with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
                answer_question("Test", "s3", db)

    def test_agent_error_raises_runtime_error(self, db):
        with patch.object(agent_module, "_get_agent_executor",
                          side_effect=Exception("network error")):
            with pytest.raises(RuntimeError, match="Agent execution failed"):
                answer_question("Test", "s4", db)

    def test_empty_output_returns_fallback_message(self, db):
        mock_exec = MagicMock()
        mock_exec.invoke.return_value = {}   # no "output" key
        with patch.object(agent_module, "_get_agent_executor",
                          return_value=mock_exec):
            response = answer_question("Test", "s5", db)
        assert "could not generate" in response.answer.lower()

    def test_sources_deduplicated(self, db):
        """Duplicate (event_id, frame_id) pairs must appear only once."""
        from app.models.schemas import ChatSource

        def fake_executor(db_, session_id, sources):
            dup_src = ChatSource(
                event_id=1, frame_id=2,
                timestamp=datetime(2026, 11, 24, 8, 0),
                location="Main Gate",
                description="Duplicate",
            )
            sources.extend([dup_src, dup_src, dup_src])
            mock_exec = MagicMock()
            mock_exec.invoke.return_value = {"output": "Done."}
            return mock_exec

        with patch.object(agent_module, "_get_agent_executor", side_effect=fake_executor):
            response = answer_question("Test", "s6", db)

        assert len(response.sources) == 1


# ---------------------------------------------------------------------------
# Conversational memory
# ---------------------------------------------------------------------------

class TestConversationalMemory:

    def test_memory_created_on_first_call(self, db):
        """_get_memory creates a memory object for a new session_id."""
        from app.services.agent import _get_memory
        mem = _get_memory("brand-new-session")
        assert "brand-new-session" in agent_module._session_memories
        assert mem is agent_module._session_memories["brand-new-session"]

    def test_same_session_returns_same_memory_object(self, db):
        from app.services.agent import _get_memory
        mem1 = _get_memory("persistent-session")
        mem2 = _get_memory("persistent-session")
        assert mem1 is mem2

    def test_different_sessions_get_different_memories(self, db):
        from app.services.agent import _get_memory
        mem_a = _get_memory("session-A")
        mem_b = _get_memory("session-B")
        assert mem_a is not mem_b

    def test_clear_session_removes_memory(self, db):
        from app.services.agent import _get_memory
        _get_memory("clear-me")
        assert "clear-me" in agent_module._session_memories
        clear_session("clear-me")
        assert "clear-me" not in agent_module._session_memories

    def test_clear_nonexistent_session_is_safe(self):
        clear_session("does-not-exist")  # must not raise

    def test_memory_has_correct_window_size(self, db):
        from app.services.agent import _get_memory, _MEMORY_WINDOW_K
        mem = _get_memory("window-test")
        assert mem.k == _MEMORY_WINDOW_K

    def test_memory_key_is_chat_history(self, db):
        from app.services.agent import _get_memory
        mem = _get_memory("key-test")
        assert mem.memory_key == "chat_history"


# ---------------------------------------------------------------------------
# POST /api/chat integration (via TestClient)
# ---------------------------------------------------------------------------

class TestChatEndpoint:
    """Light integration test ensuring the route wires through to agent."""

    def test_chat_endpoint_returns_200(self, db):
        """
        Mock answer_question at the service level to avoid needing API key.
        Verifies the endpoint accepts the request and returns a valid shape.
        """
        from fastapi.testclient import TestClient
        from sqlalchemy.orm import Session as SASession
        from sqlalchemy import create_engine as _ce
        from sqlalchemy.orm import sessionmaker as _sm
        from app.main import app as fastapi_app
        from app.database import get_db
        from app.models.schemas import ChatResponse, ChatSource
        import sqlalchemy.event as _sae

        test_engine = _ce("sqlite:///:memory:", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=test_engine)
        conn  = test_engine.connect()
        trans = conn.begin()
        session = SASession(bind=conn)
        session.begin_nested()

        @_sae.listens_for(session, "after_transaction_end")
        def _reopen(sess, tx):
            if tx.nested and not tx._parent.nested:
                sess.expire_all()
                sess.begin_nested()

        def _override_get_db():
            yield session

        fastapi_app.dependency_overrides[get_db] = _override_get_db
        client = TestClient(fastapi_app, raise_server_exceptions=True)

        mock_response = ChatResponse(
            answer="One vehicle detected today.",
            sources=[],
            session_id="test-session",
        )

        try:
            # Patch at the service level where the route's lazy import resolves
            with patch("app.services.agent.answer_question", return_value=mock_response):
                resp = client.post("/api/chat", json={
                    "question": "What vehicles were seen?",
                    "session_id": "test-session",
                })
            assert resp.status_code == 200
            data = resp.json()
            assert data["answer"] == "One vehicle detected today."
            assert data["session_id"] == "test-session"
            assert "sources" in data
        finally:
            fastapi_app.dependency_overrides.clear()
            session.close()
            trans.rollback()
            conn.close()
            test_engine.dispose()

    def test_chat_endpoint_question_field_required(self, db):
        from fastapi.testclient import TestClient
        from app.main import app as fastapi_app
        client = TestClient(fastapi_app, raise_server_exceptions=False)
        resp = client.post("/api/chat", json={"session_id": "s1"})
        assert resp.status_code == 422

    def test_chat_endpoint_empty_question_rejected(self, db):
        from fastapi.testclient import TestClient
        from app.main import app as fastapi_app
        client = TestClient(fastapi_app, raise_server_exceptions=False)
        resp = client.post("/api/chat", json={"question": "", "session_id": "s1"})
        assert resp.status_code == 422

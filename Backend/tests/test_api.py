"""
tests/test_api.py
Integration tests for all FastAPI endpoints.

Strategy
- A fresh in-memory SQLite database is created for every test class.
- Gemini is mocked at gemini_service.analyze_frame -- no API key needed.
- Tests verify HTTP status codes, response shapes, and key field values.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app as fastapi_app
from app.models.schemas import DetectedActivity, DetectedObject, GeminiAnalysis
import app.database as _db_module
import app.models.db_models  # noqa: F401 -- register ORM models with Base


# ---------------------------------------------------------------------------
# Test client factory
# ---------------------------------------------------------------------------

class _AppClient:
    """
    Test client backed by a fresh in-memory SQLite database.

    SQLAlchemy 2.0 ignores sessionmaker(bind=engine) for session routing.
    We bind the Session directly to a Connection so queries land on the
    right database.  A SAVEPOINT is kept alive between handler commits so
    data written in one request is visible to the next within the same test.
    """

    def __init__(self):
        import app.models.db_models  # noqa: F401
        from sqlalchemy import event as _sa_event
        from sqlalchemy.orm import Session as _SASession

        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(bind=engine)
        self._engine = engine

        # One persistent connection + outer transaction for the whole test
        conn  = engine.connect()
        trans = conn.begin()
        self._conn  = conn
        self._trans = trans

        # Session bound to the connection (not to the engine)
        session = _SASession(bind=conn)
        self._session = session

        # Open initial savepoint
        session.begin_nested()

        @_sa_event.listens_for(session, "after_transaction_end")
        def _reopen_sp(sess, inner_tx):
            # Re-open a savepoint after each handler commit() so data persists
            if inner_tx.nested and not inner_tx._parent.nested:
                sess.expire_all()
                sess.begin_nested()

        def _override_get_db():
            yield session

        fastapi_app.dependency_overrides[get_db] = _override_get_db
        self.client = TestClient(fastapi_app, raise_server_exceptions=True)

    def teardown(self):
        fastapi_app.dependency_overrides.clear()
        self._session.close()
        self._trans.rollback()
        self._conn.close()
        self._engine.dispose()
@pytest.fixture()
def tc():
    """Provide a fresh _AppClient for each test, with guaranteed teardown."""
    app_client = _AppClient()
    yield app_client
    app_client.teardown()


# ---------------------------------------------------------------------------
# Gemini mock helpers
# ---------------------------------------------------------------------------

def _vehicle_analysis(
    color="blue", make="Ford", model="F150",
    event_type="vehicle_entry", suspicious=False,
) -> GeminiAnalysis:
    return GeminiAnalysis(
        objects=[DetectedObject(type="vehicle", category="truck",
                                color=color, make=make, model=model)],
        activities=[DetectedActivity(subject="vehicle", action="entering",
                                     location="Main Gate")],
        event_type=event_type,
        description=f"A {color} {make} {model} entering the main gate.",
        is_suspicious=suspicious,
        raw_response="{}",
    )


def _person_analysis(
    location="Main Gate", event_type="loitering", suspicious=True,
) -> GeminiAnalysis:
    return GeminiAnalysis(
        objects=[DetectedObject(type="person", category="person")],
        activities=[DetectedActivity(subject="person", action="standing",
                                     location=location)],
        event_type=event_type,
        description="A person is standing near the gate.",
        is_suspicious=suspicious,
        raw_response="{}",
    )


def _normal_analysis() -> GeminiAnalysis:
    return GeminiAnalysis(
        objects=[DetectedObject(type="person", category="person")],
        activities=[DetectedActivity(subject="person", action="walking",
                                     location="Front Walkway")],
        event_type="normal_activity",
        description="Normal daytime activity.",
        is_suspicious=False,
        raw_response="{}",
    )


def _intrusion_analysis() -> GeminiAnalysis:
    return GeminiAnalysis(
        objects=[DetectedObject(type="person", category="person")],
        activities=[],
        event_type="restricted_area_intrusion",
        description="Person in restricted area.",
        is_suspicious=True,
        raw_response="{}",
    )


# ---------------------------------------------------------------------------
# Standard frame payloads
# ---------------------------------------------------------------------------

FRAME_BLUE_TRUCK: Dict[str, Any] = {
    "frame_id": 2,
    "timestamp": "2026-11-24T08:15:00",
    "location": "Main Gate",
    "altitude": 22.0,
    "latitude": 18.5204,
    "longitude": 73.8567,
    "description": "A blue Ford F150 is entering the main gate.",
}

FRAME_PERSON_MIDNIGHT: Dict[str, Any] = {
    "frame_id": 1,
    "timestamp": "2026-11-24T02:00:00",
    "location": "Main Gate",
    "altitude": 20.0,
    "latitude": 18.5204,
    "longitude": 73.8567,
    "description": "A person is standing near the main gate at night.",
}

FRAME_GARAGE_INTRUSION: Dict[str, Any] = {
    "frame_id": 6,
    "timestamp": "2026-11-24T11:20:00",
    "location": "Garage",
    "altitude": 12.0,
    "latitude": 18.5200,
    "longitude": 73.8570,
    "description": "A person is in the restricted garage area.",
}

FRAME_NORMAL: Dict[str, Any] = {
    "frame_id": 3,
    "timestamp": "2026-11-24T09:30:00",
    "location": "Front Walkway",
    "altitude": 18.0,
    "latitude": 18.5205,
    "longitude": 73.8565,
    "description": "A person walking normally during daytime.",
}


# ---------------------------------------------------------------------------
# POST /api/frames/process
# ---------------------------------------------------------------------------

class TestProcessSingleFrame:

    def test_returns_201_on_success(self, tc):
        with patch("app.api.frames.gemini_service.analyze_frame",
                   return_value=_vehicle_analysis()):
            resp = tc.client.post("/api/frames/process", json=FRAME_BLUE_TRUCK)
        assert resp.status_code == 201

    def test_response_has_frame_id(self, tc):
        with patch("app.api.frames.gemini_service.analyze_frame",
                   return_value=_vehicle_analysis()):
            resp = tc.client.post("/api/frames/process", json=FRAME_BLUE_TRUCK)
        assert resp.json()["frame_id"] == 2

    def test_response_has_events_created(self, tc):
        with patch("app.api.frames.gemini_service.analyze_frame",
                   return_value=_vehicle_analysis()):
            resp = tc.client.post("/api/frames/process", json=FRAME_BLUE_TRUCK)
        assert resp.json()["events_created"] >= 1

    def test_response_has_alerts_created(self, tc):
        with patch("app.api.frames.gemini_service.analyze_frame",
                   return_value=_vehicle_analysis()):
            resp = tc.client.post("/api/frames/process", json=FRAME_BLUE_TRUCK)
        assert "alerts_created" in resp.json()

    def test_response_includes_analysis(self, tc):
        with patch("app.api.frames.gemini_service.analyze_frame",
                   return_value=_vehicle_analysis()):
            resp = tc.client.post("/api/frames/process", json=FRAME_BLUE_TRUCK)
        assert resp.json()["analysis"]["event_type"] == "vehicle_entry"

    def test_frame_stored_in_db(self, tc):
        with patch("app.api.frames.gemini_service.analyze_frame",
                   return_value=_vehicle_analysis()):
            tc.client.post("/api/frames/process", json=FRAME_BLUE_TRUCK)
        resp = tc.client.get("/api/frames/2")
        assert resp.status_code == 200
        assert resp.json()["frame"]["frame_id"] == 2

    def test_reprocessing_same_frame_is_idempotent(self, tc):
        with patch("app.api.frames.gemini_service.analyze_frame",
                   return_value=_vehicle_analysis()):
            tc.client.post("/api/frames/process", json=FRAME_BLUE_TRUCK)
            tc.client.post("/api/frames/process", json=FRAME_BLUE_TRUCK)
        frames = tc.client.get("/api/frames").json()
        assert len(frames) == 1

    def test_invalid_payload_returns_422(self, tc):
        resp = tc.client.post("/api/frames/process", json={"frame_id": "bad"})
        assert resp.status_code == 422

    def test_intrusion_generates_alert(self, tc):
        with patch("app.api.frames.gemini_service.analyze_frame",
                   return_value=_intrusion_analysis()):
            tc.client.post("/api/frames/process", json=FRAME_GARAGE_INTRUSION)
        alerts = tc.client.get("/api/alerts").json()
        assert any(a["alert_type"] == "restricted_area_intrusion" for a in alerts)


# ---------------------------------------------------------------------------
# POST /api/frames/batch
# ---------------------------------------------------------------------------

class TestBatchProcessing:

    def test_returns_201_with_stats(self, tc):
        payload = {"frames": [FRAME_BLUE_TRUCK, FRAME_NORMAL]}
        with patch("app.api.frames.gemini_service.analyze_frame",
                   side_effect=[_vehicle_analysis(), _normal_analysis()]):
            resp = tc.client.post("/api/frames/batch", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["total_frames"] == 2
        assert data["processed"] == 2
        assert data["failed"] == 0

    def test_batch_stores_all_frames(self, tc):
        payload = {"frames": [FRAME_BLUE_TRUCK, FRAME_NORMAL]}
        with patch("app.api.frames.gemini_service.analyze_frame",
                   side_effect=[_vehicle_analysis(), _normal_analysis()]):
            tc.client.post("/api/frames/batch", json=payload)
        assert len(tc.client.get("/api/frames").json()) == 2

    def test_batch_total_events(self, tc):
        payload = {"frames": [FRAME_BLUE_TRUCK, FRAME_NORMAL]}
        with patch("app.api.frames.gemini_service.analyze_frame",
                   side_effect=[_vehicle_analysis(), _normal_analysis()]):
            resp = tc.client.post("/api/frames/batch", json=payload)
        assert resp.json()["total_events"] >= 2

    def test_one_failed_frame_continues(self, tc):
        payload = {"frames": [FRAME_BLUE_TRUCK, FRAME_NORMAL]}
        with patch("app.api.frames.gemini_service.analyze_frame",
                   side_effect=[RuntimeError("API down"), _normal_analysis()]):
            resp = tc.client.post("/api/frames/batch", json=payload)
        data = resp.json()
        assert data["failed"] == 1
        assert data["processed"] == 1

    def test_empty_batch_returns_zero_counts(self, tc):
        resp = tc.client.post("/api/frames/batch", json={"frames": []})
        assert resp.status_code == 201
        assert resp.json()["total_frames"] == 0


# ---------------------------------------------------------------------------
# GET /api/frames
# ---------------------------------------------------------------------------

class TestFrameRetrieval:

    @pytest.fixture(autouse=True)
    def seed(self, tc):
        self.tc = tc
        with patch("app.api.frames.gemini_service.analyze_frame",
                   side_effect=[_vehicle_analysis(), _normal_analysis()]):
            tc.client.post("/api/frames/batch",
                           json={"frames": [FRAME_BLUE_TRUCK, FRAME_NORMAL]})

    def test_list_all_frames(self):
        resp = self.tc.client.get("/api/frames")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_filter_by_location(self):
        resp = self.tc.client.get("/api/frames?location=Main+Gate")
        assert resp.status_code == 200
        for f in resp.json():
            assert f["location"] == "Main Gate"

    def test_get_frame_by_id(self):
        resp = self.tc.client.get("/api/frames/2")
        assert resp.status_code == 200
        data = resp.json()
        assert "frame" in data and "events" in data
        assert data["frame"]["frame_id"] == 2

    def test_nonexistent_frame_404(self):
        assert self.tc.client.get("/api/frames/999").status_code == 404

    def test_simulator_frames_returns_20(self):
        resp = self.tc.client.get("/api/frames/simulator/frames")
        assert resp.status_code == 200
        assert len(resp.json()) == 20

    def test_simulator_summary_returns_20(self):
        resp = self.tc.client.get("/api/frames/simulator/summary")
        assert resp.status_code == 200
        summary = resp.json()
        assert len(summary) == 20
        assert "frame_id" in summary[0]


# ---------------------------------------------------------------------------
# GET /api/events
# ---------------------------------------------------------------------------

class TestEventRetrieval:

    @pytest.fixture(autouse=True)
    def seed(self, tc):
        self.tc = tc
        analyses = [_vehicle_analysis(), _intrusion_analysis()]
        with patch("app.api.frames.gemini_service.analyze_frame",
                   side_effect=analyses):
            tc.client.post("/api/frames/batch",
                           json={"frames": [FRAME_BLUE_TRUCK, FRAME_GARAGE_INTRUSION]})

    def test_list_all_events(self):
        assert len(self.tc.client.get("/api/events").json()) >= 2

    def test_filter_vehicle(self):
        for ev in self.tc.client.get("/api/events?object_type=vehicle").json():
            assert ev["object_type"] == "vehicle"

    def test_filter_person(self):
        for ev in self.tc.client.get("/api/events?object_type=person").json():
            assert ev["object_type"] == "person"

    def test_filter_location(self):
        for ev in self.tc.client.get("/api/events?location=Garage").json():
            assert ev["location"] == "Garage"

    def test_filter_event_type(self):
        for ev in self.tc.client.get("/api/events?event_type=vehicle_entry").json():
            assert ev["event_type"] == "vehicle_entry"

    def test_events_summary(self):
        data = self.tc.client.get("/api/events/summary").json()
        assert "total" in data
        assert "by_object_type" in data
        assert "by_event_type" in data

    def test_get_event_by_id(self):
        events = self.tc.client.get("/api/events").json()
        eid = events[0]["id"]
        resp = self.tc.client.get(f"/api/events/{eid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == eid

    def test_nonexistent_event_404(self):
        assert self.tc.client.get("/api/events/9999").status_code == 404


# ---------------------------------------------------------------------------
# GET /api/alerts + PATCH /api/alerts/{id}
# ---------------------------------------------------------------------------

class TestAlertEndpoints:

    @pytest.fixture(autouse=True)
    def seed(self, tc):
        self.tc = tc
        with patch("app.api.frames.gemini_service.analyze_frame",
                   return_value=_intrusion_analysis()):
            tc.client.post("/api/frames/process", json=FRAME_GARAGE_INTRUSION)

    def test_list_alerts(self):
        assert len(self.tc.client.get("/api/alerts").json()) >= 1

    def test_alert_has_required_fields(self):
        a = self.tc.client.get("/api/alerts").json()[0]
        for field in ["id", "alert_type", "severity", "message",
                      "timestamp", "location", "status"]:
            assert field in a

    def test_filter_severity_critical(self):
        for a in self.tc.client.get("/api/alerts?severity=critical").json():
            assert a["severity"] == "critical"

    def test_filter_status_unread(self):
        for a in self.tc.client.get("/api/alerts?status=unread").json():
            assert a["status"] == "unread"

    def test_get_alert_by_id(self):
        aid = self.tc.client.get("/api/alerts").json()[0]["id"]
        resp = self.tc.client.get(f"/api/alerts/{aid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == aid

    def test_nonexistent_alert_404(self):
        assert self.tc.client.get("/api/alerts/9999").status_code == 404

    def test_patch_to_read(self):
        aid = self.tc.client.get("/api/alerts").json()[0]["id"]
        resp = self.tc.client.patch(f"/api/alerts/{aid}", json={"status": "read"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "read"

    def test_patch_to_resolved(self):
        aid = self.tc.client.get("/api/alerts").json()[0]["id"]
        self.tc.client.patch(f"/api/alerts/{aid}", json={"status": "resolved"})
        assert self.tc.client.get(f"/api/alerts/{aid}").json()["status"] == "resolved"

    def test_patch_invalid_status_422(self):
        aid = self.tc.client.get("/api/alerts").json()[0]["id"]
        resp = self.tc.client.patch(f"/api/alerts/{aid}", json={"status": "deleted"})
        assert resp.status_code == 422

    def test_patch_nonexistent_404(self):
        assert self.tc.client.patch(
            "/api/alerts/9999", json={"status": "read"}
        ).status_code == 404

    def test_alert_stats(self):
        data = self.tc.client.get("/api/alerts/stats").json()
        for field in ["total", "by_severity", "by_status", "unread", "high_priority"]:
            assert field in data


# ---------------------------------------------------------------------------
# GET /api/dashboard/summary
# ---------------------------------------------------------------------------

class TestDashboard:

    def test_summary_empty_db(self, tc):
        data = tc.client.get("/api/dashboard/summary").json()
        assert data["total_frames"] == 0
        assert data["monitoring_status"] == "idle"

    def test_summary_after_processing(self, tc):
        with patch("app.api.frames.gemini_service.analyze_frame",
                   return_value=_intrusion_analysis()):
            tc.client.post("/api/frames/process", json=FRAME_GARAGE_INTRUSION)
        data = tc.client.get("/api/dashboard/summary").json()
        assert data["total_frames"] == 1
        assert data["total_events"] >= 1
        assert data["total_alerts"] >= 1
        assert data["monitoring_status"] == "active"

    def test_summary_has_all_fields(self, tc):
        data = tc.client.get("/api/dashboard/summary").json()
        for field in ["total_frames", "total_events", "total_alerts",
                      "high_priority_alerts", "unread_alerts", "monitoring_status"]:
            assert field in data


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:

    def test_health_200(self, tc):
        resp = tc.client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_root_200(self, tc):
        assert tc.client.get("/").status_code == 200


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_blue_truck_no_unauthorized_alert(self, tc):
        """Authorised vehicle must not trigger unauthorized_vehicle alert."""
        with patch("app.api.frames.gemini_service.analyze_frame",
                   return_value=_vehicle_analysis()):
            tc.client.post("/api/frames/process", json=FRAME_BLUE_TRUCK)
        events = tc.client.get("/api/events?object_type=vehicle").json()
        assert len(events) == 1
        assert events[0]["event_type"] == "vehicle_entry"
        unauth = tc.client.get("/api/alerts?alert_type=unauthorized_vehicle").json()
        assert len(unauth) == 0

    def test_garage_intrusion_critical_alert(self, tc):
        with patch("app.api.frames.gemini_service.analyze_frame",
                   return_value=_intrusion_analysis()):
            tc.client.post("/api/frames/process", json=FRAME_GARAGE_INTRUSION)
        alerts = tc.client.get(
            "/api/alerts?alert_type=restricted_area_intrusion"
        ).json()
        assert len(alerts) >= 1
        assert alerts[0]["severity"] == "critical"

    def test_repeated_vehicle_alert(self, tc):
        frame7 = {
            "frame_id": 7,
            "timestamp": "2026-11-24T14:00:00",
            "location": "Parking Lot",
            "altitude": 25.0,
            "latitude": 18.5209,
            "longitude": 73.8564,
            "description": "The blue Ford F150 is in the parking lot.",
        }
        with patch("app.api.frames.gemini_service.analyze_frame",
                   side_effect=[_vehicle_analysis(), _vehicle_analysis()]):
            tc.client.post("/api/frames/process", json=FRAME_BLUE_TRUCK)
            tc.client.post("/api/frames/process", json=frame7)
        alerts = tc.client.get(
            "/api/alerts?alert_type=repeated_vehicle_appearance"
        ).json()
        assert len(alerts) >= 1

    def test_loitering_alert_on_second_midnight_sighting(self, tc):
        frame9 = {
            "frame_id": 9,
            "timestamp": "2026-11-24T02:30:00",
            "location": "Main Gate",
            "altitude": 20.0,
            "latitude": 18.5204,
            "longitude": 73.8567,
            "description": "Person still near the gate at 02:30.",
        }
        pa = _person_analysis(location="Main Gate", event_type="loitering")
        with patch("app.api.frames.gemini_service.analyze_frame",
                   side_effect=[pa, pa]):
            tc.client.post("/api/frames/process", json=FRAME_PERSON_MIDNIGHT)
            tc.client.post("/api/frames/process", json=frame9)
        alerts = tc.client.get(
            "/api/alerts?alert_type=suspicious_loitering"
        ).json()
        assert len(alerts) >= 1

    def test_normal_daytime_no_alerts(self, tc):
        with patch("app.api.frames.gemini_service.analyze_frame",
                   return_value=_normal_analysis()):
            tc.client.post("/api/frames/process", json=FRAME_NORMAL)
        assert len(tc.client.get("/api/alerts").json()) == 0

    def test_dashboard_reflects_data(self, tc):
        with patch("app.api.frames.gemini_service.analyze_frame",
                   return_value=_intrusion_analysis()):
            tc.client.post("/api/frames/process", json=FRAME_GARAGE_INTRUSION)
        summary = tc.client.get("/api/dashboard/summary").json()
        assert summary["total_frames"] == 1
        assert summary["total_alerts"] >= 1
        assert summary["high_priority_alerts"] >= 1
        assert summary["monitoring_status"] == "active"

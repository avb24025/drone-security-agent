"""
tests/test_alert_engine.py
───────────────────────────
Unit tests for the automatic security alert engine.

All tests use an in-memory SQLite database and pre-built ProcessingContexts
so no Gemini API calls or file I/O are needed.

Coverage
────────
  Rule 1 – suspicious_loitering
  Rule 2 – unauthorized_vehicle
  Rule 3 – repeated_vehicle_appearance
  Rule 4 – restricted_area_intrusion
  Deduplication (same rule + frame + event never fires twice)
  Alert persistence (rows in SQLite)
  get_alerts() filtering
  update_alert_status()
  Rule isolation (one rule failing does not block others)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from unittest.mock import patch

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
from app.services.alert_engine import (
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    AlertEngineConfig,
    evaluate,
    get_alerts,
    update_alert_status,
)
from app.services.event_processor import (
    MatchConfidence,
    ProcessingContext,
    VehicleAppearance,
    process_frame,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures
# ─────────────────────────────────────────────────────────────────────────────

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


@pytest.fixture
def cfg() -> AlertEngineConfig:
    """Standard test config matching the simulated dataset settings."""
    return AlertEngineConfig(
        restricted_hour_start=22,
        restricted_hour_end=6,
        loitering_min_frames=2,
        authorized_vehicles=["blue ford f150", "white toyota camry"],
        restricted_areas=["garage", "server room", "storage shed", "restricted zone"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Builder helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_drone_frame(
    frame_id: int = 1,
    location: str = "Main Gate",
    hour: int = 10,
    minute: int = 0,
    description: str = "Test frame.",
) -> DroneFrame:
    return DroneFrame(
        frame_id=frame_id,
        timestamp=datetime(2026, 11, 24, hour, minute, 0),
        location=location,
        altitude=20.0,
        latitude=18.5204,
        longitude=73.8567,
        description=description,
    )


def _make_analysis(
    objects: Optional[List[DetectedObject]] = None,
    activities: Optional[List[DetectedActivity]] = None,
    event_type: str = "normal_activity",
    description: str = "Test.",
    is_suspicious: bool = False,
) -> GeminiAnalysis:
    return GeminiAnalysis(
        objects=objects or [],
        activities=activities or [],
        event_type=event_type,
        description=description,
        is_suspicious=is_suspicious,
    )


def _person() -> DetectedObject:
    return DetectedObject(type="person", category="person")


def _vehicle(
    color: str = "blue",
    make: str = "Ford",
    model: str = "F150",
    category: str = "truck",
) -> DetectedObject:
    return DetectedObject(type="vehicle", category=category, color=color, make=make, model=model)


def _build_ctx(
    db: Session,
    frame_id: int = 1,
    location: str = "Main Gate",
    hour: int = 10,
    objects: Optional[List[DetectedObject]] = None,
    event_type: str = "normal_activity",
    prior_vehicles: Optional[List[VehicleAppearance]] = None,
    prior_persons: int = 0,
) -> ProcessingContext:
    """
    Build a ProcessingContext directly (without calling Gemini) by using
    event_processor.process_frame() with mocked analysis.
    """
    frame = _make_drone_frame(frame_id=frame_id, location=location, hour=hour)
    analysis = _make_analysis(objects=objects or [], event_type=event_type)
    ctx = process_frame(frame, analysis, db)
    # Override context fields that depend on prior state
    if prior_vehicles is not None:
        ctx.prior_vehicle_appearances = prior_vehicles
    ctx.prior_person_appearances_at_location = prior_persons
    db.flush()
    return ctx


def _prior_vehicle(
    frame_id: int = 2,
    color: str = "blue",
    make: str = "Ford",
    model: str = "F150",
    category: str = "truck",
    confidence: MatchConfidence = MatchConfidence.CONFIRMED,
) -> VehicleAppearance:
    return VehicleAppearance(
        event_id=frame_id * 100,
        frame_id=frame_id,
        timestamp=datetime(2026, 11, 24, 8, 0, 0),
        location="Main Gate",
        color=color,
        make=make,
        model=model,
        category=category,
        confidence=confidence,
        description="Prior vehicle sighting.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rule 1 – Suspicious Loitering
# ─────────────────────────────────────────────────────────────────────────────

class TestLoiteringRule:

    def test_fires_when_threshold_met(self, db, cfg):
        """Second restricted-hour person detection at Main Gate fires alert."""
        ctx = _build_ctx(
            db, frame_id=9, location="Main Gate", hour=2,
            objects=[_person()], event_type="loitering",
            prior_persons=1,   # 1 prior + current = 2 → meets threshold of 2
        )
        alerts = evaluate(ctx, db, cfg)
        db.commit()

        loitering = [a for a in alerts if a.alert_type == "suspicious_loitering"]
        assert len(loitering) == 1
        assert loitering[0].severity == SEVERITY_HIGH
        assert "Main Gate" in loitering[0].message
        assert loitering[0].status == "unread"

    def test_no_alert_on_first_sighting(self, db, cfg):
        """First restricted-hour person: prior_persons=0 → below threshold."""
        ctx = _build_ctx(
            db, frame_id=1, location="Main Gate", hour=2,
            objects=[_person()], event_type="loitering",
            prior_persons=0,
        )
        alerts = evaluate(ctx, db, cfg)
        loitering = [a for a in alerts if a.alert_type == "suspicious_loitering"]
        assert len(loitering) == 0

    def test_no_alert_during_daytime(self, db, cfg):
        """Person at gate at 10:00 is NOT restricted hours → no loitering alert."""
        ctx = _build_ctx(
            db, frame_id=3, location="Main Gate", hour=10,
            objects=[_person()], event_type="normal_activity",
            prior_persons=5,   # lots of prior persons, still daytime
        )
        alerts = evaluate(ctx, db, cfg)
        loitering = [a for a in alerts if a.alert_type == "suspicious_loitering"]
        assert len(loitering) == 0

    def test_no_alert_for_vehicle_at_restricted_hours(self, db, cfg):
        """Vehicles do not trigger the loitering rule."""
        ctx = _build_ctx(
            db, frame_id=2, location="Main Gate", hour=2,
            objects=[_vehicle()], event_type="vehicle_entry",
            prior_persons=5,
        )
        alerts = evaluate(ctx, db, cfg)
        loitering = [a for a in alerts if a.alert_type == "suspicious_loitering"]
        assert len(loitering) == 0

    def test_alert_message_contains_appearance_count(self, db, cfg):
        ctx = _build_ctx(
            db, frame_id=9, location="Main Gate", hour=2,
            objects=[_person()], event_type="loitering",
            prior_persons=1,
        )
        alerts = evaluate(ctx, db, cfg)
        loitering = [a for a in alerts if a.alert_type == "suspicious_loitering"]
        assert "2" in loitering[0].message   # total = prior(1) + current(1)

    def test_configurable_threshold_of_three(self, db, cfg):
        """With loitering_min_frames=3, two priors are needed before alert fires."""
        cfg.loitering_min_frames = 3

        ctx_no_alert = _build_ctx(
            db, frame_id=5, location="Main Gate", hour=2,
            objects=[_person()], event_type="loitering",
            prior_persons=1,   # only 1 prior, need 2
        )
        alerts_no = evaluate(ctx_no_alert, db, cfg)
        assert not any(a.alert_type == "suspicious_loitering" for a in alerts_no)

        ctx_alert = _build_ctx(
            db, frame_id=6, location="Main Gate", hour=2,
            objects=[_person()], event_type="loitering",
            prior_persons=2,   # 2 priors → meets threshold
        )
        alerts_yes = evaluate(ctx_alert, db, cfg)
        assert any(a.alert_type == "suspicious_loitering" for a in alerts_yes)

    def test_restricted_hours_boundary_22h(self, db, cfg):
        """22:00 is the first restricted hour → should trigger loitering check."""
        ctx = _build_ctx(
            db, frame_id=13, location="Main Gate", hour=22,
            objects=[_person()], event_type="person_detected",
            prior_persons=1,
        )
        alerts = evaluate(ctx, db, cfg)
        loitering = [a for a in alerts if a.alert_type == "suspicious_loitering"]
        assert len(loitering) == 1

    def test_hour_6_not_restricted(self, db, cfg):
        """06:00 is after the restricted window (end=6 exclusive) → no alert."""
        ctx = _build_ctx(
            db, frame_id=20, location="Main Gate", hour=6,
            objects=[_person()], event_type="person_detected",
            prior_persons=5,
        )
        alerts = evaluate(ctx, db, cfg)
        loitering = [a for a in alerts if a.alert_type == "suspicious_loitering"]
        assert len(loitering) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Rule 2 – Unauthorized Vehicle
# ─────────────────────────────────────────────────────────────────────────────

class TestUnauthorizedVehicleRule:

    def test_fires_for_explicit_unauthorized_event_type(self, db, cfg):
        ctx = _build_ctx(
            db, frame_id=5, location="Main Gate", hour=10,
            objects=[DetectedObject(type="vehicle", color="dark",
                                   make=None, model=None, category="unknown")],
            event_type="unauthorized_vehicle",
        )
        alerts = evaluate(ctx, db, cfg)
        unauth = [a for a in alerts if a.alert_type == "unauthorized_vehicle"]
        assert len(unauth) == 1
        assert unauth[0].severity == SEVERITY_HIGH

    def test_fires_for_vehicle_not_in_authorized_list(self, db, cfg):
        ctx = _build_ctx(
            db, frame_id=11, location="Side Gate", hour=16,
            objects=[DetectedObject(type="vehicle", color="silver",
                                   make=None, model=None, category="SUV")],
            event_type="vehicle_entry",
        )
        alerts = evaluate(ctx, db, cfg)
        unauth = [a for a in alerts if a.alert_type == "unauthorized_vehicle"]
        assert len(unauth) == 1

    def test_no_alert_for_authorized_blue_ford_f150(self, db, cfg):
        """Blue Ford F150 is in the authorised list → no unauthorized alert."""
        ctx = _build_ctx(
            db, frame_id=2, location="Main Gate", hour=8,
            objects=[_vehicle(color="blue", make="Ford", model="F150")],
            event_type="vehicle_entry",
        )
        alerts = evaluate(ctx, db, cfg)
        unauth = [a for a in alerts if a.alert_type == "unauthorized_vehicle"]
        assert len(unauth) == 0

    def test_no_alert_for_authorized_white_toyota_camry(self, db, cfg):
        ctx = _build_ctx(
            db, frame_id=12, location="Main Gate", hour=17,
            objects=[_vehicle(color="white", make="Toyota", model="Camry",
                              category="sedan")],
            event_type="vehicle_exit",
        )
        alerts = evaluate(ctx, db, cfg)
        unauth = [a for a in alerts if a.alert_type == "unauthorized_vehicle"]
        assert len(unauth) == 0

    def test_alert_message_contains_location(self, db, cfg):
        ctx = _build_ctx(
            db, frame_id=5, location="Side Gate", hour=10,
            objects=[DetectedObject(type="vehicle", color="silver",
                                   make=None, model=None, category="SUV")],
            event_type="unauthorized_vehicle",
        )
        alerts = evaluate(ctx, db, cfg)
        unauth = [a for a in alerts if a.alert_type == "unauthorized_vehicle"]
        assert "Side Gate" in unauth[0].message

    def test_no_alert_for_person_event(self, db, cfg):
        """The unauthorized-vehicle rule must not fire for person events."""
        ctx = _build_ctx(
            db, frame_id=1, location="Main Gate", hour=10,
            objects=[_person()], event_type="unauthorized_vehicle",
        )
        alerts = evaluate(ctx, db, cfg)
        unauth = [a for a in alerts if a.alert_type == "unauthorized_vehicle"]
        assert len(unauth) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Rule 3 – Repeated Vehicle Appearance
# ─────────────────────────────────────────────────────────────────────────────

class TestRepeatedVehicleRule:

    def test_fires_on_second_confirmed_appearance(self, db, cfg):
        ctx = _build_ctx(
            db, frame_id=7, location="Parking Lot", hour=14,
            objects=[_vehicle()], event_type="vehicle_entry",
            prior_vehicles=[_prior_vehicle(frame_id=2)],
        )
        alerts = evaluate(ctx, db, cfg)
        repeated = [a for a in alerts if a.alert_type == "repeated_vehicle_appearance"]
        assert len(repeated) == 1
        assert repeated[0].severity == SEVERITY_MEDIUM

    def test_fires_on_third_confirmed_appearance(self, db, cfg):
        ctx = _build_ctx(
            db, frame_id=14, location="Main Gate", hour=23,
            objects=[_vehicle()], event_type="vehicle_entry",
            prior_vehicles=[
                _prior_vehicle(frame_id=2),
                _prior_vehicle(frame_id=7),
            ],
        )
        alerts = evaluate(ctx, db, cfg)
        repeated = [a for a in alerts if a.alert_type == "repeated_vehicle_appearance"]
        assert len(repeated) == 1
        assert "3" in repeated[0].message   # total = 2 priors + current

    def test_no_alert_without_prior_appearances(self, db, cfg):
        ctx = _build_ctx(
            db, frame_id=2, location="Main Gate", hour=8,
            objects=[_vehicle()], event_type="vehicle_entry",
            prior_vehicles=[],
        )
        alerts = evaluate(ctx, db, cfg)
        repeated = [a for a in alerts if a.alert_type == "repeated_vehicle_appearance"]
        assert len(repeated) == 0

    def test_possible_match_does_not_fire(self, db, cfg):
        """POSSIBLE confidence (colour only) must not trigger the repeated alert."""
        ctx = _build_ctx(
            db, frame_id=7, location="Parking Lot", hour=14,
            objects=[_vehicle()], event_type="vehicle_entry",
            prior_vehicles=[_prior_vehicle(
                frame_id=2, confidence=MatchConfidence.POSSIBLE
            )],
        )
        alerts = evaluate(ctx, db, cfg)
        repeated = [a for a in alerts if a.alert_type == "repeated_vehicle_appearance"]
        assert len(repeated) == 0

    def test_alert_message_names_vehicle(self, db, cfg):
        ctx = _build_ctx(
            db, frame_id=7, location="Parking Lot", hour=14,
            objects=[_vehicle(color="blue", make="Ford", model="F150")],
            event_type="vehicle_entry",
            prior_vehicles=[_prior_vehicle(frame_id=2)],
        )
        alerts = evaluate(ctx, db, cfg)
        repeated = [a for a in alerts if a.alert_type == "repeated_vehicle_appearance"]
        msg = repeated[0].message
        assert "Blue" in msg or "Ford" in msg or "F150" in msg

    def test_no_alert_for_person_prior(self, db, cfg):
        """Prior person appearances must not trigger the vehicle-repeat rule."""
        ctx = _build_ctx(
            db, frame_id=3, location="Main Gate", hour=10,
            objects=[_person()], event_type="person_detected",
            prior_vehicles=[],   # no vehicle priors
            prior_persons=5,
        )
        alerts = evaluate(ctx, db, cfg)
        repeated = [a for a in alerts if a.alert_type == "repeated_vehicle_appearance"]
        assert len(repeated) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Rule 4 – Restricted Area Intrusion
# ─────────────────────────────────────────────────────────────────────────────

class TestRestrictedAreaRule:

    def test_fires_for_person_in_garage(self, db, cfg):
        ctx = _build_ctx(
            db, frame_id=6, location="Garage", hour=11,
            objects=[_person()], event_type="restricted_area_intrusion",
        )
        alerts = evaluate(ctx, db, cfg)
        intrusion = [a for a in alerts if a.alert_type == "restricted_area_intrusion"]
        assert len(intrusion) == 1
        assert intrusion[0].severity == SEVERITY_CRITICAL

    def test_fires_for_person_in_server_room(self, db, cfg):
        ctx = _build_ctx(
            db, frame_id=13, location="Server Room", hour=22,
            objects=[_person()], event_type="restricted_area_intrusion",
        )
        alerts = evaluate(ctx, db, cfg)
        intrusion = [a for a in alerts if a.alert_type == "restricted_area_intrusion"]
        assert len(intrusion) == 1
        assert intrusion[0].severity == SEVERITY_CRITICAL

    def test_fires_by_location_even_without_explicit_event_type(self, db, cfg):
        """Location match alone (Garage) is enough to trigger the rule."""
        ctx = _build_ctx(
            db, frame_id=6, location="Garage", hour=10,
            objects=[_vehicle(color="red", make=None, model=None)],
            event_type="normal_activity",   # no explicit intrusion type
        )
        alerts = evaluate(ctx, db, cfg)
        intrusion = [a for a in alerts if a.alert_type == "restricted_area_intrusion"]
        assert len(intrusion) == 1

    def test_no_alert_for_normal_location(self, db, cfg):
        ctx = _build_ctx(
            db, frame_id=3, location="Front Walkway", hour=10,
            objects=[_person()], event_type="normal_activity",
        )
        alerts = evaluate(ctx, db, cfg)
        intrusion = [a for a in alerts if a.alert_type == "restricted_area_intrusion"]
        assert len(intrusion) == 0

    def test_alert_message_names_subject_person(self, db, cfg):
        ctx = _build_ctx(
            db, frame_id=6, location="Garage", hour=11,
            objects=[_person()], event_type="restricted_area_intrusion",
        )
        alerts = evaluate(ctx, db, cfg)
        intrusion = [a for a in alerts if a.alert_type == "restricted_area_intrusion"]
        assert "Person" in intrusion[0].message

    def test_alert_message_names_subject_vehicle(self, db, cfg):
        ctx = _build_ctx(
            db, frame_id=4, location="Garage", hour=10,
            objects=[_vehicle(color="red", make=None, model=None)],
            event_type="restricted_area_intrusion",
        )
        alerts = evaluate(ctx, db, cfg)
        intrusion = [a for a in alerts if a.alert_type == "restricted_area_intrusion"]
        assert "Vehicle" in intrusion[0].message

    def test_fires_for_storage_shed(self, db, cfg):
        ctx = _build_ctx(
            db, frame_id=20, location="Storage Shed", hour=0,
            objects=[_person()], event_type="person_detected",
        )
        alerts = evaluate(ctx, db, cfg)
        intrusion = [a for a in alerts if a.alert_type == "restricted_area_intrusion"]
        assert len(intrusion) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────

class TestDeduplication:

    def test_same_rule_does_not_fire_twice_for_same_event(self, db, cfg):
        ctx = _build_ctx(
            db, frame_id=6, location="Garage", hour=11,
            objects=[_person()], event_type="restricted_area_intrusion",
        )
        alerts_first = evaluate(ctx, db, cfg)
        db.commit()

        # Evaluate the same context again (simulate a re-call)
        alerts_second = evaluate(ctx, db, cfg)
        db.commit()

        intrusion_first  = [a for a in alerts_first  if a.alert_type == "restricted_area_intrusion"]
        intrusion_second = [a for a in alerts_second if a.alert_type == "restricted_area_intrusion"]

        assert len(intrusion_first) == 1
        assert len(intrusion_second) == 0, "Second evaluation must be deduplicated"

    def test_total_alert_count_in_db_is_one_after_two_calls(self, db, cfg):
        ctx = _build_ctx(
            db, frame_id=6, location="Garage", hour=11,
            objects=[_person()], event_type="restricted_area_intrusion",
        )
        evaluate(ctx, db, cfg)
        db.commit()
        evaluate(ctx, db, cfg)
        db.commit()

        total = db.query(Alert).filter(
            Alert.alert_type == "restricted_area_intrusion",
            Alert.frame_id == 6,
        ).count()
        assert total == 1

    def test_different_frames_produce_separate_alerts(self, db, cfg):
        """Same rule, different frame → different dedup keys → both fire."""
        for fid in [6, 13]:
            ctx = _build_ctx(
                db, frame_id=fid, location="Server Room", hour=22,
                objects=[_person()], event_type="restricted_area_intrusion",
            )
            evaluate(ctx, db, cfg)
        db.commit()

        total = db.query(Alert).filter(
            Alert.alert_type == "restricted_area_intrusion"
        ).count()
        assert total == 2


# ─────────────────────────────────────────────────────────────────────────────
# Alert persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestAlertPersistence:

    def test_alert_stored_in_sqlite(self, db, cfg):
        ctx = _build_ctx(
            db, frame_id=6, location="Garage", hour=11,
            objects=[_person()], event_type="restricted_area_intrusion",
        )
        evaluate(ctx, db, cfg)
        db.commit()

        stored = db.query(Alert).all()
        assert len(stored) >= 1

    def test_alert_fields_populated_correctly(self, db, cfg):
        ctx = _build_ctx(
            db, frame_id=9, location="Main Gate", hour=2,
            objects=[_person()], event_type="loitering",
            prior_persons=1,
        )
        evaluate(ctx, db, cfg)
        db.commit()

        alert = db.query(Alert).filter(Alert.alert_type == "suspicious_loitering").first()
        assert alert is not None
        assert alert.frame_id == 9
        assert alert.location == "Main Gate"
        assert alert.severity == SEVERITY_HIGH
        assert alert.status == "unread"
        assert alert.dedup_key is not None
        assert alert.timestamp == datetime(2026, 11, 24, 2, 0, 0)

    def test_multiple_rules_fire_on_same_frame(self, db, cfg):
        """Frame 6: person in Garage (restricted intrusion) + first loitering check."""
        ctx = _build_ctx(
            db, frame_id=6, location="Garage", hour=2,
            objects=[_person()], event_type="restricted_area_intrusion",
            prior_persons=1,   # triggers loitering too
        )
        alerts = evaluate(ctx, db, cfg)
        db.commit()

        types = {a.alert_type for a in alerts}
        assert "restricted_area_intrusion" in types
        assert "suspicious_loitering" in types


# ─────────────────────────────────────────────────────────────────────────────
# get_alerts() query helper
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAlerts:

    def _seed(self, db: Session, cfg: AlertEngineConfig):
        # Restricted intrusion at Garage
        ctx1 = _build_ctx(db, frame_id=6, location="Garage", hour=11,
                          objects=[_person()], event_type="restricted_area_intrusion")
        evaluate(ctx1, db, cfg)
        # Loitering at Main Gate (restricted hours)
        ctx2 = _build_ctx(db, frame_id=9, location="Main Gate", hour=2,
                          objects=[_person()], event_type="loitering", prior_persons=1)
        evaluate(ctx2, db, cfg)
        # Unauthorized vehicle at Side Gate
        ctx3 = _build_ctx(db, frame_id=11, location="Side Gate", hour=16,
                          objects=[DetectedObject(type="vehicle", color="silver",
                                                  make=None, model=None, category="SUV")],
                          event_type="unauthorized_vehicle")
        evaluate(ctx3, db, cfg)
        db.commit()

    def test_get_all_alerts(self, db, cfg):
        self._seed(db, cfg)
        alerts = get_alerts(db)
        assert len(alerts) >= 3

    def test_filter_by_severity_critical(self, db, cfg):
        self._seed(db, cfg)
        alerts = get_alerts(db, severity="critical")
        assert all(a.severity == "critical" for a in alerts)
        assert len(alerts) >= 1

    def test_filter_by_severity_high(self, db, cfg):
        self._seed(db, cfg)
        alerts = get_alerts(db, severity="high")
        assert all(a.severity == "high" for a in alerts)

    def test_filter_by_status_unread(self, db, cfg):
        self._seed(db, cfg)
        alerts = get_alerts(db, status="unread")
        assert all(a.status == "unread" for a in alerts)

    def test_filter_by_location(self, db, cfg):
        self._seed(db, cfg)
        alerts = get_alerts(db, location="Garage")
        assert all(a.location == "Garage" for a in alerts)

    def test_filter_by_alert_type(self, db, cfg):
        self._seed(db, cfg)
        alerts = get_alerts(db, alert_type="unauthorized_vehicle")
        assert all(a.alert_type == "unauthorized_vehicle" for a in alerts)
        assert len(alerts) >= 1

    def test_empty_db_returns_empty_list(self, db, cfg):
        assert get_alerts(db) == []


# ─────────────────────────────────────────────────────────────────────────────
# update_alert_status()
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateAlertStatus:

    def _create_alert(self, db: Session, cfg: AlertEngineConfig) -> Alert:
        ctx = _build_ctx(
            db, frame_id=6, location="Garage", hour=11,
            objects=[_person()], event_type="restricted_area_intrusion",
        )
        alerts = evaluate(ctx, db, cfg)
        db.commit()
        return db.query(Alert).filter(Alert.alert_type == "restricted_area_intrusion").first()

    def test_mark_as_read(self, db, cfg):
        alert = self._create_alert(db, cfg)
        updated = update_alert_status(alert.id, "read", db)
        db.commit()
        assert updated.status == "read"

    def test_mark_as_resolved(self, db, cfg):
        alert = self._create_alert(db, cfg)
        updated = update_alert_status(alert.id, "resolved", db)
        db.commit()
        assert updated.status == "resolved"

    def test_returns_none_for_nonexistent_id(self, db, cfg):
        result = update_alert_status(999, "read", db)
        assert result is None

    def test_invalid_status_raises(self, db, cfg):
        alert = self._create_alert(db, cfg)
        with pytest.raises(ValueError, match="Invalid status"):
            update_alert_status(alert.id, "dismissed", db)

    def test_persistence_survives_session_query(self, db, cfg):
        """Status change must be visible in a subsequent query."""
        alert = self._create_alert(db, cfg)
        update_alert_status(alert.id, "resolved", db)
        db.commit()

        refetched = db.query(Alert).filter(Alert.id == alert.id).first()
        assert refetched.status == "resolved"


# ─────────────────────────────────────────────────────────────────────────────
# Rule isolation — one rule exception must not block others
# ─────────────────────────────────────────────────────────────────────────────

class TestRuleIsolation:

    def test_one_rule_error_does_not_block_others(self, db, cfg):
        """
        Patch the loitering rule to raise.  The other rules (unauthorized
        vehicle, restricted area) should still fire.
        """
        ctx = _build_ctx(
            db, frame_id=5, location="Garage", hour=11,
            objects=[
                DetectedObject(type="vehicle", color="silver",
                               make=None, model=None, category="SUV"),
                _person(),
            ],
            event_type="restricted_area_intrusion",
            prior_persons=1,
        )

        import app.services.alert_engine as ae
        original = ae._rule_suspicious_loitering

        def exploding_rule(*args, **kwargs):
            raise RuntimeError("Simulated rule failure")

        ae._rule_suspicious_loitering = exploding_rule
        try:
            alerts = evaluate(ctx, db, cfg)
        finally:
            ae._rule_suspicious_loitering = original

        # Restricted intrusion and unauthorized vehicle should still fire
        types = {a.alert_type for a in alerts}
        assert "restricted_area_intrusion" in types

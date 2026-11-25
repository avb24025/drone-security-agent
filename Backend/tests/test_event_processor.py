"""
tests/test_event_processor.py
──────────────────────────────
Unit tests for the event extraction and context management service.

Uses an in-memory SQLite database spun up fresh for every test class so
tests are fully isolated.  No Gemini API calls are made.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base
from app.models.db_models import Event, Frame
from app.models.schemas import (
    DetectedActivity,
    DetectedObject,
    DroneFrame,
    GeminiAnalysis,
)
from app.services.event_processor import (
    MatchConfidence,
    ProcessingContext,
    VehicleAppearance,
    get_frame_with_events,
    get_prior_person_appearances,
    get_vehicle_history,
    process_frame,
    search_events,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def db() -> Session:
    """
    Provide a fresh in-memory SQLite session for each test.
    All tables are created before the test and the connection is closed after.
    """
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


def _frame(
    frame_id: int = 1,
    location: str = "Main Gate",
    hour: int = 10,
    description: str = "Test frame.",
) -> DroneFrame:
    return DroneFrame(
        frame_id=frame_id,
        timestamp=datetime(2026, 11, 24, hour, 0, 0),
        location=location,
        altitude=20.0,
        latitude=18.5204,
        longitude=73.8567,
        description=description,
    )


def _analysis(
    objects: List[DetectedObject] | None = None,
    activities: List[DetectedActivity] | None = None,
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


def _vehicle_obj(
    color: str = "blue",
    make: str = "Ford",
    model: str = "F150",
    category: str = "truck",
) -> DetectedObject:
    return DetectedObject(
        type="vehicle",
        category=category,
        color=color,
        make=make,
        model=model,
    )


def _person_obj() -> DetectedObject:
    return DetectedObject(type="person", category="person")


# ─────────────────────────────────────────────────────────────────────────────
# Frame persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestFramePersistence:

    def test_new_frame_stored_in_db(self, db):
        ctx = process_frame(_frame(1), _analysis(), db)
        db.commit()

        stored = db.query(Frame).filter(Frame.frame_id == 1).first()
        assert stored is not None
        assert stored.location == "Main Gate"
        assert ctx.is_reprocessed is False

    def test_frame_stores_gemini_analysis_json(self, db):
        analysis = _analysis(event_type="vehicle_entry", description="Truck entering.")
        process_frame(_frame(1), analysis, db)
        db.commit()

        stored = db.query(Frame).filter(Frame.frame_id == 1).first()
        assert stored.gemini_analysis is not None
        assert stored.gemini_analysis["event_type"] == "vehicle_entry"

    def test_reprocessing_updates_not_duplicates(self, db):
        process_frame(_frame(1, location="Main Gate"), _analysis(), db)
        db.commit()

        ctx2 = process_frame(_frame(1, location="Parking Lot"), _analysis(), db)
        db.commit()

        assert ctx2.is_reprocessed is True
        count = db.query(Frame).filter(Frame.frame_id == 1).count()
        assert count == 1, "Reprocessing must not create a duplicate Frame row"

        updated = db.query(Frame).filter(Frame.frame_id == 1).first()
        assert updated.location == "Parking Lot"

    def test_reprocessing_replaces_events(self, db):
        # First pass: vehicle event
        process_frame(
            _frame(1),
            _analysis(objects=[_vehicle_obj()], event_type="vehicle_entry"),
            db,
        )
        db.commit()
        count_after_first = db.query(Event).filter(Event.frame_id == 1).count()

        # Second pass: person event (different analysis)
        process_frame(
            _frame(1),
            _analysis(objects=[_person_obj()], event_type="person_detected"),
            db,
        )
        db.commit()
        events = db.query(Event).filter(Event.frame_id == 1).all()
        # Should only have events from the second pass
        assert all(e.object_type == "person" for e in events)

    def test_multiple_different_frames_stored_independently(self, db):
        for fid in [1, 2, 3]:
            process_frame(_frame(fid), _analysis(), db)
        db.commit()

        assert db.query(Frame).count() == 3


# ─────────────────────────────────────────────────────────────────────────────
# Event creation
# ─────────────────────────────────────────────────────────────────────────────

class TestEventCreation:

    def test_one_object_produces_one_event(self, db):
        process_frame(
            _frame(1),
            _analysis(objects=[_vehicle_obj()], event_type="vehicle_entry"),
            db,
        )
        db.commit()

        events = db.query(Event).filter(Event.frame_id == 1).all()
        assert len(events) == 1
        assert events[0].object_type == "vehicle"
        assert events[0].event_type == "vehicle_entry"

    def test_two_objects_produce_two_events(self, db):
        process_frame(
            _frame(1),
            _analysis(
                objects=[_vehicle_obj(), _person_obj()],
                event_type="restricted_area_intrusion",
            ),
            db,
        )
        db.commit()

        events = db.query(Event).filter(Event.frame_id == 1).all()
        assert len(events) == 2
        types = {e.object_type for e in events}
        assert "vehicle" in types
        assert "person" in types

    def test_no_objects_produces_fallback_event(self, db):
        """Frames with no detected objects still produce one synthetic event."""
        process_frame(
            _frame(1),
            _analysis(objects=[], event_type="unknown"),
            db,
        )
        db.commit()

        events = db.query(Event).filter(Event.frame_id == 1).all()
        assert len(events) == 1
        assert events[0].object_type == "unknown"

    def test_vehicle_attributes_stored_correctly(self, db):
        process_frame(
            _frame(1),
            _analysis(objects=[_vehicle_obj(color="blue", make="Ford", model="F150")]),
            db,
        )
        db.commit()

        ev = db.query(Event).filter(Event.frame_id == 1).first()
        assert ev.object_attributes["color"] == "blue"
        assert ev.object_attributes["make"] == "Ford"
        assert ev.object_attributes["model"] == "F150"

    def test_activity_matched_to_correct_object(self, db):
        activity = DetectedActivity(subject="vehicle", action="entering", location="Main Gate")
        process_frame(
            _frame(1),
            _analysis(objects=[_vehicle_obj()], activities=[activity]),
            db,
        )
        db.commit()

        ev = db.query(Event).filter(Event.frame_id == 1, Event.object_type == "vehicle").first()
        assert ev.activity == "entering"

    def test_event_timestamp_matches_frame_timestamp(self, db):
        f = _frame(1, hour=14)
        process_frame(f, _analysis(), db)
        db.commit()

        ev = db.query(Event).filter(Event.frame_id == 1).first()
        assert ev.timestamp == datetime(2026, 11, 24, 14, 0, 0)

    def test_event_location_matches_frame_location(self, db):
        process_frame(_frame(1, location="Garage"), _analysis(), db)
        db.commit()

        ev = db.query(Event).filter(Event.frame_id == 1).first()
        assert ev.location == "Garage"

    def test_processing_context_contains_new_events(self, db):
        ctx = process_frame(
            _frame(1),
            _analysis(objects=[_vehicle_obj(), _person_obj()]),
            db,
        )
        db.commit()

        assert len(ctx.events) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle context matching
# ─────────────────────────────────────────────────────────────────────────────

class TestVehicleContextMatching:

    def test_prior_vehicle_appearance_detected(self, db):
        """Frame 2 should detect the vehicle from Frame 1 as a prior appearance."""
        # Frame 1: blue Ford F150 arrives
        process_frame(
            _frame(1, hour=8),
            _analysis(objects=[_vehicle_obj(color="blue", make="Ford", model="F150")],
                      event_type="vehicle_entry"),
            db,
        )
        db.commit()

        # Frame 7: same blue Ford F150 reappears
        ctx = process_frame(
            _frame(7, hour=14),
            _analysis(objects=[_vehicle_obj(color="blue", make="Ford", model="F150")],
                      event_type="vehicle_entry"),
            db,
        )
        db.commit()

        assert len(ctx.prior_vehicle_appearances) == 1
        assert ctx.prior_vehicle_appearances[0].frame_id == 1

    def test_match_confidence_confirmed_on_make_plus_color(self, db):
        process_frame(
            _frame(1, hour=8),
            _analysis(objects=[_vehicle_obj(color="blue", make="Ford", model="F150")]),
            db,
        )
        db.commit()

        ctx = process_frame(
            _frame(2, hour=14),
            _analysis(objects=[_vehicle_obj(color="blue", make="Ford", model="F150")]),
            db,
        )
        db.commit()

        assert ctx.prior_vehicle_appearances[0].confidence == MatchConfidence.CONFIRMED

    def test_match_confidence_possible_on_color_only(self, db):
        """Colour-only match → POSSIBLE, not CONFIRMED."""
        process_frame(
            _frame(1),
            _analysis(objects=[DetectedObject(type="vehicle", color="blue",
                                              make=None, model=None, category="sedan")]),
            db,
        )
        db.commit()

        ctx = process_frame(
            _frame(2),
            _analysis(objects=[DetectedObject(type="vehicle", color="blue",
                                              make=None, model=None, category="truck")]),
            db,
        )
        db.commit()

        # Different category means only colour matched → POSSIBLE
        assert ctx.prior_vehicle_appearances[0].confidence == MatchConfidence.POSSIBLE

    def test_different_vehicle_no_prior_match(self, db):
        """A red Toyota should not match a blue Ford."""
        process_frame(
            _frame(1),
            _analysis(objects=[_vehicle_obj(color="blue", make="Ford", model="F150")]),
            db,
        )
        db.commit()

        ctx = process_frame(
            _frame(2),
            _analysis(objects=[DetectedObject(type="vehicle", color="red",
                                              make="Toyota", model="Camry",
                                              category="sedan")]),
            db,
        )
        db.commit()

        assert ctx.prior_vehicle_appearances == []

    def test_no_self_match(self, db):
        """A frame must never match itself in prior appearances."""
        ctx = process_frame(
            _frame(1),
            _analysis(objects=[_vehicle_obj()], event_type="vehicle_entry"),
            db,
        )
        db.commit()

        # frame_id=1 is the current frame; prior search uses frame_id < 1
        assert ctx.prior_vehicle_appearances == []

    def test_third_appearance_found(self, db):
        """Blue F150 appearing in frames 2, 7, 14 — frame 14 should find 2 priors."""
        for fid, h in [(2, 8), (7, 14)]:
            process_frame(
                _frame(fid, hour=h),
                _analysis(objects=[_vehicle_obj()], event_type="vehicle_entry"),
                db,
            )
        db.commit()

        ctx = process_frame(
            _frame(14, hour=23),
            _analysis(objects=[_vehicle_obj()], event_type="vehicle_entry"),
            db,
        )
        db.commit()

        assert len(ctx.prior_vehicle_appearances) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Person loitering context
# ─────────────────────────────────────────────────────────────────────────────

class TestPersonLoiteringContext:

    def test_first_person_at_gate_no_prior(self, db):
        ctx = process_frame(
            _frame(1, location="Main Gate", hour=2),
            _analysis(objects=[_person_obj()], event_type="loitering"),
            db,
        )
        db.commit()

        assert ctx.prior_person_appearances_at_location == 0

    def test_second_person_at_gate_counts_first(self, db):
        """Second restricted-hour person detection should report 1 prior."""
        process_frame(
            _frame(1, location="Main Gate", hour=2),
            _analysis(objects=[_person_obj()], event_type="loitering"),
            db,
        )
        db.commit()

        ctx = process_frame(
            _frame(9, location="Main Gate", hour=2),
            _analysis(objects=[_person_obj()], event_type="loitering"),
            db,
        )
        db.commit()

        assert ctx.prior_person_appearances_at_location == 1

    def test_daytime_person_not_counted_as_loitering(self, db):
        """A person at the gate at 10:00 should not count toward restricted-hour total."""
        process_frame(
            _frame(3, location="Main Gate", hour=10),
            _analysis(objects=[_person_obj()], event_type="normal_activity"),
            db,
        )
        db.commit()

        ctx = process_frame(
            _frame(9, location="Main Gate", hour=2),
            _analysis(objects=[_person_obj()], event_type="loitering"),
            db,
        )
        db.commit()

        # The 10:00 frame must NOT be counted
        assert ctx.prior_person_appearances_at_location == 0

    def test_person_at_different_location_not_counted(self, db):
        process_frame(
            _frame(1, location="Parking Lot", hour=2),
            _analysis(objects=[_person_obj()], event_type="loitering"),
            db,
        )
        db.commit()

        ctx = process_frame(
            _frame(2, location="Main Gate", hour=2),
            _analysis(objects=[_person_obj()], event_type="loitering"),
            db,
        )
        db.commit()

        assert ctx.prior_person_appearances_at_location == 0


# ─────────────────────────────────────────────────────────────────────────────
# get_vehicle_history()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetVehicleHistory:

    def _seed(self, db):
        """Seed three vehicle events: 2× blue Ford F150, 1× red Toyota."""
        for fid, color, make, model in [
            (2, "blue", "Ford", "F150"),
            (7, "blue", "Ford", "F150"),
            (4, "red", "Toyota", "Camry"),
        ]:
            process_frame(
                _frame(fid),
                _analysis(objects=[_vehicle_obj(color=color, make=make, model=model)]),
                db,
            )
        db.commit()

    def test_find_blue_ford_f150_twice(self, db):
        self._seed(db)
        results = get_vehicle_history(color="blue", make="Ford", model="F150", db=db)
        assert len(results) == 2

    def test_color_only_search(self, db):
        self._seed(db)
        results = get_vehicle_history(color="blue", db=db)
        assert len(results) == 2
        assert all(r.color == "blue" for r in results)

    def test_make_only_search(self, db):
        self._seed(db)
        results = get_vehicle_history(make="Ford", db=db)
        assert len(results) == 2

    def test_no_results_for_nonexistent_vehicle(self, db):
        self._seed(db)
        results = get_vehicle_history(color="green", make="BMW", db=db)
        assert results == []

    def test_raises_without_any_attribute(self, db):
        with pytest.raises(ValueError, match="At least one"):
            get_vehicle_history(db=db)

    def test_raises_without_db(self):
        with pytest.raises(ValueError, match="db session"):
            get_vehicle_history(color="blue", db=None)

    def test_results_ordered_oldest_first(self, db):
        self._seed(db)
        results = get_vehicle_history(color="blue", make="Ford", db=db)
        assert results[0].frame_id < results[1].frame_id


# ─────────────────────────────────────────────────────────────────────────────
# get_prior_person_appearances()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetPriorPersonAppearances:

    def test_counts_restricted_hour_persons(self, db):
        for fid, h in [(1, 2), (9, 2), (15, 1)]:
            process_frame(
                _frame(fid, location="Main Gate", hour=h),
                _analysis(objects=[_person_obj()]),
                db,
            )
        db.commit()

        count = get_prior_person_appearances("Main Gate", db=db)
        assert count == 3

    def test_ignores_daytime_persons(self, db):
        for fid, h in [(3, 10), (8, 14)]:
            process_frame(
                _frame(fid, location="Main Gate", hour=h),
                _analysis(objects=[_person_obj()]),
                db,
            )
        db.commit()

        count = get_prior_person_appearances("Main Gate", db=db)
        assert count == 0

    def test_ignores_other_locations(self, db):
        process_frame(
            _frame(1, location="Parking Lot", hour=2),
            _analysis(objects=[_person_obj()]),
            db,
        )
        db.commit()

        count = get_prior_person_appearances("Main Gate", db=db)
        assert count == 0


# ─────────────────────────────────────────────────────────────────────────────
# search_events()
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchEvents:

    def _seed(self, db):
        scenarios = [
            (_frame(1, location="Main Gate", hour=2),
             _analysis(objects=[_person_obj()], event_type="loitering")),
            (_frame(2, location="Main Gate", hour=8),
             _analysis(objects=[_vehicle_obj()], event_type="vehicle_entry")),
            (_frame(4, location="Garage", hour=10),
             _analysis(objects=[DetectedObject(type="vehicle", color="red",
                                               make=None, model=None, category="sedan")],
                       event_type="suspicious_object")),
            (_frame(6, location="Garage", hour=11),
             _analysis(objects=[_person_obj()], event_type="restricted_area_intrusion")),
        ]
        for frame, analysis in scenarios:
            process_frame(frame, analysis, db)
        db.commit()

    def test_filter_by_object_type_vehicle(self, db):
        self._seed(db)
        results = search_events(db, object_type="vehicle")
        assert all(e.object_type == "vehicle" for e in results)
        assert len(results) == 2

    def test_filter_by_object_type_person(self, db):
        self._seed(db)
        results = search_events(db, object_type="person")
        assert all(e.object_type == "person" for e in results)
        assert len(results) == 2

    def test_filter_by_location(self, db):
        self._seed(db)
        results = search_events(db, location="Garage")
        assert all(e.location == "Garage" for e in results)
        assert len(results) == 2

    def test_filter_by_location_case_insensitive(self, db):
        self._seed(db)
        results_lower = search_events(db, location="garage")
        results_upper = search_events(db, location="GARAGE")
        assert len(results_lower) == len(results_upper) == 2

    def test_filter_by_event_type(self, db):
        self._seed(db)
        results = search_events(db, event_type="vehicle_entry")
        assert all(e.event_type == "vehicle_entry" for e in results)
        assert len(results) == 1

    def test_filter_by_time_range(self, db):
        self._seed(db)
        start = datetime(2026, 11, 24, 0, 0, 0)
        end = datetime(2026, 11, 24, 5, 59, 59)
        results = search_events(db, start_time=start, end_time=end)
        assert all(0 <= e.timestamp.hour < 6 for e in results)

    def test_combined_filters(self, db):
        self._seed(db)
        results = search_events(db, object_type="person", location="Garage")
        assert len(results) == 1
        assert results[0].event_type == "restricted_area_intrusion"

    def test_no_filters_returns_all(self, db):
        self._seed(db)
        results = search_events(db)
        assert len(results) == 4

    def test_empty_db_returns_empty_list(self, db):
        results = search_events(db)
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# get_frame_with_events()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetFrameWithEvents:

    def test_returns_frame_and_events(self, db):
        process_frame(
            _frame(1),
            _analysis(objects=[_vehicle_obj(), _person_obj()]),
            db,
        )
        db.commit()

        frame_row, events = get_frame_with_events(1, db)
        assert frame_row is not None
        assert frame_row.frame_id == 1
        assert len(events) == 2

    def test_returns_none_for_missing_frame(self, db):
        frame_row, events = get_frame_with_events(999, db)
        assert frame_row is None
        assert events == []

    def test_events_belong_to_correct_frame(self, db):
        for fid in [1, 2]:
            process_frame(_frame(fid), _analysis(objects=[_vehicle_obj()]), db)
        db.commit()

        _, events = get_frame_with_events(1, db)
        assert all(e.frame_id == 1 for e in events)

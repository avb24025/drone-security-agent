"""
tests/test_simulator.py
───────────────────────
Unit tests for the drone simulator data module.

These tests do not hit Gemini or the database; they verify the integrity
of the simulated dataset and the helper functions.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.simulator.drone_simulator import (
    get_all_frames,
    get_frame,
    get_frames_by_location,
    get_frames_in_time_range,
    get_scenario_summary,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetCompleteness:

    def test_returns_twenty_frames(self):
        frames = get_all_frames()
        assert len(frames) == 20, "Dataset must contain exactly 20 frames"

    def test_all_frame_ids_are_unique(self):
        frames = get_all_frames()
        ids = [f.frame_id for f in frames]
        assert len(ids) == len(set(ids)), "All frame_id values must be unique"

    def test_frame_ids_range_one_to_twenty(self):
        frames = get_all_frames()
        ids = sorted(f.frame_id for f in frames)
        assert ids == list(range(1, 21)), "frame_ids must be 1–20 inclusive"

    def test_all_frames_have_descriptions(self):
        for frame in get_all_frames():
            assert frame.description.strip(), (
                f"Frame {frame.frame_id} has an empty description"
            )

    def test_all_frames_have_locations(self):
        for frame in get_all_frames():
            assert frame.location.strip(), (
                f"Frame {frame.frame_id} has an empty location"
            )

    def test_all_frames_have_valid_timestamps(self):
        for frame in get_all_frames():
            assert isinstance(frame.timestamp, datetime), (
                f"Frame {frame.frame_id} timestamp is not a datetime"
            )

    def test_all_frames_on_expected_date(self):
        """All simulated frames are fixed to 2026-11-24 for demo consistency."""
        for frame in get_all_frames():
            assert frame.timestamp.date().isoformat() == "2026-11-24", (
                f"Frame {frame.frame_id} is not on 2026-11-24"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Required scenario coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestScenarioCoverage:

    def _descriptions(self):
        return [(f.frame_id, f.description.lower()) for f in get_all_frames()]

    def test_blue_ford_f150_appears_at_least_twice(self):
        """The same blue Ford F150 must appear in at least two separate frames."""
        matches = [
            fid for fid, d in self._descriptions()
            if "ford f150" in d or ("blue" in d and "f150" in d)
        ]
        assert len(matches) >= 2, (
            f"Expected ≥2 blue Ford F150 frames, found frames: {matches}"
        )

    def test_person_at_main_gate_during_restricted_hours(self):
        """At least one frame must show a person at Main Gate between 22:00 and 06:00."""
        restricted_gate_frames = [
            f for f in get_all_frames()
            if f.location == "Main Gate"
            and "person" in f.description.lower()
            and (f.timestamp.hour >= 22 or f.timestamp.hour < 6)
        ]
        assert len(restricted_gate_frames) >= 1, (
            "No frame with a person at Main Gate during restricted hours found"
        )

    def test_at_least_two_loitering_candidate_frames(self):
        """
        The loitering rule requires ≥2 frames so the dataset must supply at
        least 2 person-at-gate frames during restricted hours.
        """
        candidates = [
            f for f in get_all_frames()
            if f.location == "Main Gate"
            and "person" in f.description.lower()
            and (f.timestamp.hour >= 22 or f.timestamp.hour < 6)
        ]
        assert len(candidates) >= 2, (
            f"Need ≥2 loitering candidates at Main Gate, found {len(candidates)}"
        )

    def test_person_walking_during_daytime(self):
        daytime_walker = [
            f for f in get_all_frames()
            if "walking" in f.description.lower()
            and 7 <= f.timestamp.hour < 19
        ]
        assert len(daytime_walker) >= 1, "No daytime pedestrian frame found"

    def test_unknown_vehicle_entry_exists(self):
        unknown = [
            f for f in get_all_frames()
            if any(kw in f.description.lower() for kw in
                   ["unidentified", "unknown", "no visible license", "obscured", "no plates"])
        ]
        assert len(unknown) >= 1, "No unknown/unauthorized vehicle frame found"

    def test_restricted_area_intrusion_exists(self):
        restricted_locations = {"garage", "server room"}
        intrusions = [
            f for f in get_all_frames()
            if f.location.lower() in restricted_locations
            and "person" in f.description.lower()
        ]
        assert len(intrusions) >= 1, "No restricted-area intrusion frame found"

    def test_red_car_near_garage_exists(self):
        red_car = [
            f for f in get_all_frames()
            if "red" in f.description.lower()
            and "garage" in f.location.lower()
        ]
        assert len(red_car) >= 1, "No red car near garage frame found"

    def test_normal_activity_frames_exist(self):
        """At least two frames should represent routine, non-suspicious activity."""
        normal_keywords = ["routine", "normal", "authorised", "staff", "scheduled delivery"]
        normal = [
            f for f in get_all_frames()
            if any(kw in f.description.lower() for kw in normal_keywords)
        ]
        assert len(normal) >= 2, (
            f"Expected ≥2 normal-activity frames, found {len(normal)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# get_frame()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetFrame:

    def test_returns_correct_frame(self):
        frame = get_frame(2)
        assert frame is not None
        assert frame.frame_id == 2
        assert "ford f150" in frame.description.lower()

    def test_returns_none_for_missing_id(self):
        assert get_frame(999) is None

    def test_returns_none_for_zero(self):
        assert get_frame(0) is None

    def test_returns_deep_copy(self):
        """Mutating the returned frame must not affect subsequent calls."""
        f1 = get_frame(1)
        f1.location = "MUTATED"
        f2 = get_frame(1)
        assert f2.location == "Main Gate"


# ─────────────────────────────────────────────────────────────────────────────
# get_frames_by_location()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetFramesByLocation:

    def test_main_gate_frames(self):
        frames = get_frames_by_location("Main Gate")
        assert len(frames) >= 4, "Expected multiple Main Gate frames"
        for f in frames:
            assert f.location == "Main Gate"

    def test_case_insensitive(self):
        lower = get_frames_by_location("main gate")
        upper = get_frames_by_location("MAIN GATE")
        assert len(lower) == len(upper)

    def test_nonexistent_location_returns_empty(self):
        assert get_frames_by_location("Moon Base Alpha") == []


# ─────────────────────────────────────────────────────────────────────────────
# get_frames_in_time_range()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetFramesInTimeRange:

    def test_midnight_to_6am_range(self):
        start = datetime(2026, 11, 24, 0, 0, 0)
        end = datetime(2026, 11, 24, 5, 59, 59)
        frames = get_frames_in_time_range(start, end)
        # Frames 1 (02:00), 9 (02:30), 15 (01:00), 17 (03:00), 20 (00:30) all fall here
        assert len(frames) >= 5
        for f in frames:
            assert 0 <= f.timestamp.hour < 6

    def test_business_hours_range(self):
        start = datetime(2026, 11, 24, 8, 0, 0)
        end = datetime(2026, 11, 24, 17, 59, 59)
        frames = get_frames_in_time_range(start, end)
        assert len(frames) >= 8
        for f in frames:
            assert 8 <= f.timestamp.hour < 18

    def test_empty_range_returns_nothing(self):
        start = datetime(2025, 1, 1, 0, 0, 0)
        end = datetime(2025, 1, 1, 23, 59, 59)
        assert get_frames_in_time_range(start, end) == []

    def test_single_exact_match(self):
        ts = datetime(2026, 11, 24, 8, 15, 0)
        frames = get_frames_in_time_range(ts, ts)
        assert len(frames) == 1
        assert frames[0].frame_id == 2


# ─────────────────────────────────────────────────────────────────────────────
# get_scenario_summary()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetScenarioSummary:

    def test_returns_twenty_entries(self):
        summary = get_scenario_summary()
        assert len(summary) == 20

    def test_summary_has_required_keys(self):
        for entry in get_scenario_summary():
            assert "frame_id" in entry
            assert "timestamp" in entry
            assert "location" in entry
            assert "altitude" in entry

    def test_summary_contains_no_description(self):
        """Summary is lightweight; descriptions should not be included."""
        for entry in get_scenario_summary():
            assert "description" not in entry


# ─────────────────────────────────────────────────────────────────────────────
# Data immutability: get_all_frames() must return fresh copies every call
# ─────────────────────────────────────────────────────────────────────────────

class TestImmutability:

    def test_mutating_one_call_does_not_affect_next(self):
        frames_a = get_all_frames()
        frames_a[0].location = "MUTATED"
        frames_b = get_all_frames()
        assert frames_b[0].location != "MUTATED", (
            "get_all_frames() must return deep copies, not references to the raw data"
        )

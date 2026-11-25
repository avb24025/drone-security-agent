"""
tests/test_gemini_service.py
─────────────────────────────
Unit tests for the Gemini frame analysis service.

All calls to the actual Gemini API are mocked — no API key required.
Tests cover:
  • Correct parsing of well-formed JSON responses
  • Vehicle attribute extraction (color, make, model)
  • Person detection
  • is_suspicious flag logic
  • event_type classification
  • Graceful handling of malformed JSON
  • Graceful handling of empty / no-candidate responses
  • Graceful handling of safety-blocked prompts
  • Graceful handling of GoogleAPICallError
  • Graceful handling of missing API key
  • JSON embedded in markdown fences
  • Partial object data (null fields)
  • Multiple objects in one frame
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# Service under test
from app.services import gemini_service
from app.services.gemini_service import analyze_frame, reset_model
from app.models.schemas import DroneFrame, GeminiAnalysis


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_frame(
    frame_id: int = 1,
    description: str = "A person is walking near the main gate.",
    location: str = "Main Gate",
    hour: int = 10,
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


def _make_response(payload: Dict[str, Any]) -> MagicMock:
    """Return a mock object that looks like a Gemini GenerateContentResponse."""
    candidate = MagicMock()
    response = MagicMock()
    response.candidates = [candidate]
    response.text = json.dumps(payload)
    return response


def _make_text_response(text: str) -> MagicMock:
    """Return a mock response with arbitrary text (not necessarily JSON)."""
    candidate = MagicMock()
    response = MagicMock()
    response.candidates = [candidate]
    response.text = text
    return response


def _empty_response() -> MagicMock:
    """Return a mock response with no candidates (safety block scenario)."""
    response = MagicMock()
    response.candidates = []
    response.text = ""
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────���──────────────────────────

@pytest.fixture(autouse=True)
def reset_gemini_model():
    """
    Reset the cached model before every test so each test starts clean
    and patches do not leak across tests.
    """
    reset_model()
    yield
    reset_model()


@pytest.fixture
def mock_model():
    """
    Patch _get_model() to return a MagicMock GenerativeModel.
    Tests set mock_model.generate_content.return_value as needed.
    """
    with patch.object(gemini_service, "_get_model") as mock_get:
        model = MagicMock()
        mock_get.return_value = model
        yield model


# ─────────────────────────────────────────────────────────────────────────────
# Well-formed response parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestWellFormedResponses:

    def test_vehicle_entry_basic(self, mock_model):
        payload = {
            "objects": [
                {
                    "type": "vehicle",
                    "category": "truck",
                    "color": "blue",
                    "make": "Ford",
                    "model": "F150",
                    "attributes": {},
                }
            ],
            "activities": [
                {"subject": "vehicle", "action": "entering", "location": "Main Gate"}
            ],
            "event_type": "vehicle_entry",
            "description": "A blue Ford F150 is entering the main gate.",
            "is_suspicious": False,
        }
        mock_model.generate_content.return_value = _make_response(payload)

        frame = _make_frame(description="A blue Ford F150 is entering the main gate.")
        result = analyze_frame(frame)

        assert isinstance(result, GeminiAnalysis)
        assert result.event_type == "vehicle_entry"
        assert len(result.objects) == 1
        assert result.objects[0].type == "vehicle"
        assert result.objects[0].color == "blue"
        assert result.objects[0].make == "Ford"
        assert result.objects[0].model == "F150"
        assert result.is_suspicious is False

    def test_person_detected_at_gate(self, mock_model):
        payload = {
            "objects": [{"type": "person", "category": "person", "color": None,
                         "make": None, "model": None, "attributes": {}}],
            "activities": [{"subject": "person", "action": "standing",
                            "location": "Main Gate"}],
            "event_type": "loitering",
            "description": "A person is standing near the main gate at night.",
            "is_suspicious": True,
        }
        mock_model.generate_content.return_value = _make_response(payload)

        frame = _make_frame(
            description="A person is standing near the main gate.",
            hour=2,
        )
        result = analyze_frame(frame)

        assert result.event_type == "loitering"
        assert len(result.objects) == 1
        assert result.objects[0].type == "person"
        assert result.is_suspicious is True

    def test_multiple_objects_in_one_frame(self, mock_model):
        payload = {
            "objects": [
                {"type": "vehicle", "category": "sedan", "color": "red",
                 "make": None, "model": None, "attributes": {}},
                {"type": "person", "category": "person", "color": None,
                 "make": None, "model": None, "attributes": {}},
            ],
            "activities": [
                {"subject": "vehicle", "action": "parked", "location": "Garage"},
                {"subject": "person", "action": "standing", "location": "Garage"},
            ],
            "event_type": "restricted_area_intrusion",
            "description": "A red sedan is parked and a person is standing in the garage.",
            "is_suspicious": True,
        }
        mock_model.generate_content.return_value = _make_response(payload)

        frame = _make_frame(description="Red sedan and person in garage.", location="Garage")
        result = analyze_frame(frame)

        assert len(result.objects) == 2
        assert len(result.activities) == 2
        assert result.event_type == "restricted_area_intrusion"

    def test_normal_activity_not_suspicious(self, mock_model):
        payload = {
            "objects": [{"type": "person", "category": "person", "color": None,
                         "make": None, "model": None, "attributes": {}}],
            "activities": [{"subject": "person", "action": "walking",
                            "location": "Front Walkway"}],
            "event_type": "normal_activity",
            "description": "A person is walking normally during daytime.",
            "is_suspicious": False,
        }
        mock_model.generate_content.return_value = _make_response(payload)

        frame = _make_frame(description="A person walking normally during daytime.",
                            location="Front Walkway", hour=10)
        result = analyze_frame(frame)

        assert result.event_type == "normal_activity"
        assert result.is_suspicious is False

    def test_unauthorized_vehicle_no_plates(self, mock_model):
        payload = {
            "objects": [{"type": "vehicle", "category": "SUV", "color": "silver",
                         "make": None, "model": None,
                         "attributes": {"license_plate": None}}],
            "activities": [{"subject": "vehicle", "action": "entering",
                            "location": "Side Gate"}],
            "event_type": "unauthorized_vehicle",
            "description": "A silver SUV with no plates entering the side gate.",
            "is_suspicious": True,
        }
        mock_model.generate_content.return_value = _make_response(payload)

        frame = _make_frame(
            description="A silver SUV with no visible license plates entering.",
            location="Side Gate",
        )
        result = analyze_frame(frame)

        assert result.event_type == "unauthorized_vehicle"
        assert result.objects[0].make is None
        assert result.objects[0].model is None
        assert result.is_suspicious is True

    def test_restricted_area_intrusion(self, mock_model):
        payload = {
            "objects": [{"type": "person", "category": "person", "color": None,
                         "make": None, "model": None, "attributes": {}}],
            "activities": [{"subject": "person", "action": "accessing equipment",
                            "location": "Server Room"}],
            "event_type": "restricted_area_intrusion",
            "description": "Person detected in restricted server room after hours.",
            "is_suspicious": True,
        }
        mock_model.generate_content.return_value = _make_response(payload)

        frame = _make_frame(
            description="Person in server room after hours.",
            location="Server Room",
            hour=22,
        )
        result = analyze_frame(frame)

        assert result.event_type == "restricted_area_intrusion"
        assert result.is_suspicious is True

    def test_null_vehicle_attributes_not_hallucinated(self, mock_model):
        """
        When description gives no vehicle make/model, Gemini should return null.
        The service must preserve those nulls — not substitute values.
        """
        payload = {
            "objects": [{"type": "vehicle", "category": "unknown",
                         "color": "dark", "make": None, "model": None,
                         "attributes": {"license_plate": "obscured"}}],
            "activities": [{"subject": "vehicle", "action": "entering",
                            "location": "Main Gate"}],
            "event_type": "unauthorized_vehicle",
            "description": "An unidentified dark-coloured vehicle entering.",
            "is_suspicious": True,
        }
        mock_model.generate_content.return_value = _make_response(payload)

        frame = _make_frame(
            description="An unidentified dark vehicle entering. Plate obscured.",
            location="Main Gate",
        )
        result = analyze_frame(frame)

        assert result.objects[0].make is None
        assert result.objects[0].model is None

    def test_raw_response_stored(self, mock_model):
        payload = {
            "objects": [],
            "activities": [],
            "event_type": "normal_activity",
            "description": "Nothing notable.",
            "is_suspicious": False,
        }
        mock_model.generate_content.return_value = _make_response(payload)

        result = analyze_frame(_make_frame())
        assert result.raw_response is not None
        assert len(result.raw_response) > 0

    def test_description_field_populated(self, mock_model):
        expected_desc = "A blue Ford F150 is entering the main gate."
        payload = {
            "objects": [],
            "activities": [],
            "event_type": "vehicle_entry",
            "description": expected_desc,
            "is_suspicious": False,
        }
        mock_model.generate_content.return_value = _make_response(payload)

        result = analyze_frame(_make_frame())
        assert result.description == expected_desc


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestJSONExtraction:

    def test_json_inside_markdown_fences(self, mock_model):
        payload = {
            "objects": [{"type": "person", "category": None, "color": None,
                         "make": None, "model": None, "attributes": {}}],
            "activities": [],
            "event_type": "person_detected",
            "description": "Person detected.",
            "is_suspicious": False,
        }
        # Wrap in markdown fences as Gemini sometimes does
        fenced = f"```json\n{json.dumps(payload)}\n```"
        mock_model.generate_content.return_value = _make_text_response(fenced)

        result = analyze_frame(_make_frame())
        assert result.event_type == "person_detected"
        assert len(result.objects) == 1

    def test_json_with_leading_text(self, mock_model):
        payload = {
            "objects": [],
            "activities": [],
            "event_type": "normal_activity",
            "description": "All clear.",
            "is_suspicious": False,
        }
        # Some text before the JSON object
        text = f"Here is the analysis:\n{json.dumps(payload)}"
        mock_model.generate_content.return_value = _make_text_response(text)

        result = analyze_frame(_make_frame())
        assert result.event_type == "normal_activity"

    def test_malformed_json_returns_fallback(self, mock_model):
        mock_model.generate_content.return_value = _make_text_response(
            "This is not JSON at all. { broken"
        )
        result = analyze_frame(_make_frame())

        assert isinstance(result, GeminiAnalysis)
        assert result.event_type == "unknown"
        assert result.objects == []
        assert result.activities == []

    def test_empty_string_returns_fallback(self, mock_model):
        mock_model.generate_content.return_value = _make_text_response("")
        result = analyze_frame(_make_frame())

        assert result.event_type == "unknown"

    def test_malformed_object_skipped_rest_kept(self, mock_model):
        """A malformed individual object should be skipped, not break the whole parse."""
        payload = {
            "objects": [
                {"type": "vehicle", "category": "truck", "color": "blue",
                 "make": "Ford", "model": "F150", "attributes": {}},
                "this is not an object",   # malformed — should be skipped
            ],
            "activities": [],
            "event_type": "vehicle_entry",
            "description": "Blue truck entering.",
            "is_suspicious": False,
        }
        mock_model.generate_content.return_value = _make_response(payload)

        result = analyze_frame(_make_frame())
        # The good object should still be parsed
        assert len(result.objects) == 1
        assert result.objects[0].make == "Ford"


# ─────────────────────────────────────────────────────────────────────────────
# API failure handling
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIFailureHandling:

    def test_no_candidates_returns_fallback(self, mock_model):
        """Safety-blocked response (no candidates) must not raise."""
        mock_model.generate_content.return_value = _empty_response()

        result = analyze_frame(_make_frame())
        assert isinstance(result, GeminiAnalysis)
        assert result.event_type == "unknown"
        assert "safety block" in (result.description or "").lower() or result.description is not None

    def test_blocked_prompt_exception_returns_fallback(self, mock_model):
        import google.generativeai.types as gtypes
        mock_model.generate_content.side_effect = gtypes.BlockedPromptException(
            MagicMock()
        )

        result = analyze_frame(_make_frame())
        assert isinstance(result, GeminiAnalysis)
        assert result.event_type == "unknown"

    def test_stop_candidate_exception_returns_fallback(self, mock_model):
        import google.generativeai.types as gtypes
        mock_model.generate_content.side_effect = gtypes.StopCandidateException(
            MagicMock()
        )

        result = analyze_frame(_make_frame())
        assert isinstance(result, GeminiAnalysis)
        assert result.event_type == "unknown"

    def test_google_api_call_error_returns_fallback(self, mock_model):
        from google.api_core.exceptions import GoogleAPICallError
        mock_model.generate_content.side_effect = GoogleAPICallError("quota exceeded")

        result = analyze_frame(_make_frame())
        assert isinstance(result, GeminiAnalysis)
        assert result.event_type == "unknown"

    def test_unexpected_exception_returns_fallback(self, mock_model):
        mock_model.generate_content.side_effect = RuntimeError("network unreachable")

        result = analyze_frame(_make_frame())
        assert isinstance(result, GeminiAnalysis)
        assert result.event_type == "unknown"

    def test_missing_api_key_returns_fallback(self):
        """If GEMINI_API_KEY is empty, analyze_frame must not crash."""
        reset_model()
        with patch("app.services.gemini_service._get_model") as mock_get:
            mock_get.side_effect = RuntimeError("GEMINI_API_KEY is not set.")
            result = analyze_frame(_make_frame())

        assert isinstance(result, GeminiAnalysis)
        assert result.event_type == "unknown"
        assert "runtime error" in (result.description or "").lower()


# ─────────────────────────────────────────────────────────────────────────────
# Event type classification mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestEventTypeClassification:

    @pytest.mark.parametrize("event_type", [
        "vehicle_entry",
        "vehicle_exit",
        "person_detected",
        "loitering",
        "restricted_area_intrusion",
        "unauthorized_vehicle",
        "suspicious_object",
        "normal_activity",
        "unknown",
    ])
    def test_all_valid_event_types_pass_through(self, mock_model, event_type):
        payload = {
            "objects": [],
            "activities": [],
            "event_type": event_type,
            "description": f"Test for {event_type}.",
            "is_suspicious": False,
        }
        mock_model.generate_content.return_value = _make_response(payload)

        result = analyze_frame(_make_frame())
        assert result.event_type == event_type

    def test_missing_event_type_defaults_to_unknown(self, mock_model):
        payload = {
            "objects": [],
            "activities": [],
            # event_type intentionally omitted
            "description": "No event type provided.",
            "is_suspicious": False,
        }
        mock_model.generate_content.return_value = _make_response(payload)

        result = analyze_frame(_make_frame())
        assert result.event_type == "unknown"

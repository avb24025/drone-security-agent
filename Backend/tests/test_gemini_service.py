"""
tests/test_gemini_service.py
-----------------------------
Unit tests for the frame analysis service (Groq-backed).

All Groq API calls are mocked — no API key required.
Tests cover:
  - Correct parsing of well-formed JSON responses
  - Vehicle attribute extraction (color, make, model)
  - Person detection and is_suspicious flag
  - event_type classification
  - Graceful handling of malformed JSON
  - Graceful handling of empty responses
  - Graceful handling of API errors (rate limit, auth, connection)
  - Graceful handling of missing API key
  - JSON embedded in markdown fences
  - Partial object data (null fields)
  - Multiple objects in one frame
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from app.services import gemini_service
from app.services.gemini_service import analyze_frame, reset_model
from app.models.schemas import DroneFrame, GeminiAnalysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _make_groq_response(payload: Dict[str, Any]) -> MagicMock:
    """Return a mock that looks like a Groq ChatCompletion response."""
    message = MagicMock()
    message.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_text_groq_response(text: str) -> MagicMock:
    """Return a mock Groq response with arbitrary text content."""
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _empty_groq_response() -> MagicMock:
    """Return a mock Groq response with empty content."""
    message = MagicMock()
    message.content = ""
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_groq_client():
    """Reset the cached Groq client before and after every test."""
    reset_model()
    yield
    reset_model()


@pytest.fixture
def mock_client():
    """
    Patch _get_client() to return a MagicMock Groq client.
    Tests set mock_client.chat.completions.create.return_value as needed.
    """
    with patch.object(gemini_service, "_get_client") as mock_get:
        client = MagicMock()
        mock_get.return_value = client
        yield client


# ---------------------------------------------------------------------------
# Well-formed response parsing
# ---------------------------------------------------------------------------

class TestWellFormedResponses:

    def test_vehicle_entry_basic(self, mock_client):
        payload = {
            "objects": [{"type": "vehicle", "category": "truck",
                         "color": "blue", "make": "Ford", "model": "F150",
                         "attributes": {}}],
            "activities": [{"subject": "vehicle", "action": "entering",
                            "location": "Main Gate"}],
            "event_type": "vehicle_entry",
            "description": "A blue Ford F150 is entering the main gate.",
            "is_suspicious": False,
        }
        mock_client.chat.completions.create.return_value = _make_groq_response(payload)

        result = analyze_frame(_make_frame(description="A blue Ford F150 entering."))

        assert isinstance(result, GeminiAnalysis)
        assert result.event_type == "vehicle_entry"
        assert len(result.objects) == 1
        assert result.objects[0].color == "blue"
        assert result.objects[0].make == "Ford"
        assert result.objects[0].model == "F150"
        assert result.is_suspicious is False

    def test_person_detected_at_gate(self, mock_client):
        payload = {
            "objects": [{"type": "person", "category": "person",
                         "color": None, "make": None, "model": None, "attributes": {}}],
            "activities": [{"subject": "person", "action": "standing",
                            "location": "Main Gate"}],
            "event_type": "loitering",
            "description": "A person standing near the gate at night.",
            "is_suspicious": True,
        }
        mock_client.chat.completions.create.return_value = _make_groq_response(payload)

        result = analyze_frame(_make_frame(hour=2))

        assert result.event_type == "loitering"
        assert result.objects[0].type == "person"
        assert result.is_suspicious is True

    def test_multiple_objects_in_one_frame(self, mock_client):
        payload = {
            "objects": [
                {"type": "vehicle", "category": "sedan", "color": "red",
                 "make": None, "model": None, "attributes": {}},
                {"type": "person", "category": "person", "color": None,
                 "make": None, "model": None, "attributes": {}},
            ],
            "activities": [],
            "event_type": "restricted_area_intrusion",
            "description": "Red sedan and person in garage.",
            "is_suspicious": True,
        }
        mock_client.chat.completions.create.return_value = _make_groq_response(payload)

        result = analyze_frame(_make_frame(location="Garage"))

        assert len(result.objects) == 2
        assert result.event_type == "restricted_area_intrusion"

    def test_normal_activity_not_suspicious(self, mock_client):
        payload = {
            "objects": [{"type": "person", "category": "person",
                         "color": None, "make": None, "model": None, "attributes": {}}],
            "activities": [],
            "event_type": "normal_activity",
            "description": "Person walking normally.",
            "is_suspicious": False,
        }
        mock_client.chat.completions.create.return_value = _make_groq_response(payload)

        result = analyze_frame(_make_frame(hour=10))

        assert result.event_type == "normal_activity"
        assert result.is_suspicious is False

    def test_unauthorized_vehicle_no_plates(self, mock_client):
        payload = {
            "objects": [{"type": "vehicle", "category": "SUV", "color": "silver",
                         "make": None, "model": None,
                         "attributes": {"license_plate": None}}],
            "activities": [],
            "event_type": "unauthorized_vehicle",
            "description": "Silver SUV with no plates.",
            "is_suspicious": True,
        }
        mock_client.chat.completions.create.return_value = _make_groq_response(payload)

        result = analyze_frame(_make_frame(location="Side Gate"))

        assert result.event_type == "unauthorized_vehicle"
        assert result.objects[0].make is None
        assert result.objects[0].model is None

    def test_restricted_area_intrusion(self, mock_client):
        payload = {
            "objects": [{"type": "person", "category": "person",
                         "color": None, "make": None, "model": None, "attributes": {}}],
            "activities": [],
            "event_type": "restricted_area_intrusion",
            "description": "Person in restricted server room.",
            "is_suspicious": True,
        }
        mock_client.chat.completions.create.return_value = _make_groq_response(payload)

        result = analyze_frame(_make_frame(location="Server Room", hour=22))

        assert result.event_type == "restricted_area_intrusion"
        assert result.is_suspicious is True

    def test_null_vehicle_attributes_not_hallucinated(self, mock_client):
        payload = {
            "objects": [{"type": "vehicle", "category": "unknown",
                         "color": "dark", "make": None, "model": None,
                         "attributes": {}}],
            "activities": [],
            "event_type": "unauthorized_vehicle",
            "description": "Unidentified dark vehicle.",
            "is_suspicious": True,
        }
        mock_client.chat.completions.create.return_value = _make_groq_response(payload)

        result = analyze_frame(_make_frame())

        assert result.objects[0].make is None
        assert result.objects[0].model is None

    def test_raw_response_stored(self, mock_client):
        payload = {"objects": [], "activities": [],
                   "event_type": "normal_activity",
                   "description": "Nothing.", "is_suspicious": False}
        mock_client.chat.completions.create.return_value = _make_groq_response(payload)

        result = analyze_frame(_make_frame())
        assert result.raw_response is not None
        assert len(result.raw_response) > 0

    def test_description_field_populated(self, mock_client):
        expected = "A blue Ford F150 is entering the main gate."
        payload = {"objects": [], "activities": [],
                   "event_type": "vehicle_entry",
                   "description": expected, "is_suspicious": False}
        mock_client.chat.completions.create.return_value = _make_groq_response(payload)

        result = analyze_frame(_make_frame())
        assert result.description == expected


# ---------------------------------------------------------------------------
# JSON extraction edge cases
# ---------------------------------------------------------------------------

class TestJSONExtraction:

    def test_json_inside_markdown_fences(self, mock_client):
        payload = {"objects": [{"type": "person", "category": None,
                                "color": None, "make": None, "model": None,
                                "attributes": {}}],
                   "activities": [], "event_type": "person_detected",
                   "description": "Person detected.", "is_suspicious": False}
        fenced = f"```json\n{json.dumps(payload)}\n```"
        mock_client.chat.completions.create.return_value = _make_text_groq_response(fenced)

        result = analyze_frame(_make_frame())
        assert result.event_type == "person_detected"

    def test_json_with_leading_text(self, mock_client):
        payload = {"objects": [], "activities": [],
                   "event_type": "normal_activity",
                   "description": "All clear.", "is_suspicious": False}
        text = f"Here is my analysis:\n{json.dumps(payload)}"
        mock_client.chat.completions.create.return_value = _make_text_groq_response(text)

        result = analyze_frame(_make_frame())
        assert result.event_type == "normal_activity"

    def test_malformed_json_returns_fallback(self, mock_client):
        mock_client.chat.completions.create.return_value = _make_text_groq_response(
            "This is not JSON at all. { broken"
        )
        result = analyze_frame(_make_frame())

        assert result.event_type == "unknown"
        assert result.objects == []

    def test_empty_string_returns_fallback(self, mock_client):
        mock_client.chat.completions.create.return_value = _empty_groq_response()

        result = analyze_frame(_make_frame())
        assert result.event_type == "unknown"

    def test_malformed_object_skipped_rest_kept(self, mock_client):
        payload = {
            "objects": [
                {"type": "vehicle", "category": "truck", "color": "blue",
                 "make": "Ford", "model": "F150", "attributes": {}},
                "this is not an object",
            ],
            "activities": [],
            "event_type": "vehicle_entry",
            "description": "Blue truck entering.",
            "is_suspicious": False,
        }
        mock_client.chat.completions.create.return_value = _make_groq_response(payload)

        result = analyze_frame(_make_frame())
        assert len(result.objects) == 1
        assert result.objects[0].make == "Ford"


# ---------------------------------------------------------------------------
# API failure handling
# ---------------------------------------------------------------------------

class TestAPIFailureHandling:

    def test_rate_limit_returns_fallback(self, mock_client):
        from groq import RateLimitError
        mock_client.chat.completions.create.side_effect = RateLimitError(
            "rate limit", response=MagicMock(status_code=429, headers={}), body={}
        )
        result = analyze_frame(_make_frame())
        assert result.event_type == "unknown"

    def test_auth_error_returns_fallback(self, mock_client):
        from groq import AuthenticationError
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            "invalid key", response=MagicMock(status_code=401, headers={}), body={}
        )
        result = analyze_frame(_make_frame())
        assert result.event_type == "unknown"

    def test_api_error_returns_fallback(self, mock_client):
        from groq import APIError
        import httpx
        mock_client.chat.completions.create.side_effect = APIError(
            "api error",
            request=httpx.Request("POST", "https://api.groq.com"),
            body={},
        )
        result = analyze_frame(_make_frame())
        assert result.event_type == "unknown"

    def test_unexpected_exception_returns_fallback(self, mock_client):
        mock_client.chat.completions.create.side_effect = RuntimeError("network error")
        result = analyze_frame(_make_frame())
        assert result.event_type == "unknown"

    def test_missing_api_key_returns_fallback(self):
        reset_model()
        with patch.object(gemini_service, "_get_client") as mock_get:
            mock_get.side_effect = RuntimeError("GROQ_API_KEY is not set.")
            result = analyze_frame(_make_frame())

        assert result.event_type == "unknown"
        assert "runtime error" in (result.description or "").lower()


# ---------------------------------------------------------------------------
# Event type classification
# ---------------------------------------------------------------------------

class TestEventTypeClassification:

    @pytest.mark.parametrize("event_type", [
        "vehicle_entry", "vehicle_exit", "person_detected", "loitering",
        "restricted_area_intrusion", "unauthorized_vehicle",
        "suspicious_object", "normal_activity", "unknown",
    ])
    def test_all_valid_event_types_pass_through(self, mock_client, event_type):
        payload = {"objects": [], "activities": [],
                   "event_type": event_type,
                   "description": f"Test {event_type}.", "is_suspicious": False}
        mock_client.chat.completions.create.return_value = _make_groq_response(payload)

        result = analyze_frame(_make_frame())
        assert result.event_type == event_type

    def test_missing_event_type_defaults_to_unknown(self, mock_client):
        payload = {"objects": [], "activities": [],
                   "description": "No event type.", "is_suspicious": False}
        mock_client.chat.completions.create.return_value = _make_groq_response(payload)

        result = analyze_frame(_make_frame())
        assert result.event_type == "unknown"

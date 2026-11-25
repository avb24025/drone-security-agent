"""
app/services/gemini_service.py
──────────────────────────────
Frame analysis service — powered by Groq.

Responsibilities
────────────────
1. Accept a DroneFrame and ask the Groq LLM to identify objects,
   activities, and security-relevant events in the description.
2. Return a validated GeminiAnalysis Pydantic object — never raw strings.
   (GeminiAnalysis name is kept for schema/DB compatibility.)
3. Handle every failure mode without crashing; fall back to a safe result.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from groq import Groq
from groq import APIError, APIConnectionError, RateLimitError, AuthenticationError

from app.config import get_settings
from app.models.schemas import DetectedActivity, DetectedObject, DroneFrame, GeminiAnalysis

logger = logging.getLogger(__name__)

# ── Module-level client (lazy-initialised) ────────────────────────────────────
_client: Optional[Groq] = None

# ── System instruction ────────────────────────────────────────────────────────
_SYSTEM_INSTRUCTION = """\
You are an AI security analyst processing drone surveillance footage descriptions.
Your task is to analyze frame descriptions and extract structured security intelligence.

IMPORTANT RULES:
- Base your analysis ONLY on what is explicitly described in the input.
- Do NOT invent or assume vehicle makes, models, colors, or license plates not mentioned.
- Use null for any attribute that cannot be determined from the description.
- When a vehicle's identity is uncertain, say so explicitly.
- Classify suspicious activity conservatively — only mark is_suspicious=true when
  the description contains clear indicators (restricted hours, unauthorized access,
  obscured plates, loitering, etc.).

Output ONLY a single valid JSON object. No markdown, no code fences, no extra text.
"""

_JSON_SCHEMA = """\
Return a JSON object with this exact structure:
{
  "objects": [
    {
      "type": "<person | vehicle | object | animal | unknown>",
      "category": "<null or sub-category e.g. truck, sedan, SUV>",
      "color": "<null or color string>",
      "make": "<null or manufacturer if explicitly stated>",
      "model": "<null or model if explicitly stated>",
      "attributes": {}
    }
  ],
  "activities": [
    {
      "subject": "<who or what is acting>",
      "action": "<what they are doing>",
      "location": "<null or where>"
    }
  ],
  "event_type": "<vehicle_entry|vehicle_exit|person_detected|loitering|restricted_area_intrusion|unauthorized_vehicle|suspicious_object|normal_activity|unknown>",
  "description": "<one-sentence factual summary>",
  "is_suspicious": <true or false>
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_client() -> Groq:
    global _client
    if _client is not None:
        return _client
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
    _client = Groq(api_key=settings.groq_api_key)
    logger.info("Groq client initialised (frame analysis), model: %s", settings.groq_model)
    return _client


def _build_prompt(frame: DroneFrame) -> str:
    return f"""\
Analyze the following drone surveillance frame and extract security intelligence.

FRAME METADATA:
- Frame ID   : {frame.frame_id}
- Timestamp  : {frame.timestamp.isoformat()}
- Location   : {frame.location}
- Altitude   : {frame.altitude} metres
- Coordinates: {frame.latitude}, {frame.longitude}

FRAME DESCRIPTION (simulated drone perception — treat as text, not actual video):
\"\"\"{frame.description}\"\"\"

{_JSON_SCHEMA}
"""


def _extract_json(text: str) -> Optional[str]:
    text = text.strip()
    if text.startswith("{"):
        return text
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    obj = re.search(r"\{.*\}", text, re.DOTALL)
    if obj:
        return obj.group(0)
    return None


def _parse_analysis(raw_text: str, frame_id: int) -> GeminiAnalysis:
    json_str = _extract_json(raw_text)
    if not json_str:
        logger.warning("Frame %d: no parseable JSON in response.", frame_id)
        return _fallback_analysis(raw_text, "No JSON found in response")

    try:
        data: Dict[str, Any] = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("Frame %d: JSON decode failed: %s", frame_id, exc)
        return _fallback_analysis(raw_text, f"JSON decode error: {exc}")

    objects = []
    for raw_obj in data.get("objects") or []:
        try:
            objects.append(DetectedObject(**raw_obj))
        except Exception as exc:
            logger.debug("Frame %d: skipping malformed object: %s", frame_id, exc)

    activities = []
    for raw_act in data.get("activities") or []:
        try:
            activities.append(DetectedActivity(**raw_act))
        except Exception as exc:
            logger.debug("Frame %d: skipping malformed activity: %s", frame_id, exc)

    return GeminiAnalysis(
        objects=objects,
        activities=activities,
        event_type=data.get("event_type") or "unknown",
        description=data.get("description"),
        is_suspicious=bool(data.get("is_suspicious")) if data.get("is_suspicious") is not None else None,
        raw_response=raw_text,
    )


def _fallback_analysis(raw_text: str, reason: str) -> GeminiAnalysis:
    return GeminiAnalysis(
        objects=[],
        activities=[],
        event_type="unknown",
        description=f"Analysis unavailable: {reason}",
        is_suspicious=None,
        raw_response=raw_text,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_frame(frame: DroneFrame) -> GeminiAnalysis:
    """
    Analyze a drone frame description using Groq and return structured results.
    Never raises — failures are returned as a GeminiAnalysis with event_type='unknown'.
    """
    logger.info("Analyzing frame %d at %s (%s)", frame.frame_id, frame.timestamp, frame.location)

    try:
        settings = get_settings()
        client = _get_client()

        completion = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": _SYSTEM_INSTRUCTION},
                {"role": "user",   "content": _build_prompt(frame)},
            ],
            temperature=0.6,
            max_completion_tokens=1024,
            top_p=0.95,
            stream=False,
            stop=None,
        )

        raw_text = completion.choices[0].message.content or ""
        analysis = _parse_analysis(raw_text, frame.frame_id)

        logger.info(
            "Frame %d analysis complete: event_type=%s objects=%d suspicious=%s",
            frame.frame_id, analysis.event_type, len(analysis.objects), analysis.is_suspicious,
        )
        return analysis

    except AuthenticationError as exc:
        logger.error("Frame %d: invalid Groq API key: %s", frame.frame_id, exc)
        return _fallback_analysis("", f"Authentication error: {exc}")

    except RateLimitError as exc:
        logger.warning("Frame %d: Groq rate limit hit: %s", frame.frame_id, exc)
        return _fallback_analysis("", f"Rate limit: {exc}")

    except APIConnectionError as exc:
        logger.error("Frame %d: Groq connection error: %s", frame.frame_id, exc)
        return _fallback_analysis("", f"Connection error: {exc}")

    except APIError as exc:
        logger.error("Frame %d: Groq API error: %s", frame.frame_id, exc)
        return _fallback_analysis("", f"API error: {exc}")

    except RuntimeError as exc:
        logger.error("Frame %d: runtime error (check API key?): %s", frame.frame_id, exc)
        return _fallback_analysis("", f"Runtime error: {exc}")

    except Exception as exc:
        logger.exception("Frame %d: unexpected error: %s", frame.frame_id, exc)
        return _fallback_analysis("", f"Unexpected error: {exc}")


def reset_model() -> None:
    """Reset the cached Groq client. Used in tests."""
    global _client
    _client = None

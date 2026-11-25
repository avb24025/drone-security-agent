"""
app/services/event_processor.py
────────────────────────────────
Event extraction and context management service.

Responsibilities
────────────────
1. Receive a DroneFrame + its GeminiAnalysis and produce one or more
   structured Event ORM rows that are persisted in SQLite.
2. Store the Frame itself (idempotent — re-processing the same frame_id
   updates the record rather than duplicating it).
3. Maintain cross-frame context by querying the database for prior
   appearances of the same objects, so callers can know whether a vehicle
   or person has been seen before.
4. Expose a lightweight "vehicle history" look-up used by the alert engine
   and the LangChain agent.

Design decisions
────────────────
Context without hallucination
  Matching is attribute-based (color + category + make + model) and always
  returns a MatchConfidence of POSSIBLE or CONFIRMED, never fabricating
  identity.  A single attribute match (e.g. colour only) is POSSIBLE; two
  or more specific attributes (colour + make, or colour + model) is
  CONFIRMED.  This avoids claiming "same vehicle" when only colour matches.

One frame → many events
  Each detected object in GeminiAnalysis becomes its own Event row.  When
  Gemini returns no objects but does classify an event_type, we create one
  synthetic Event so the frame is never silently dropped.

Idempotency
  Frame rows are upserted (SELECT then INSERT-or-UPDATE) so that calling
  process_frame() twice with the same frame_id is safe.  Events for a
  frame are deleted and re-created on reprocessing to keep them in sync
  with updated analysis.

Public surface
──────────────
  process_frame(frame, analysis, db)  → ProcessingContext
  get_vehicle_history(attrs, db)      → list[VehicleAppearance]
  get_prior_person_appearances(loc, db) → int
  search_events(filters, db)          → list[Event]
  get_frame_with_events(frame_id, db) → tuple[Frame | None, list[Event]]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from app.models.db_models import Event, Frame
from app.models.schemas import DetectedObject, DroneFrame, GeminiAnalysis

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Supporting data structures
# ─────────────────────────────────────────────────────────────────────────────

class MatchConfidence(str, Enum):
    NONE = "none"
    POSSIBLE = "possible"       # one attribute matches
    CONFIRMED = "confirmed"     # two or more specific attributes match


@dataclass
class VehicleAppearance:
    """One prior sighting of a vehicle that matches search attributes."""
    event_id: int
    frame_id: int
    timestamp: datetime
    location: str
    color: Optional[str]
    category: Optional[str]
    make: Optional[str]
    model: Optional[str]
    confidence: MatchConfidence
    description: Optional[str]


@dataclass
class ProcessingContext:
    """
    Everything produced when a single frame is processed end-to-end.
    Returned to the caller (FastAPI route / alert engine) so it can act
    on the results without hitting the database again.
    """
    frame_id: int
    db_frame_id: int                           # primary key of the Frame row
    events: List[Event] = field(default_factory=list)
    prior_vehicle_appearances: List[VehicleAppearance] = field(default_factory=list)
    prior_person_appearances_at_location: int = 0
    is_reprocessed: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _vehicle_signature(obj: DetectedObject) -> Dict[str, Optional[str]]:
    """Extract the matchable attributes from a detected object."""
    return {
        "color":    (obj.color or "").strip().lower() or None,
        "category": (obj.category or "").strip().lower() or None,
        "make":     (obj.make or "").strip().lower() or None,
        "model":    (obj.model or "").strip().lower() or None,
    }


def _match_confidence(sig: Dict[str, Optional[str]], attrs: Dict[str, Any]) -> MatchConfidence:
    """
    Score how well a stored event's object_attributes match a query signature.

    Scoring rules:
    - make  match → 2 points  (highly specific)
    - model match → 2 points  (highly specific)
    - color match → 1 point
    - category match → 1 point

    Result:
    - 0 points  → NONE
    - 1 point   → POSSIBLE
    - 2+ points → CONFIRMED
    """
    score = 0

    stored_color    = (attrs.get("color") or "").strip().lower() or None
    stored_category = (attrs.get("category") or "").strip().lower() or None
    stored_make     = (attrs.get("make") or "").strip().lower() or None
    stored_model    = (attrs.get("model") or "").strip().lower() or None

    if sig["make"] and stored_make and sig["make"] == stored_make:
        score += 2
    if sig["model"] and stored_model and sig["model"] == stored_model:
        score += 2
    if sig["color"] and stored_color and sig["color"] == stored_color:
        score += 1
    if sig["category"] and stored_category and sig["category"] == stored_category:
        score += 1

    if score == 0:
        return MatchConfidence.NONE
    if score == 1:
        return MatchConfidence.POSSIBLE
    return MatchConfidence.CONFIRMED


def _upsert_frame(frame: DroneFrame, analysis: GeminiAnalysis, db: Session) -> Tuple[Frame, bool]:
    """
    Insert a new Frame row or update the existing one for this frame_id.
    Returns (frame_orm, is_reprocessed).
    """
    existing: Optional[Frame] = (
        db.query(Frame).filter(Frame.frame_id == frame.frame_id).first()
    )

    # Serialise analysis — exclude raw_response to keep the JSON column lean
    analysis_dict = analysis.model_dump(exclude={"raw_response"})

    if existing:
        existing.timestamp           = frame.timestamp
        existing.location            = frame.location
        existing.altitude            = frame.altitude
        existing.latitude            = frame.latitude
        existing.longitude           = frame.longitude
        existing.original_description = frame.description
        existing.gemini_analysis     = analysis_dict
        db.flush()
        logger.debug("Frame %d updated (reprocessed).", frame.frame_id)
        return existing, True

    db_frame = Frame(
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        location=frame.location,
        altitude=frame.altitude,
        latitude=frame.latitude,
        longitude=frame.longitude,
        original_description=frame.description,
        gemini_analysis=analysis_dict,
    )
    db.add(db_frame)
    db.flush()   # populate db_frame.id without committing
    logger.debug("Frame %d inserted (new).", frame.frame_id)
    return db_frame, False


def _build_events(
    frame: DroneFrame,
    analysis: GeminiAnalysis,
    db_frame: Frame,
) -> List[Event]:
    """
    Convert GeminiAnalysis objects + activities into Event ORM instances.

    Strategy:
    - Each detected object becomes one Event.
    - The activity whose subject matches the object type is associated with it.
    - If no objects were detected, one fallback Event is created from the
      top-level event_type so the frame is never silently skipped.
    """
    events: List[Event] = []

    if not analysis.objects:
        # Fallback: one event summarising the frame
        fallback = Event(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            location=frame.location,
            object_type="unknown",
            object_attributes={},
            activity=None,
            event_type=analysis.event_type or "unknown",
            description=analysis.description,
        )
        events.append(fallback)
        return events

    for obj in analysis.objects:
        # Find the best-matching activity for this object
        activity_str: Optional[str] = None
        for act in analysis.activities:
            if obj.type.lower() in act.subject.lower() or act.subject.lower() in obj.type.lower():
                activity_str = act.action
                break
        # Fall back to the first activity if none matched
        if activity_str is None and analysis.activities:
            activity_str = analysis.activities[0].action

        attrs: Dict[str, Any] = {
            "category": obj.category,
            "color":    obj.color,
            "make":     obj.make,
            "model":    obj.model,
        }
        # Merge any extra attributes Gemini returned
        if obj.attributes:
            attrs.update(obj.attributes)

        event = Event(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            location=frame.location,
            object_type=obj.type,
            object_attributes=attrs,
            activity=activity_str,
            event_type=analysis.event_type or "unknown",
            description=analysis.description,
        )
        events.append(event)

    return events


def _find_prior_vehicle_appearances(
    objects: List[DetectedObject],
    current_frame_id: int,
    db: Session,
) -> List[VehicleAppearance]:
    """
    For every vehicle object in this frame, query the database for prior
    sightings of vehicles with matching attributes.

    Only events from *earlier* frames (frame_id < current_frame_id) are
    considered so we do not match the event we just created.
    """
    if not objects:
        return []

    appearances: List[VehicleAppearance] = []
    seen_event_ids: set[int] = set()

    vehicle_objects = [o for o in objects if o.type.lower() == "vehicle"]
    if not vehicle_objects:
        return []

    prior_vehicle_events: List[Event] = (
        db.query(Event)
        .filter(
            Event.object_type == "vehicle",
            Event.frame_id < current_frame_id,
        )
        .order_by(Event.timestamp.asc())
        .all()
    )

    for obj in vehicle_objects:
        sig = _vehicle_signature(obj)
        # Skip objects with no matchable attributes at all
        if not any(sig.values()):
            continue

        for prior in prior_vehicle_events:
            if prior.id in seen_event_ids:
                continue

            stored_attrs: Dict[str, Any] = prior.object_attributes or {}
            confidence = _match_confidence(sig, stored_attrs)

            if confidence == MatchConfidence.NONE:
                continue

            seen_event_ids.add(prior.id)
            appearances.append(
                VehicleAppearance(
                    event_id=prior.id,
                    frame_id=prior.frame_id,
                    timestamp=prior.timestamp,
                    location=prior.location,
                    color=stored_attrs.get("color"),
                    category=stored_attrs.get("category"),
                    make=stored_attrs.get("make"),
                    model=stored_attrs.get("model"),
                    confidence=confidence,
                    description=prior.description,
                )
            )

    return appearances


def _count_prior_person_appearances(
    location: str,
    current_frame_id: int,
    restricted_hour_start: int,
    restricted_hour_end: int,
    db: Session,
) -> int:
    """
    Count how many prior person events exist at this location during
    restricted hours.  Used to determine whether loitering threshold is met.
    """
    prior: List[Event] = (
        db.query(Event)
        .filter(
            Event.object_type == "person",
            Event.location == location,
            Event.frame_id < current_frame_id,
        )
        .all()
    )

    count = 0
    for ev in prior:
        h = ev.timestamp.hour
        if restricted_hour_start <= 23:   # e.g. 22 → 06
            in_restricted = (h >= restricted_hour_start) or (h < restricted_hour_end)
        else:
            in_restricted = restricted_hour_start <= h < restricted_hour_end
        if in_restricted:
            count += 1

    return count


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def process_frame(
    frame: DroneFrame,
    analysis: GeminiAnalysis,
    db: Session,
    restricted_hour_start: int = 22,
    restricted_hour_end: int = 6,
) -> ProcessingContext:
    """
    Persist a frame + its analysis and extract structured events.

    Steps:
    1. Upsert the Frame row.
    2. If reprocessing, delete existing events for this frame first.
    3. Build Event rows from GeminiAnalysis.
    4. Persist the events.
    5. Query prior context (vehicle appearances, person loitering count).
    6. Return a ProcessingContext the caller can use directly.

    This function does NOT commit — the caller owns the transaction so it
    can roll back if the alert engine subsequently fails.
    """
    logger.info("Processing frame %d (location=%s)", frame.frame_id, frame.location)

    # 1. Upsert frame
    db_frame, is_reprocessed = _upsert_frame(frame, analysis, db)

    # 2. Clear stale events on reprocessing
    if is_reprocessed:
        db.query(Event).filter(Event.frame_id == frame.frame_id).delete()
        db.flush()
        logger.debug("Cleared existing events for reprocessed frame %d.", frame.frame_id)

    # 3. Build events
    new_events = _build_events(frame, analysis, db_frame)

    # 4. Persist
    for ev in new_events:
        db.add(ev)
    db.flush()   # assign primary keys without committing
    logger.info("Frame %d: %d event(s) created.", frame.frame_id, len(new_events))

    # 5. Context — query AFTER flush so the new events have IDs but we
    #    still exclude them (frame_id < current) from "prior" searches
    prior_vehicles = _find_prior_vehicle_appearances(
        analysis.objects, frame.frame_id, db
    )

    prior_persons = 0
    has_person = any(o.type.lower() == "person" for o in analysis.objects)
    if has_person:
        prior_persons = _count_prior_person_appearances(
            frame.location,
            frame.frame_id,
            restricted_hour_start,
            restricted_hour_end,
            db,
        )

    if prior_vehicles:
        logger.info(
            "Frame %d: %d prior vehicle appearance(s) matched.",
            frame.frame_id,
            len(prior_vehicles),
        )
    if prior_persons:
        logger.info(
            "Frame %d: %d prior person appearance(s) at %s during restricted hours.",
            frame.frame_id,
            prior_persons,
            frame.location,
        )

    return ProcessingContext(
        frame_id=frame.frame_id,
        db_frame_id=db_frame.id,
        events=new_events,
        prior_vehicle_appearances=prior_vehicles,
        prior_person_appearances_at_location=prior_persons,
        is_reprocessed=is_reprocessed,
    )


def get_vehicle_history(
    color: Optional[str] = None,
    make: Optional[str] = None,
    model: Optional[str] = None,
    category: Optional[str] = None,
    db: Session = None,
) -> List[VehicleAppearance]:
    """
    Search the event history for all vehicle sightings matching the given
    attributes.  At least one attribute must be provided.

    Returns appearances ordered oldest-first with their match confidence.
    """
    if db is None:
        raise ValueError("db session is required")

    if not any([color, make, model, category]):
        raise ValueError("At least one of color, make, model, category must be provided.")

    all_vehicle_events: List[Event] = (
        db.query(Event)
        .filter(Event.object_type == "vehicle")
        .order_by(Event.timestamp.asc())
        .all()
    )

    sig: Dict[str, Optional[str]] = {
        "color":    (color or "").strip().lower() or None,
        "category": (category or "").strip().lower() or None,
        "make":     (make or "").strip().lower() or None,
        "model":    (model or "").strip().lower() or None,
    }

    results: List[VehicleAppearance] = []
    for ev in all_vehicle_events:
        attrs = ev.object_attributes or {}
        confidence = _match_confidence(sig, attrs)
        if confidence == MatchConfidence.NONE:
            continue
        results.append(
            VehicleAppearance(
                event_id=ev.id,
                frame_id=ev.frame_id,
                timestamp=ev.timestamp,
                location=ev.location,
                color=attrs.get("color"),
                category=attrs.get("category"),
                make=attrs.get("make"),
                model=attrs.get("model"),
                confidence=confidence,
                description=ev.description,
            )
        )

    return results


def get_prior_person_appearances(
    location: str,
    restricted_hour_start: int = 22,
    restricted_hour_end: int = 6,
    db: Session = None,
) -> int:
    """
    Return the total number of person events recorded at the given location
    during restricted hours.  Used by the alert engine.
    """
    if db is None:
        raise ValueError("db session is required")

    all_person_events: List[Event] = (
        db.query(Event)
        .filter(
            Event.object_type == "person",
            Event.location == location,
        )
        .all()
    )

    count = 0
    for ev in all_person_events:
        h = ev.timestamp.hour
        in_restricted = (h >= restricted_hour_start) or (h < restricted_hour_end)
        if in_restricted:
            count += 1
    return count


def search_events(
    db: Session,
    object_type: Optional[str] = None,
    location: Optional[str] = None,
    event_type: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    frame_id: Optional[int] = None,
    limit: int = 200,
    offset: int = 0,
) -> List[Event]:
    """
    Parameterised event search.  All filters are optional and combinable.
    Returns events ordered newest-first.
    """
    q = db.query(Event)

    if object_type:
        q = q.filter(Event.object_type == object_type.lower())
    if location:
        q = q.filter(func.lower(Event.location) == location.lower())
    if event_type:
        q = q.filter(Event.event_type == event_type)
    if frame_id is not None:
        q = q.filter(Event.frame_id == frame_id)
    if start_time:
        q = q.filter(Event.timestamp >= start_time)
    if end_time:
        q = q.filter(Event.timestamp <= end_time)

    return q.order_by(Event.timestamp.desc()).offset(offset).limit(limit).all()


def get_frame_with_events(
    frame_id: int,
    db: Session,
) -> Tuple[Optional[Frame], List[Event]]:
    """
    Return a Frame row and all its associated Event rows.
    Returns (None, []) if the frame does not exist.
    """
    db_frame: Optional[Frame] = (
        db.query(Frame).filter(Frame.frame_id == frame_id).first()
    )
    if not db_frame:
        return None, []

    events: List[Event] = (
        db.query(Event)
        .filter(Event.frame_id == frame_id)
        .order_by(Event.id.asc())
        .all()
    )
    return db_frame, events


def get_all_frames(
    db: Session,
    location: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 200,
    offset: int = 0,
) -> List[Frame]:
    """
    Return indexed frames with optional filters, newest-first.
    """
    q = db.query(Frame)
    if location:
        q = q.filter(func.lower(Frame.location) == location.lower())
    if start_time:
        q = q.filter(Frame.timestamp >= start_time)
    if end_time:
        q = q.filter(Frame.timestamp <= end_time)
    return q.order_by(Frame.timestamp.desc()).offset(offset).limit(limit).all()

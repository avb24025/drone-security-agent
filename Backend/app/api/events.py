"""
app/api/events.py
─────────────────
Event retrieval endpoints.

GET /api/events            – list events (filterable by type, location, time)
GET /api/events/{event_id} – single event detail
GET /api/events/summary    – per-location and per-type counts for dashboard
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import Event
from app.models.schemas import EventResponse
from app.services.event_processor import search_events

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("", response_model=List[EventResponse])
def list_events(
    object_type: Optional[str] = Query(None, description="e.g. vehicle, person"),
    location: Optional[str] = Query(None, description="Filter by location label"),
    event_type: Optional[str] = Query(None, description="e.g. vehicle_entry, loitering"),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    frame_id: Optional[int] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> List[EventResponse]:
    """
    Return events with optional filters. All filters are combinable.
    Results are ordered newest-first.
    """
    events = search_events(
        db=db,
        object_type=object_type,
        location=location,
        event_type=event_type,
        start_time=start_time,
        end_time=end_time,
        frame_id=frame_id,
        limit=limit,
        offset=offset,
    )
    return [EventResponse.model_validate(e) for e in events]


@router.get("/summary", response_model=Dict[str, Any])
def events_summary(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Return aggregate counts grouped by object_type and event_type.
    Used by the dashboard overview cards.
    """
    all_events: List[Event] = db.query(Event).all()

    by_type: Dict[str, int] = {}
    by_event_type: Dict[str, int] = {}
    by_location: Dict[str, int] = {}

    for ev in all_events:
        by_type[ev.object_type] = by_type.get(ev.object_type, 0) + 1
        by_event_type[ev.event_type] = by_event_type.get(ev.event_type, 0) + 1
        by_location[ev.location] = by_location.get(ev.location, 0) + 1

    return {
        "total": len(all_events),
        "by_object_type": by_type,
        "by_event_type": by_event_type,
        "by_location": by_location,
    }


@router.get("/{event_id}", response_model=EventResponse)
def get_event(
    event_id: int,
    db: Session = Depends(get_db),
) -> EventResponse:
    """Return a single event by its primary key."""
    ev: Optional[Event] = db.query(Event).filter(Event.id == event_id).first()
    if ev is None:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found.")
    return EventResponse.model_validate(ev)

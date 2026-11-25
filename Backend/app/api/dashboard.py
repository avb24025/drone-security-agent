"""
app/api/dashboard.py
─────────────────────
Dashboard summary endpoint.

GET /api/dashboard/summary – aggregated stats for the overview cards.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import Alert, Event, Frame
from app.models.schemas import DashboardSummary

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummary)
def dashboard_summary(db: Session = Depends(get_db)) -> DashboardSummary:
    """
    Return aggregate statistics used by the React overview panel.

    All counts are computed with single COUNT queries to keep it fast.
    """
    total_frames = db.query(Frame).count()
    total_events = db.query(Event).count()
    total_alerts = db.query(Alert).count()

    high_priority = (
        db.query(Alert)
        .filter(Alert.severity.in_(["critical", "high"]))
        .count()
    )
    unread_alerts = (
        db.query(Alert)
        .filter(Alert.status == "unread")
        .count()
    )

    monitoring_status = "active" if total_frames > 0 else "idle"

    return DashboardSummary(
        total_frames=total_frames,
        total_events=total_events,
        total_alerts=total_alerts,
        high_priority_alerts=high_priority,
        unread_alerts=unread_alerts,
        monitoring_status=monitoring_status,
    )

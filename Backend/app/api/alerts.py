"""
app/api/alerts.py
─────────────────
Alert management endpoints.

GET   /api/alerts             – list alerts (filterable)
GET   /api/alerts/{alert_id}  – single alert
PATCH /api/alerts/{alert_id}  – update status (unread → read → resolved)
GET   /api/alerts/stats       – severity and status counts for dashboard
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import Alert
from app.models.schemas import AlertResponse, AlertStatusUpdate
from app.services.alert_engine import get_alerts, update_alert_status

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("", response_model=List[AlertResponse])
def list_alerts(
    severity: Optional[str] = Query(None, description="critical | high | medium | low"),
    status: Optional[str] = Query(None, description="unread | read | resolved"),
    location: Optional[str] = Query(None),
    alert_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> List[AlertResponse]:
    """Return security alerts with optional filters, newest-first."""
    alerts = get_alerts(
        db=db,
        severity=severity,
        status=status,
        location=location,
        alert_type=alert_type,
        limit=limit,
        offset=offset,
    )
    return [AlertResponse.model_validate(a) for a in alerts]


@router.get("/stats", response_model=Dict[str, Any])
def alert_stats(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Return counts grouped by severity and status.
    Used by the dashboard overview cards.
    """
    all_alerts: List[Alert] = db.query(Alert).all()

    by_severity: Dict[str, int] = {}
    by_status: Dict[str, int] = {}
    by_type: Dict[str, int] = {}

    for a in all_alerts:
        by_severity[a.severity] = by_severity.get(a.severity, 0) + 1
        by_status[a.status]     = by_status.get(a.status, 0) + 1
        by_type[a.alert_type]   = by_type.get(a.alert_type, 0) + 1

    return {
        "total": len(all_alerts),
        "by_severity": by_severity,
        "by_status": by_status,
        "by_type": by_type,
        "unread": by_status.get("unread", 0),
        "high_priority": (
            by_severity.get("critical", 0) + by_severity.get("high", 0)
        ),
    }


@router.get("/{alert_id}", response_model=AlertResponse)
def get_alert(
    alert_id: int,
    db: Session = Depends(get_db),
) -> AlertResponse:
    """Return a single alert by its primary key."""
    alert: Optional[Alert] = db.query(Alert).filter(Alert.id == alert_id).first()
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")
    return AlertResponse.model_validate(alert)


@router.patch("/{alert_id}", response_model=AlertResponse)
def patch_alert_status(
    alert_id: int,
    body: AlertStatusUpdate,
    db: Session = Depends(get_db),
) -> AlertResponse:
    """Update the workflow status of an alert (unread → read → resolved)."""
    try:
        updated = update_alert_status(alert_id, body.status, db)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if updated is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")

    db.commit()
    return AlertResponse.model_validate(updated)

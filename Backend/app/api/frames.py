"""
app/api/frames.py
─────────────────
Frame processing and retrieval endpoints.

POST /api/frames/process   – single frame → Gemini → events → alerts → DB
POST /api/frames/batch     – sequential batch of frames
GET  /api/frames           – list indexed frames (filterable)
GET  /api/frames/{frame_id} – one frame + its analysis + events
GET  /api/simulator/frames – return the raw simulated dataset (no processing)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.schemas import (
    BatchFrameRequest,
    BatchProcessingResult,
    DroneFrame,
    FrameProcessingResult,
    FrameResponse,
    GeminiAnalysis,
)
from app.services import alert_engine, event_processor, gemini_service
from app.services.alert_engine import AlertEngineConfig
from app.simulator.drone_simulator import get_all_frames, get_scenario_summary

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/frames", tags=["frames"])


# ─────────────────────────────────────────────────────────────────────────────
# Internal orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def _process_one_frame(
    frame: DroneFrame,
    db: Session,
    cfg: AlertEngineConfig,
) -> FrameProcessingResult:
    """
    Run the full pipeline for a single frame inside an existing DB session.
    Does NOT commit — caller owns the transaction.
    Returns a FrameProcessingResult with counts; raises on hard failure.
    """
    # Step 1 – Gemini analysis
    analysis: GeminiAnalysis = gemini_service.analyze_frame(frame)

    # Step 2 – Event extraction + context
    ctx = event_processor.process_frame(
        frame=frame,
        analysis=analysis,
        db=db,
        restricted_hour_start=cfg.restricted_hour_start,
        restricted_hour_end=cfg.restricted_hour_end,
    )

    # Step 3 – Alert engine
    new_alerts = alert_engine.evaluate(ctx, db, cfg)

    return FrameProcessingResult(
        frame_id=frame.frame_id,
        status="processed",
        events_created=len(ctx.events),
        alerts_created=len(new_alerts),
        analysis=analysis,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-frame processing
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/process", response_model=FrameProcessingResult, status_code=201)
def process_single_frame(
    frame: DroneFrame,
    db: Session = Depends(get_db),
) -> FrameProcessingResult:
    """
    Process one drone frame end-to-end.

    - Sends the description to Gemini for analysis.
    - Extracts structured events and maintains context.
    - Evaluates security rules and creates alerts.
    - Persists everything in Neon PostgreSQL.
    """
    cfg = AlertEngineConfig.from_settings()
    try:
        result = _process_one_frame(frame, db, cfg)
        db.commit()
        logger.info(
            "Frame %d processed: events=%d alerts=%d",
            frame.frame_id, result.events_created, result.alerts_created,
        )
        return result
    except Exception as exc:
        db.rollback()
        logger.exception("Failed to process frame %d: %s", frame.frame_id, exc)
        raise HTTPException(status_code=500, detail=f"Frame processing failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/batch", response_model=BatchProcessingResult, status_code=201)
def process_batch(
    request: BatchFrameRequest,
    db: Session = Depends(get_db),
) -> BatchProcessingResult:
    """
    Process multiple drone frames sequentially.

    Each frame is processed in its own commit so a failure on one frame
    does not roll back already-processed frames.  Statistics are returned
    for the whole batch.
    """
    cfg = AlertEngineConfig.from_settings()
    results: List[FrameProcessingResult] = []
    total_events = 0
    total_alerts = 0
    failed = 0

    for frame in request.frames:
        try:
            result = _process_one_frame(frame, db, cfg)
            db.commit()
            results.append(result)
            total_events += result.events_created
            total_alerts += result.alerts_created
        except Exception as exc:
            db.rollback()
            failed += 1
            logger.error("Batch: frame %d failed: %s", frame.frame_id, exc)
            results.append(FrameProcessingResult(
                frame_id=frame.frame_id,
                status=f"failed: {exc}",
                events_created=0,
                alerts_created=0,
            ))

    logger.info(
        "Batch complete: total=%d processed=%d failed=%d events=%d alerts=%d",
        len(request.frames),
        len(request.frames) - failed,
        failed,
        total_events,
        total_alerts,
    )

    return BatchProcessingResult(
        total_frames=len(request.frames),
        processed=len(request.frames) - failed,
        failed=failed,
        total_events=total_events,
        total_alerts=total_alerts,
        results=results,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Frame retrieval
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", response_model=List[FrameResponse])
def list_frames(
    location: Optional[str] = Query(None, description="Filter by location"),
    start_time: Optional[datetime] = Query(None, description="ISO-8601 start timestamp"),
    end_time: Optional[datetime] = Query(None, description="ISO-8601 end timestamp"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> List[FrameResponse]:
    """Return indexed frames with optional filters, newest-first."""
    frames = event_processor.get_all_frames(
        db=db,
        location=location,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset,
    )
    return [FrameResponse.model_validate(f) for f in frames]


@router.get("/{frame_id}", response_model=Dict[str, Any])
def get_frame(
    frame_id: int,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Return one indexed frame together with its extracted events.
    404 if the frame has not been processed yet.
    """
    db_frame, events = event_processor.get_frame_with_events(frame_id, db)
    if db_frame is None:
        raise HTTPException(status_code=404, detail=f"Frame {frame_id} not found.")

    from app.models.schemas import EventResponse
    return {
        "frame": FrameResponse.model_validate(db_frame).model_dump(),
        "events": [EventResponse.model_validate(e).model_dump() for e in events],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Simulator data endpoint (no processing)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/simulator/frames", response_model=List[Dict[str, Any]])
def get_simulator_frames() -> List[Dict[str, Any]]:
    """
    Return the raw simulated frame dataset without processing it.
    Used by the React dashboard to populate the simulator panel.
    """
    frames = get_all_frames()
    return [f.model_dump() for f in frames]


@router.get("/simulator/summary", response_model=List[Dict[str, Any]])
def get_simulator_summary() -> List[Dict[str, Any]]:
    """Lightweight summary (no descriptions) for the simulator frame list UI."""
    return get_scenario_summary()

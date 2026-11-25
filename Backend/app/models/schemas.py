"""
app/models/schemas.py
─────────────────────
Pydantic v2 schemas for all API request and response bodies.
These are intentionally separate from the SQLAlchemy ORM models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────────────────────
# Drone / Simulator
# ─────────────────────────────────────────────────────────────────────────────

class DroneFrame(BaseModel):
    """Incoming telemetry frame from the drone simulator."""

    frame_id: int = Field(..., description="Unique frame identifier")
    timestamp: datetime = Field(..., description="ISO-8601 capture timestamp")
    location: str = Field(..., description="Human-readable location label")
    altitude: Optional[float] = Field(None, description="Drone altitude in metres")
    latitude: Optional[float] = Field(None, description="GPS latitude")
    longitude: Optional[float] = Field(None, description="GPS longitude")
    description: str = Field(..., description="Text description of what the frame shows")


class BatchFrameRequest(BaseModel):
    """Request body for batch frame processing."""
    frames: List[DroneFrame]


# ─────────────────────────────────────────────────────────────────────────────
# Gemini / AI analysis
# ─────────────────────────────────────────────────────────────────────────────

class DetectedObject(BaseModel):
    """One object identified in a frame."""
    type: str = Field(..., description="broad category: person, vehicle, animal, etc.")
    category: Optional[str] = Field(None, description="sub-category, e.g. truck, sedan")
    color: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None


class DetectedActivity(BaseModel):
    """One activity identified in a frame."""
    subject: str
    action: str
    location: Optional[str] = None


class GeminiAnalysis(BaseModel):
    """
    Structured output returned by the Gemini frame analysis service.
    All fields are optional because Gemini may not always identify everything.
    """
    objects: List[DetectedObject] = Field(default_factory=list)
    activities: List[DetectedActivity] = Field(default_factory=list)
    event_type: Optional[str] = None
    description: Optional[str] = None
    is_suspicious: Optional[bool] = None
    raw_response: Optional[str] = None   # kept for debugging, not shown in API


# ─────────────────────────────────────────────────────────────────────────────
# Frames
# ─────────────────────────────────────────────────────────────────────────────

class FrameResponse(BaseModel):
    """API response for a single indexed frame."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    frame_id: int
    timestamp: datetime
    location: str
    altitude: Optional[float]
    latitude: Optional[float]
    longitude: Optional[float]
    original_description: str
    gemini_analysis: Optional[Dict[str, Any]]
    processed_at: datetime


class FrameProcessingResult(BaseModel):
    """Returned after a single frame is processed end-to-end."""
    frame_id: int
    status: str
    events_created: int
    alerts_created: int
    analysis: Optional[GeminiAnalysis] = None


class BatchProcessingResult(BaseModel):
    """Returned after a batch of frames is processed."""
    total_frames: int
    processed: int
    failed: int
    total_events: int
    total_alerts: int
    results: List[FrameProcessingResult]


# ─────────────────────────────────────────────────────────────────────────────
# Events
# ─────────────────────────────────────────────────────────────────────────────

class EventResponse(BaseModel):
    """API response for a single indexed event."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    frame_id: int
    timestamp: datetime
    location: str
    object_type: str
    object_attributes: Optional[Dict[str, Any]]
    activity: Optional[str]
    event_type: str
    description: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Alerts
# ─────────────────────────────────────────────────────────────────────────────

class AlertResponse(BaseModel):
    """API response for a single security alert."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    event_id: Optional[int]
    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    location: str
    frame_id: Optional[int]
    status: str


class AlertStatusUpdate(BaseModel):
    """Request body for PATCH /api/alerts/{alert_id}."""
    status: str = Field(..., pattern="^(unread|read|resolved)$")


# ─────────────────────────────────────────────────────────────────────────────
# Chatbot
# ─────────────────────────────────────────────────────────────────────────────

class ChatSource(BaseModel):
    """One piece of evidence the agent used to form its answer."""
    event_id: Optional[int] = None
    frame_id: Optional[int] = None
    timestamp: Optional[datetime] = None
    location: Optional[str] = None
    description: Optional[str] = None


class ChatRequest(BaseModel):
    """Incoming chat message from the user."""
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default", description="Conversation session identifier")


class ChatResponse(BaseModel):
    """Answer returned by the AI Security Analyst Agent."""
    answer: str
    sources: List[ChatSource] = Field(default_factory=list)
    session_id: str


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard summary
# ─────────────────────────────────────────────────────────────────────────────

class DashboardSummary(BaseModel):
    """High-level statistics for the overview cards."""
    total_frames: int
    total_events: int
    total_alerts: int
    high_priority_alerts: int
    unread_alerts: int
    monitoring_status: str

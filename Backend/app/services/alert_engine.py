"""
app/services/alert_engine.py
─────────────────────────────
Automatic rule-based security alert engine.

Responsibilities
────────────────
1. Receive a ProcessingContext (produced by event_processor) and evaluate
   every active rule against it.
2. For each rule that fires, create an Alert ORM row in the database.
3. Deduplicate: a rule may not produce a second alert for the same logical
   event.  Deduplication is keyed on a deterministic string stored in
   Alert.dedup_key.
4. Never generate alerts inside the chatbot path — alert generation is
   independent of user interaction.

Rules implemented
─────────────────
  RULE 1 – suspicious_loitering
      Fires when a person is at the Main Gate (or any configurable location)
      during restricted hours AND at least `loitering_min_frames` prior
      person observations already exist at that location.
      Severity: high

  RULE 2 – unauthorized_vehicle
      Fires when a vehicle event is classified as "unauthorized_vehicle" OR
      when none of the vehicle's identifying attributes match any entry in
      the authorised-vehicle list.
      Severity: high

  RULE 3 – repeated_vehicle_appearance
      Fires when the ProcessingContext reports ≥1 prior confirmed/possible
      appearance of a vehicle in this frame.  The alert message names the
      vehicle attributes and the total appearance count.
      Severity: medium

  RULE 4 – restricted_area_intrusion
      Fires when an event is classified as "restricted_area_intrusion" or
      when a person / vehicle is detected at any location in the configured
      restricted-areas list.
      Severity: critical

Deduplication key format
────────────────────────
  "<rule_name>:<frame_id>:<event_id>"

  Using both frame_id and event_id means:
  • Re-processing the same frame re-evaluates rules but, because the event
    rows are recreated with new IDs, the new dedup keys are distinct.
    This is intentional — reprocessing may change the analysis.
  • The same rule cannot fire twice for the exact same event row.

Design decisions
────────────────
• The engine is a pure function: `evaluate(ctx, db, settings) → list[Alert]`.
  It has no internal state, making it easy to test.
• All DB writes (INSERT of Alert rows) happen inside this call but the
  commit is owned by the caller (FastAPI route), maintaining one atomic
  transaction per frame.
• Authorised-vehicle matching uses the same normalised lower-case comparison
  as the event processor's confidence scorer.
• Restricted areas are configurable via settings.  The default list covers
  the scenarios in the simulated dataset.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.db_models import Alert, Event
from app.models.schemas import GeminiAnalysis
from app.services.event_processor import MatchConfidence, ProcessingContext

logger = logging.getLogger(__name__)

# ── Default restricted areas (overridable via Settings) ──────────────────────
DEFAULT_RESTRICTED_AREAS: List[str] = [
    "garage",
    "server room",
    "restricted zone",
    "storage shed",
]

# ── Severity constants ────────────────────────────────────────────────────────
SEVERITY_CRITICAL = "critical"
SEVERITY_HIGH     = "high"
SEVERITY_MEDIUM   = "medium"
SEVERITY_LOW      = "low"


# ─────────────────────────────────────────────────────────────────────────────
# Public configuration container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlertEngineConfig:
    """
    Runtime configuration for the alert engine.

    All values are read from app.config.Settings when the engine is called
    through FastAPI.  Tests can pass custom values directly.
    """
    restricted_hour_start: int = 22
    restricted_hour_end: int = 6
    loitering_min_frames: int = 2
    authorized_vehicles: List[str] = field(default_factory=lambda: [
        "blue ford f150",
        "white toyota camry",
    ])
    restricted_areas: List[str] = field(
        default_factory=lambda: list(DEFAULT_RESTRICTED_AREAS)
    )

    @classmethod
    def from_settings(cls) -> "AlertEngineConfig":
        """Build config from the application Settings singleton."""
        from app.config import get_settings
        s = get_settings()
        return cls(
            restricted_hour_start=s.restricted_hour_start,
            restricted_hour_end=s.restricted_hour_end,
            loitering_min_frames=s.loitering_min_frames,
            authorized_vehicles=s.authorized_vehicle_list,
            restricted_areas=DEFAULT_RESTRICTED_AREAS,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dedup_key(rule: str, frame_id: int, event_id: int) -> str:
    """Build a deterministic deduplication key for one rule + event pair."""
    raw = f"{rule}:{frame_id}:{event_id}"
    # Hash to a fixed-length string safe for the VARCHAR(512) column.
    return hashlib.sha256(raw.encode()).hexdigest()[:64] + f":{raw}"


def _alert_exists(dedup_key: str, db: Session) -> bool:
    """Return True if an alert with this dedup_key already exists."""
    return db.query(Alert).filter(Alert.dedup_key == dedup_key).count() > 0


def _make_alert(
    *,
    rule: str,
    severity: str,
    message: str,
    timestamp: datetime,
    location: str,
    frame_id: int,
    event_id: Optional[int],
    db: Session,
) -> Optional[Alert]:
    """
    Create and persist an Alert row unless a duplicate already exists.
    Returns the new Alert or None if deduplicated.
    """
    dk = _dedup_key(rule, frame_id, event_id or 0)
    if _alert_exists(dk, db):
        logger.debug("Alert deduplicated: rule=%s frame=%d event=%s", rule, frame_id, event_id)
        return None

    alert = Alert(
        event_id=event_id,
        alert_type=rule,
        severity=severity,
        message=message,
        timestamp=timestamp,
        location=location,
        frame_id=frame_id,
        dedup_key=dk,
        status="unread",
    )
    db.add(alert)
    db.flush()   # assign PK without committing
    logger.info(
        "Alert created: rule=%s severity=%s frame=%d location=%s",
        rule, severity, frame_id, location,
    )
    return alert


def _in_restricted_hours(hour: int, start: int, end: int) -> bool:
    """
    Return True if `hour` falls within [start, end) wrapping around midnight.
    e.g. start=22, end=6 → hours 22,23,0,1,2,3,4,5 are restricted.
    """
    if start > end:          # wraps midnight, e.g. 22 → 06
        return hour >= start or hour < end
    return start <= hour < end   # does not wrap, e.g. 09 → 17


def _vehicle_label(attrs: dict) -> str:
    """Build a human-readable vehicle label from stored attributes."""
    parts = []
    if attrs.get("color"):
        parts.append(attrs["color"].capitalize())
    if attrs.get("make"):
        parts.append(attrs["make"])
    if attrs.get("model"):
        parts.append(attrs["model"])
    if attrs.get("category") and not parts:
        parts.append(attrs["category"].capitalize())
    return " ".join(parts) if parts else "Unknown vehicle"


def _is_authorized(attrs: dict, authorized_list: List[str]) -> bool:
    """
    Return True if the vehicle described by `attrs` matches any entry in the
    authorised list.  Matching is by substring (normalized lower-case) so
    "blue Ford F150" in attrs matches "blue ford f150" in the list.
    """
    label = _vehicle_label(attrs).lower()
    color = (attrs.get("color") or "").lower()
    make  = (attrs.get("make") or "").lower()
    model = (attrs.get("model") or "").lower()

    for auth in authorized_list:
        auth_lower = auth.lower()
        # Accept if the label is a substring of the authorised entry or vice-versa
        if auth_lower in label or label in auth_lower:
            return True
        # Also accept if individual key attributes appear together in the entry
        if make and make in auth_lower and (not color or color in auth_lower):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Rule implementations
# ─────────────────────────────────────────────────────────────────────────────

def _rule_suspicious_loitering(
    ctx: ProcessingContext,
    config: AlertEngineConfig,
    db: Session,
) -> List[Alert]:
    """
    Rule 1 – Suspicious Loitering

    Fires when ALL of:
    - At least one person event exists in this frame.
    - The event timestamp is within restricted hours.
    - The number of PRIOR person observations at this location during
      restricted hours is >= (loitering_min_frames - 1), meaning this frame
      is the Nth sighting needed to meet the threshold.
    """
    alerts: List[Alert] = []

    for ev in ctx.events:
        if ev.object_type != "person":
            continue

        hour = ev.timestamp.hour
        if not _in_restricted_hours(hour, config.restricted_hour_start, config.restricted_hour_end):
            continue

        # ctx already carries the prior-person count computed after flush
        prior_count = ctx.prior_person_appearances_at_location
        threshold = config.loitering_min_frames - 1   # priors needed before THIS frame

        if prior_count < threshold:
            logger.debug(
                "Loitering rule: frame=%d has %d prior person(s) at %s, need %d — not yet.",
                ctx.frame_id, prior_count, ev.location, threshold,
            )
            continue

        total = prior_count + 1   # include current frame
        message = (
            f"Suspicious loitering detected: a person has been observed near "
            f"{ev.location} {total} time(s) during restricted hours "
            f"({ev.timestamp.strftime('%H:%M')})."
        )
        alert = _make_alert(
            rule="suspicious_loitering",
            severity=SEVERITY_HIGH,
            message=message,
            timestamp=ev.timestamp,
            location=ev.location,
            frame_id=ctx.frame_id,
            event_id=ev.id,
            db=db,
        )
        if alert:
            alerts.append(alert)

    return alerts


def _rule_unauthorized_vehicle(
    ctx: ProcessingContext,
    config: AlertEngineConfig,
    db: Session,
) -> List[Alert]:
    """
    Rule 2 – Unauthorized Vehicle

    Fires when a vehicle event is present AND either:
    (a) The event_type is explicitly "unauthorized_vehicle", OR
    (b) The vehicle's attributes do not match any entry in the authorised list
        AND the vehicle has at least one identifiable attribute (so we don't
        alert on every completely-unknown vehicle seen in a routine scan).
    """
    alerts: List[Alert] = []

    for ev in ctx.events:
        if ev.object_type != "vehicle":
            continue

        attrs = ev.object_attributes or {}
        explicit_unauthorized = ev.event_type == "unauthorized_vehicle"
        has_identity = any([
            attrs.get("color"), attrs.get("make"),
            attrs.get("model"), attrs.get("category"),
        ])

        if not explicit_unauthorized and not has_identity:
            # Completely unknown vehicle with no identifying info — skip to
            # avoid alert noise on routine patrol frames.
            continue

        authorized = _is_authorized(attrs, config.authorized_vehicles)
        if authorized and not explicit_unauthorized:
            continue

        label = _vehicle_label(attrs)
        message = (
            f"Unauthorized vehicle detected at {ev.location}: "
            f"{label}. No entry authorisation observed."
        )
        alert = _make_alert(
            rule="unauthorized_vehicle",
            severity=SEVERITY_HIGH,
            message=message,
            timestamp=ev.timestamp,
            location=ev.location,
            frame_id=ctx.frame_id,
            event_id=ev.id,
            db=db,
        )
        if alert:
            alerts.append(alert)

    return alerts


def _rule_repeated_vehicle_appearance(
    ctx: ProcessingContext,
    config: AlertEngineConfig,
    db: Session,
) -> List[Alert]:
    """
    Rule 3 – Repeated Vehicle Appearance

    Fires when ctx.prior_vehicle_appearances contains at least one CONFIRMED
    match.  One alert is raised per unique prior-appearance chain (not per
    individual prior event) to avoid flooding.
    """
    alerts: List[Alert] = []

    confirmed = [
        a for a in ctx.prior_vehicle_appearances
        if a.confidence == MatchConfidence.CONFIRMED
    ]
    if not confirmed:
        return alerts

    # Deduplicate by vehicle signature — one alert per distinct vehicle seen again
    seen_signatures: set[str] = set()

    for prior in confirmed:
        sig = f"{prior.color}|{prior.make}|{prior.model}|{prior.category}"
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)

        label = _vehicle_label({
            "color": prior.color, "make": prior.make,
            "model": prior.model, "category": prior.category,
        })
        total_appearances = len([
            a for a in ctx.prior_vehicle_appearances
            if f"{a.color}|{a.make}|{a.model}|{a.category}" == sig
        ]) + 1   # +1 for the current frame

        # Find the vehicle event in this frame to link the alert
        vehicle_event: Optional[Event] = next(
            (e for e in ctx.events if e.object_type == "vehicle"), None
        )
        ev_id = vehicle_event.id if vehicle_event else None
        location = vehicle_event.location if vehicle_event else "Unknown"
        timestamp = vehicle_event.timestamp if vehicle_event else datetime.utcnow()

        message = (
            f"Repeated vehicle appearance: {label} has been detected "
            f"{total_appearances} time(s) today. "
            f"Current sighting at {location}."
        )
        alert = _make_alert(
            rule="repeated_vehicle_appearance",
            severity=SEVERITY_MEDIUM,
            message=message,
            timestamp=timestamp,
            location=location,
            frame_id=ctx.frame_id,
            event_id=ev_id,
            db=db,
        )
        if alert:
            alerts.append(alert)

    return alerts


def _rule_restricted_area_intrusion(
    ctx: ProcessingContext,
    config: AlertEngineConfig,
    db: Session,
) -> List[Alert]:
    """
    Rule 4 – Restricted Area Intrusion

    Fires when ANY event in this frame is EITHER:
    (a) classified event_type == "restricted_area_intrusion", OR
    (b) located in one of the configured restricted areas.
    """
    alerts: List[Alert] = []
    restricted_lower = [r.lower() for r in config.restricted_areas]

    for ev in ctx.events:
        is_explicit    = ev.event_type == "restricted_area_intrusion"
        is_in_area     = ev.location.lower() in restricted_lower

        if not is_explicit and not is_in_area:
            continue

        subject = "Person" if ev.object_type == "person" else "Vehicle"
        message = (
            f"Restricted area intrusion: {subject} detected in "
            f"{ev.location} at {ev.timestamp.strftime('%H:%M')}. "
            f"Unauthorised access to restricted zone."
        )
        alert = _make_alert(
            rule="restricted_area_intrusion",
            severity=SEVERITY_CRITICAL,
            message=message,
            timestamp=ev.timestamp,
            location=ev.location,
            frame_id=ctx.frame_id,
            event_id=ev.id,
            db=db,
        )
        if alert:
            alerts.append(alert)

    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    ctx: ProcessingContext,
    db: Session,
    config: Optional[AlertEngineConfig] = None,
) -> List[Alert]:
    """
    Evaluate all security rules against the given ProcessingContext and
    persist any new alerts to the database.

    Args:
        ctx:    ProcessingContext produced by event_processor.process_frame().
        db:     SQLAlchemy session.  The caller commits after this returns.
        config: Optional AlertEngineConfig.  If None, reads from Settings.

    Returns:
        List of newly created Alert ORM objects (empty if all deduplicated).
    """
    if config is None:
        config = AlertEngineConfig.from_settings()

    all_alerts: List[Alert] = []

    rules = [
        ("suspicious_loitering",        _rule_suspicious_loitering),
        ("unauthorized_vehicle",         _rule_unauthorized_vehicle),
        ("repeated_vehicle_appearance",  _rule_repeated_vehicle_appearance),
        ("restricted_area_intrusion",    _rule_restricted_area_intrusion),
    ]

    for rule_name, rule_fn in rules:
        try:
            new = rule_fn(ctx, config, db)
            all_alerts.extend(new)
        except Exception as exc:   # noqa: BLE001
            # One rule failing must not prevent the others from running.
            logger.error(
                "Alert rule %s raised an unexpected error for frame %d: %s",
                rule_name, ctx.frame_id, exc, exc_info=True,
            )

    if all_alerts:
        logger.info(
            "Frame %d: %d alert(s) generated (%s).",
            ctx.frame_id,
            len(all_alerts),
            ", ".join(a.alert_type for a in all_alerts),
        )
    else:
        logger.debug("Frame %d: no alerts generated.", ctx.frame_id)

    return all_alerts


def get_alerts(
    db: Session,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    location: Optional[str] = None,
    alert_type: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
) -> List[Alert]:
    """
    Return alerts with optional filters, newest-first.
    Used by the alerts API route and the LangChain agent.
    """
    q = db.query(Alert)
    if severity:
        q = q.filter(Alert.severity == severity)
    if status:
        q = q.filter(Alert.status == status)
    if location:
        q = q.filter(Alert.location == location)
    if alert_type:
        q = q.filter(Alert.alert_type == alert_type)
    return q.order_by(Alert.timestamp.desc()).offset(offset).limit(limit).all()


def update_alert_status(alert_id: int, new_status: str, db: Session) -> Optional[Alert]:
    """
    Update the workflow status of one alert.

    Args:
        alert_id:   Primary key of the Alert row.
        new_status: One of 'unread', 'read', 'resolved'.
        db:         SQLAlchemy session.

    Returns:
        The updated Alert, or None if not found.
    """
    valid = {"unread", "read", "resolved"}
    if new_status not in valid:
        raise ValueError(f"Invalid status {new_status!r}. Must be one of {valid}.")

    alert: Optional[Alert] = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        return None

    alert.status = new_status
    db.flush()
    logger.info("Alert %d status updated to %r.", alert_id, new_status)
    return alert

"""
app/services/agent.py
----------------------
LangChain AI Security Analyst Agent.

Responsibilities
----------------
1. Expose a single public function: answer_question(question, session_id, db)
   that the /api/chat endpoint calls.
2. Build controlled, parameterised database retrieval tools — the LLM never
   executes raw SQL, it only calls these functions.
3. Maintain per-session conversational context using ConversationBufferWindowMemory
   so the user can ask follow-up questions.
4. Ground every answer in actual database records — if the database has no
   data the agent says so explicitly rather than inventing events.
5. Keep Gemini integration separate from the event-processing pipeline:
   the agent uses a ChatGoogleGenerativeAI LLM while frame analysis uses
   the lower-level google-generativeai SDK directly.

Architecture
------------
  answer_question()
    -> _get_agent_executor(db, session_id)
         -> _build_tools(db)          : six StructuredTool instances
         -> _build_prompt()           : system + history + human turn
         -> create_tool_calling_agent : binds LLM + tools + prompt
         -> AgentExecutor             : runs the ReAct loop
    -> AgentExecutor.invoke({"input": question, "chat_history": ...})
    -> _build_response(output, sources_accumulated)

Tool isolation
--------------
Every tool function receives the db Session via closure — the LLM cannot
pass a session object, cannot call arbitrary SQL, and cannot access any
module outside what the tool exposes.

Memory
------
One ConversationBufferWindowMemory per session_id, stored in a module-level
dict.  Sessions are keyed by the string session_id supplied by the caller.
The window is capped at 10 exchanges (configurable) to avoid token bloat.

Error handling
--------------
If the database is empty, tools return a human-readable "no records found"
string.  The LLM is instructed to pass that message to the user rather
than fabricating a response.  API errors from Gemini are caught and
re-raised as RuntimeError so the chat endpoint can return a 500.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field as PField
from sqlalchemy.orm import Session

from app.models.db_models import Alert, Event, Frame
from app.models.schemas import ChatResponse, ChatSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-session memory store
# ---------------------------------------------------------------------------
# Maps session_id -> ConversationBufferWindowMemory
_session_memories: Dict[str, Any] = {}
_MEMORY_WINDOW_K = 10        # number of exchanges kept in context


# ---------------------------------------------------------------------------
# Tool input schemas  (Pydantic v2 — LangChain reads these for type checking)
# ---------------------------------------------------------------------------

class SearchEventsInput(BaseModel):
    object_type: Optional[str] = PField(
        None,
        description="Filter by object type: 'person', 'vehicle', or 'unknown'."
    )
    location: Optional[str] = PField(
        None,
        description="Filter by location label, e.g. 'Main Gate', 'Garage'."
    )
    event_type: Optional[str] = PField(
        None,
        description=(
            "Filter by event type: vehicle_entry, vehicle_exit, person_detected, "
            "loitering, restricted_area_intrusion, unauthorized_vehicle, "
            "suspicious_object, normal_activity, unknown."
        )
    )
    start_time: Optional[str] = PField(
        None,
        description="ISO-8601 start datetime, e.g. '2026-11-24T00:00:00'."
    )
    end_time: Optional[str] = PField(
        None,
        description="ISO-8601 end datetime, e.g. '2026-11-24T23:59:59'."
    )
    limit: int = PField(20, description="Maximum number of records to return (1-50).")


class SearchFramesInput(BaseModel):
    location: Optional[str] = PField(None, description="Filter by location label.")
    start_time: Optional[str] = PField(None, description="ISO-8601 start datetime.")
    end_time: Optional[str] = PField(None, description="ISO-8601 end datetime.")
    limit: int = PField(10, description="Maximum number of frames to return (1-50).")


class SearchAlertsInput(BaseModel):
    severity: Optional[str] = PField(
        None,
        description="Filter by severity: 'critical', 'high', 'medium', or 'low'."
    )
    status: Optional[str] = PField(
        None,
        description="Filter by status: 'unread', 'read', or 'resolved'."
    )
    alert_type: Optional[str] = PField(
        None,
        description=(
            "Filter by alert type: suspicious_loitering, unauthorized_vehicle, "
            "repeated_vehicle_appearance, restricted_area_intrusion."
        )
    )
    limit: int = PField(20, description="Maximum number of alerts to return (1-50).")


class GetVehicleHistoryInput(BaseModel):
    color: Optional[str]    = PField(None, description="Vehicle colour, e.g. 'blue'.")
    make:  Optional[str]    = PField(None, description="Vehicle manufacturer, e.g. 'Ford'.")
    model: Optional[str]    = PField(None, description="Vehicle model, e.g. 'F150'.")
    category: Optional[str] = PField(None, description="Vehicle category, e.g. 'truck', 'sedan'.")


class GetDailySummaryInput(BaseModel):
    date_str: Optional[str] = PField(
        None,
        description=(
            "Date in YYYY-MM-DD format. Defaults to the most recent date "
            "that has records if omitted."
        )
    )


class GetLocationActivityInput(BaseModel):
    location: str = PField(..., description="Location label to query, e.g. 'Main Gate'.")
    limit: int    = PField(20, description="Maximum number of events to return.")


# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------

def _fmt_dt(dt: Optional[datetime]) -> str:
    if dt is None:
        return "unknown time"
    return dt.strftime("%Y-%m-%d %H:%M")


def _fmt_event(ev: Event) -> Dict[str, Any]:
    attrs = ev.object_attributes or {}
    parts = [ev.object_type]
    if attrs.get("color"):
        parts.append(attrs["color"])
    if attrs.get("make"):
        parts.append(attrs["make"])
    if attrs.get("model"):
        parts.append(attrs["model"])
    return {
        "event_id":   ev.id,
        "frame_id":   ev.frame_id,
        "timestamp":  _fmt_dt(ev.timestamp),
        "location":   ev.location,
        "object":     " ".join(parts),
        "activity":   ev.activity or "unspecified",
        "event_type": ev.event_type,
        "description": ev.description or "",
    }


def _fmt_alert(a: Alert) -> Dict[str, Any]:
    return {
        "alert_id":   a.id,
        "alert_type": a.alert_type,
        "severity":   a.severity,
        "message":    a.message,
        "timestamp":  _fmt_dt(a.timestamp),
        "location":   a.location,
        "status":     a.status,
    }


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(val, hi))


# ---------------------------------------------------------------------------
# Tool functions  (closures capture the db session)
# ---------------------------------------------------------------------------

def _make_tools(db: Session, sources: List[ChatSource]):
    """
    Build all six database retrieval tools, closing over `db` and `sources`.
    Every tool appends ChatSource entries to `sources` so the caller can
    include them in the response without a second DB round-trip.
    """
    from langchain.tools import StructuredTool

    # ------------------------------------------------------------------ #
    # Tool 1 – search_events
    # ------------------------------------------------------------------ #
    def search_events(
        object_type: Optional[str] = None,
        location: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        limit = _clamp(limit, 1, 50)
        try:
            q = db.query(Event)
            if object_type:
                q = q.filter(Event.object_type == object_type.lower())
            if location:
                from sqlalchemy import func
                q = q.filter(func.lower(Event.location) == location.lower())
            if event_type:
                q = q.filter(Event.event_type == event_type)
            if start_time:
                dt = _parse_dt(start_time)
                if dt:
                    q = q.filter(Event.timestamp >= dt)
            if end_time:
                dt = _parse_dt(end_time)
                if dt:
                    q = q.filter(Event.timestamp <= dt)
            events = q.order_by(Event.timestamp.desc()).limit(limit).all()
        except Exception as exc:
            logger.error("search_events DB error: %s", exc)
            return f"Database error while searching events: {exc}"

        if not events:
            return "No events found matching the given criteria."

        for ev in events:
            sources.append(ChatSource(
                event_id=ev.id,
                frame_id=ev.frame_id,
                timestamp=ev.timestamp,
                location=ev.location,
                description=ev.description,
            ))
        return json.dumps([_fmt_event(e) for e in events], indent=2)

    # ------------------------------------------------------------------ #
    # Tool 2 – search_frames
    # ------------------------------------------------------------------ #
    def search_frames(
        location: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        limit = _clamp(limit, 1, 50)
        try:
            q = db.query(Frame)
            if location:
                from sqlalchemy import func
                q = q.filter(func.lower(Frame.location) == location.lower())
            if start_time:
                dt = _parse_dt(start_time)
                if dt:
                    q = q.filter(Frame.timestamp >= dt)
            if end_time:
                dt = _parse_dt(end_time)
                if dt:
                    q = q.filter(Frame.timestamp <= dt)
            frames = q.order_by(Frame.timestamp.desc()).limit(limit).all()
        except Exception as exc:
            logger.error("search_frames DB error: %s", exc)
            return f"Database error while searching frames: {exc}"

        if not frames:
            return "No frames found matching the given criteria."

        rows = []
        for f in frames:
            sources.append(ChatSource(
                frame_id=f.frame_id,
                timestamp=f.timestamp,
                location=f.location,
                description=f.original_description[:120],
            ))
            rows.append({
                "frame_id":    f.frame_id,
                "timestamp":   _fmt_dt(f.timestamp),
                "location":    f.location,
                "altitude":    f.altitude,
                "description": f.original_description[:200],
            })
        return json.dumps(rows, indent=2)

    # ------------------------------------------------------------------ #
    # Tool 3 – search_alerts
    # ------------------------------------------------------------------ #
    def search_alerts(
        severity: Optional[str] = None,
        status: Optional[str] = None,
        alert_type: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        limit = _clamp(limit, 1, 50)
        try:
            q = db.query(Alert)
            if severity:
                q = q.filter(Alert.severity == severity.lower())
            if status:
                q = q.filter(Alert.status == status.lower())
            if alert_type:
                q = q.filter(Alert.alert_type == alert_type)
            alerts = q.order_by(Alert.timestamp.desc()).limit(limit).all()
        except Exception as exc:
            logger.error("search_alerts DB error: %s", exc)
            return f"Database error while searching alerts: {exc}"

        if not alerts:
            return "No alerts found matching the given criteria."

        for a in alerts:
            sources.append(ChatSource(
                event_id=a.event_id,
                frame_id=a.frame_id,
                timestamp=a.timestamp,
                location=a.location,
                description=a.message,
            ))
        return json.dumps([_fmt_alert(a) for a in alerts], indent=2)

    # ------------------------------------------------------------------ #
    # Tool 4 – get_vehicle_history
    # ------------------------------------------------------------------ #
    def get_vehicle_history(
        color: Optional[str] = None,
        make: Optional[str] = None,
        model: Optional[str] = None,
        category: Optional[str] = None,
    ) -> str:
        if not any([color, make, model, category]):
            return "Please provide at least one of: color, make, model, or category."

        try:
            from app.services.event_processor import get_vehicle_history as _gvh
            appearances = _gvh(color=color, make=make, model=model,
                               category=category, db=db)
        except Exception as exc:
            logger.error("get_vehicle_history error: %s", exc)
            return f"Error retrieving vehicle history: {exc}"

        if not appearances:
            parts = [p for p in [color, make, model, category] if p]
            return f"No vehicle history found for: {' '.join(parts)}."

        rows = []
        for a in appearances:
            sources.append(ChatSource(
                event_id=a.event_id,
                frame_id=a.frame_id,
                timestamp=a.timestamp,
                location=a.location,
                description=a.description,
            ))
            rows.append({
                "event_id":   a.event_id,
                "frame_id":   a.frame_id,
                "timestamp":  _fmt_dt(a.timestamp),
                "location":   a.location,
                "color":      a.color,
                "make":       a.make,
                "model":      a.model,
                "category":   a.category,
                "confidence": a.confidence,
            })
        return (
            f"Found {len(rows)} appearance(s):\n"
            + json.dumps(rows, indent=2)
        )

    # ------------------------------------------------------------------ #
    # Tool 5 – get_daily_security_summary
    # ------------------------------------------------------------------ #
    def get_daily_security_summary(date_str: Optional[str] = None) -> str:
        try:
            if date_str:
                try:
                    target = date.fromisoformat(date_str)
                except ValueError:
                    return f"Invalid date format: {date_str!r}. Use YYYY-MM-DD."
                start = datetime(target.year, target.month, target.day, 0, 0, 0)
                end   = datetime(target.year, target.month, target.day, 23, 59, 59)
            else:
                # Find the most recent date that has records
                latest_frame = db.query(Frame).order_by(Frame.timestamp.desc()).first()
                if not latest_frame:
                    return "No drone footage has been processed yet."
                ts = latest_frame.timestamp
                start = datetime(ts.year, ts.month, ts.day, 0, 0, 0)
                end   = datetime(ts.year, ts.month, ts.day, 23, 59, 59)

            frames  = db.query(Frame).filter(
                Frame.timestamp >= start, Frame.timestamp <= end
            ).count()
            events  = db.query(Event).filter(
                Event.timestamp >= start, Event.timestamp <= end
            ).all()
            alerts  = db.query(Alert).filter(
                Alert.timestamp >= start, Alert.timestamp <= end
            ).all()

            by_type: Dict[str, int] = {}
            for ev in events:
                by_type[ev.event_type] = by_type.get(ev.event_type, 0) + 1

            by_sev: Dict[str, int] = {}
            for a in alerts:
                by_sev[a.severity] = by_sev.get(a.severity, 0) + 1
                sources.append(ChatSource(
                    event_id=a.event_id,
                    frame_id=a.frame_id,
                    timestamp=a.timestamp,
                    location=a.location,
                    description=a.message,
                ))

            summary = {
                "date":           start.date().isoformat(),
                "frames_processed": frames,
                "total_events":   len(events),
                "event_breakdown": by_type,
                "total_alerts":   len(alerts),
                "alert_severity_breakdown": by_sev,
                "unread_alerts":  sum(1 for a in alerts if a.status == "unread"),
            }
        except Exception as exc:
            logger.error("get_daily_security_summary error: %s", exc)
            return f"Error generating daily summary: {exc}"

        if frames == 0:
            return f"No drone footage was recorded on {start.date().isoformat()}."

        return json.dumps(summary, indent=2)

    # ------------------------------------------------------------------ #
    # Tool 6 – get_location_activity
    # ------------------------------------------------------------------ #
    def get_location_activity(location: str, limit: int = 20) -> str:
        limit = _clamp(limit, 1, 50)
        try:
            from sqlalchemy import func
            events = (
                db.query(Event)
                .filter(func.lower(Event.location) == location.lower())
                .order_by(Event.timestamp.desc())
                .limit(limit)
                .all()
            )
        except Exception as exc:
            logger.error("get_location_activity error: %s", exc)
            return f"Database error while querying location activity: {exc}"

        if not events:
            return f"No activity recorded at '{location}'."

        for ev in events:
            sources.append(ChatSource(
                event_id=ev.id,
                frame_id=ev.frame_id,
                timestamp=ev.timestamp,
                location=ev.location,
                description=ev.description,
            ))
        return (
            f"Found {len(events)} event(s) at '{location}':\n"
            + json.dumps([_fmt_event(e) for e in events], indent=2)
        )

    # ------------------------------------------------------------------ #
    # Assemble StructuredTool objects
    # ------------------------------------------------------------------ #
    tools = [
        StructuredTool.from_function(
            func=search_events,
            name="search_events",
            description=(
                "Search the security event history. Use to answer questions about "
                "what happened, what objects were detected, activities at locations, "
                "or events within a time window. All parameters are optional."
            ),
            args_schema=SearchEventsInput,
        ),
        StructuredTool.from_function(
            func=search_frames,
            name="search_frames",
            description=(
                "Search processed drone frames. Use to answer questions about "
                "specific frames, locations, or time periods. All parameters optional."
            ),
            args_schema=SearchFramesInput,
        ),
        StructuredTool.from_function(
            func=search_alerts,
            name="search_alerts",
            description=(
                "Search security alerts. Use to answer questions about suspicious "
                "activity alerts, their severity, or current status. "
                "All parameters optional."
            ),
            args_schema=SearchAlertsInput,
        ),
        StructuredTool.from_function(
            func=get_vehicle_history,
            name="get_vehicle_history",
            description=(
                "Look up all historical appearances of a vehicle matching the "
                "given attributes. Use when asked how many times a vehicle was "
                "seen, or when tracking a specific vehicle across frames. "
                "Provide at least one of: color, make, model, category."
            ),
            args_schema=GetVehicleHistoryInput,
        ),
        StructuredTool.from_function(
            func=get_daily_security_summary,
            name="get_daily_security_summary",
            description=(
                "Return a summary of all security activity for a given date "
                "(YYYY-MM-DD). If no date is provided, uses the most recent "
                "date with records. Use for 'what happened today/yesterday' questions."
            ),
            args_schema=GetDailySummaryInput,
        ),
        StructuredTool.from_function(
            func=get_location_activity,
            name="get_location_activity",
            description=(
                "Return all security events recorded at a specific location. "
                "Use for 'what happened at [place]' questions."
            ),
            args_schema=GetLocationActivityInput,
        ),
    ]
    return tools


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an AI Security Analyst for a property monitored by a drone surveillance system.
Your role is to answer questions from the property owner about historical security footage.

STRICT RULES:
1. You MUST call a database tool before answering any question about events, vehicles,
   people, alerts, or activities. Never answer from memory alone.
2. If the database returns no records, say "No records found" — do not invent events.
3. Always cite specific timestamps, locations, and frame IDs from the tool results.
4. When describing vehicles, only state attributes that appear in the data.
5. For follow-up questions, use context from the conversation but still verify with tools.
6. Keep answers concise and factual. Avoid speculation.
7. If asked about something outside your database tools, say you can only answer
   questions about recorded drone footage.

You have access to these tools:
- search_events: search detected security events by type, location, time
- search_frames: search raw drone frames by location or time
- search_alerts: search generated security alerts by severity or type
- get_vehicle_history: track a specific vehicle across all frames
- get_daily_security_summary: get a full summary of a day's activity
- get_location_activity: get all events at a specific location

Always use tools to retrieve data before forming your answer.
"""


# ---------------------------------------------------------------------------
# Agent executor factory  (one per request; memory is per session)
# ---------------------------------------------------------------------------

def _get_memory(session_id: str) -> Any:
    """Return (or create) the ConversationBufferWindowMemory for this session."""
    from langchain.memory import ConversationBufferWindowMemory

    if session_id not in _session_memories:
        _session_memories[session_id] = ConversationBufferWindowMemory(
            k=_MEMORY_WINDOW_K,
            memory_key="chat_history",
            return_messages=True,
        )
        logger.debug("Created new memory for session %s", session_id)
    return _session_memories[session_id]


def _get_agent_executor(db: Session, session_id: str, sources: List[ChatSource]) -> Any:
    """
    Build and return an AgentExecutor for one request.

    Uses Groq (qwen/qwen3-8b by default) as the chatbot LLM.
    Gemini is still used separately for frame analysis only.

    The executor is not cached — it is cheap to build and must hold a
    reference to the per-request `db` session and `sources` list.
    The memory object IS cached (per session_id).
    """
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_groq import ChatGroq

    from app.config import get_settings
    settings = get_settings()

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")

    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.6,
        max_tokens=2048,
    )

    tools = _make_tools(db, sources)
    memory = _get_memory(session_id)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=6,
        early_stopping_method="generate",
    )
    return executor


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    session_id: str,
    db: Session,
) -> ChatResponse:
    """
    Answer a natural-language security question using the LangChain agent.

    Args:
        question:   The property owner's question.
        session_id: Conversation identifier for memory continuity.
        db:         SQLAlchemy session providing access to the event database.

    Returns:
        ChatResponse with answer text and source references.

    Raises:
        RuntimeError: if GEMINI_API_KEY is missing.
        Exception:    any unrecoverable agent error (caught by the route handler).
    """
    logger.info("Chat request — session=%s question=%r", session_id, question[:80])

    sources: List[ChatSource] = []

    try:
        executor = _get_agent_executor(db, session_id, sources)
        result = executor.invoke({"input": question})
        answer: str = result.get("output", "I could not generate a response.")
    except RuntimeError:
        raise
    except Exception as exc:
        logger.exception("Agent execution error: %s", exc)
        raise RuntimeError(f"Agent execution failed: {exc}") from exc

    # Deduplicate sources by event_id + frame_id
    seen: set = set()
    unique_sources: List[ChatSource] = []
    for src in sources:
        key = (src.event_id, src.frame_id)
        if key not in seen:
            seen.add(key)
            unique_sources.append(src)

    logger.info(
        "Chat response — session=%s answer_len=%d sources=%d",
        session_id, len(answer), len(unique_sources),
    )

    return ChatResponse(
        answer=answer,
        sources=unique_sources[:10],   # cap at 10 to keep response lean
        session_id=session_id,
    )


def clear_session(session_id: str) -> None:
    """Remove a session's conversation memory. Used in tests and admin ops."""
    _session_memories.pop(session_id, None)

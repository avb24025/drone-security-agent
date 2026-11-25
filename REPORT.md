# Drone Security Analyst Agent — Architecture and Design Report

---

## 1. Architecture and Design Decisions

### Separation of concerns

The backend is divided into five distinct layers, each with a single responsibility:

| Layer | Module | Does |
|---|---|---|
| AI Analysis | `gemini_service.py` | Calls Gemini, parses JSON, validates with Pydantic |
| Event Processing | `event_processor.py` | Persists frames and events, manages cross-frame context |
| Alert Engine | `alert_engine.py` | Evaluates rules, writes alerts, deduplicates |
| Agent | `agent.py` | LangChain tools, memory, answer synthesis |
| API | `app/api/` | Routes only — no business logic |

Nothing crosses these boundaries in the wrong direction. The alert engine never calls Gemini. The chatbot never writes frames. The API routes never query the database directly — they call service functions.

### Transaction ownership

Every FastAPI route that writes data owns exactly one transaction. The service functions (`process_frame`, `evaluate`) call `db.flush()` to get primary keys without committing. The route commits after all three steps (Gemini → event processor → alert engine) succeed, or rolls back if any step fails. This keeps each frame processing operation atomic.

### Idempotency

Reprocessing the same `frame_id` is safe. The event processor does a SELECT before INSERT — if the frame exists, it updates it and deletes its old events before creating new ones. Alert deduplication uses a SHA-256 keyed on `rule:frame_id:event_id`, stored in a UNIQUE column, so the database itself prevents duplicates at the constraint level.

---

## 2. Gemini Model Selection

**Model chosen: `gemini-1.5-flash`**

Reasons:
- Available on the free tier with a generous rate limit, making it suitable for local development and demos.
- Fast enough for sequential frame processing without noticeable latency.
- Supports `response_mime_type="application/json"` in its `GenerationConfig`, which instructs the model to return raw JSON without markdown fences. This eliminates brittle string parsing.
- The 1 million token context window is far larger than needed — even a full batch of 20 frames fits comfortably.

Temperature is set to 0.1 for frame analysis. Lower temperature produces more consistent, deterministic object identifications and avoids the model inventing vehicle attributes that were not in the description. The chatbot uses the same temperature to keep answers factual.

### Prompt design

The frame analysis prompt has three parts:

1. A **system instruction** that establishes the analyst role, explicitly forbids hallucination of vehicle attributes, and states that the input is simulated text rather than actual video.
2. **Frame metadata** (ID, timestamp, location, altitude, coordinates) so the model can use context in its classification.
3. The **JSON output schema** described inline, which is more reliable than the SDK's `response_schema=` parameter (which conflicts with Pydantic v2 Optional fields in this SDK version).

---

## 3. Agent Design

The LangChain agent uses `create_tool_calling_agent` with an `AgentExecutor`. This was chosen over a custom ReAct loop because:
- Tool calling is native to Gemini's API, making the round-trip more reliable than parsing text-formatted tool calls.
- `AgentExecutor` handles retries, iteration limits, and parsing errors through `handle_parsing_errors=True`.
- The agent is stateless between requests — memory is stored separately and injected via `MessagesPlaceholder`.

**The critical design constraint** is that the LLM never executes SQL. It calls six named Python functions (StructuredTools) with typed Pydantic input schemas. Each function runs a parameterised SQLAlchemy query and returns JSON. This prevents prompt injection from database content affecting query behaviour.

The agent is built per-request (not cached) so it holds a reference to the per-request database session. Memory is cached per `session_id` in a module-level dict, giving continuity across questions in the same conversation without leaking data between sessions.

### Tool design

Each tool follows the same pattern:
1. Validate and clamp inputs (e.g. limit capped at 50).
2. Run a parameterised SQLAlchemy query.
3. Catch any database exception and return a human-readable error string rather than raising — the LLM can tell the user what went wrong.
4. Append `ChatSource` entries to a shared list that is returned with the final answer.

The `get_daily_security_summary` tool defaults to the most recent date with records when no date is provided. This means "what happened today?" works correctly even if the simulated data is from a fixed date (2026-11-24).

---

## 4. Context Management

Context across frames is maintained entirely in the database — there is no in-memory state between requests.

### Vehicle context

When a new frame is processed, `event_processor._find_prior_vehicle_appearances()` queries all existing vehicle events with `frame_id < current_frame_id` and scores each one against the new vehicle's attributes using a point-based matcher:

- Make match: 2 points
- Model match: 2 points
- Color match: 1 point
- Category match: 1 point

Score ≥ 2 → CONFIRMED match. Score = 1 → POSSIBLE match. Score 0 → no match.

This threshold prevents false positives. A red Toyota and a red Ford both have red in common, but the Toyota will not trigger a "repeated vehicle" alert for the Ford because they differ on make (2 points) which outweighs the colour agreement (1 point).

The `ProcessingContext` dataclass carries the results to the alert engine without a second database round-trip.

### Person loitering context

The loitering rule requires at least `loitering_min_frames` (default: 2) person observations at the same location during restricted hours. The first sighting is never enough. This prevents a single standing person from triggering an alert — repeated presence is required.

Restricted hours wrap across midnight correctly: `hour >= 22 OR hour < 6`.

---

## 5. Frame Indexing

Every processed frame is stored in the `frames` table with:
- Full telemetry (location, altitude, GPS coordinates, timestamp).
- The original text description exactly as submitted.
- The complete Gemini analysis as a JSON blob (excluding `raw_response` to keep the column lean).

Events are stored separately in the `events` table with foreign keys back to `frames.frame_id`. This normalisation allows queries like "all vehicle events at the Main Gate" without loading frame descriptions.

Compound indexes on `(object_type, location)` and `(severity, status)` cover the most common dashboard query patterns.

The SQLite WAL (Write-Ahead Logging) mode is enabled on connection so read queries from the dashboard do not block writes from frame processing.

---

## 6. Alert Rules

### Rule design principles

1. **Conservative thresholds** — rules do not fire on a single ambiguous signal.
2. **Location-aware** — restricted areas and restricted hours are configurable, not hardcoded.
3. **Deduplication by design** — the `dedup_key` UNIQUE constraint means the database itself enforces no-duplicate semantics, independent of application logic.
4. **Rule isolation** — the `evaluate()` function wraps each rule in a try/except. One rule raising an exception does not prevent the other three from running.

### Configurable parameters

| Parameter | Default | Effect |
|---|---|---|
| `RESTRICTED_HOUR_START` | 22 | Loitering and restricted-hour checks start here |
| `RESTRICTED_HOUR_END` | 6 | Restricted window ends here |
| `LOITERING_MIN_FRAMES` | 2 | Minimum sightings before loitering alert fires |
| `AUTHORIZED_VEHICLES` | blue Ford F150, white Toyota Camry | Suppresses unauthorized alert |
| Restricted areas (code) | garage, server room, storage shed | Triggers intrusion alert by location name |

---

## 7. Test Results

All 242 tests pass with zero failures.

| Module tested | Tests | Key scenarios verified |
|---|---|---|
| Drone simulator | 30 | 20-frame dataset integrity, all accessor functions, immutability |
| Gemini service | 30 | Vehicle/person extraction, null attribute preservation, all error types (blocked, API error, malformed JSON, missing key) |
| Event processor | 45 | Frame upsert, one-to-many events, vehicle matching confidence levels, person loitering counting, all search filters |
| Alert engine | 46 | All four rules, boundary conditions (hour 22, hour 6), configurable thresholds, dedup, status updates |
| API integration | 50 | All 15 endpoints, full pipeline (truck→events→no-alert, intrusion→critical, repeated vehicle, loitering sequence), 422 validation |
| LangChain agent | 41 | All six tools with real in-memory SQLite, answer_question flow, source deduplication, memory lifecycle |

Tests do not make real Gemini API calls. The `gemini_service` module is mocked at the `analyze_frame` function level using `unittest.mock.patch`. LangChain's `AgentExecutor` is mocked at the `_get_agent_executor` level for unit tests; tool functions are tested directly against real in-memory SQLite for integration coverage.

---

## 8. Limitations and Future Improvements

### Current limitations

**Simulated data only.** The system processes text descriptions, not actual video frames. Gemini's vision capabilities are available in the SDK but not wired up — adding image support would require the drone simulator to supply base64-encoded images alongside descriptions.

**Sequential batch processing.** Frames in a batch are processed one at a time. For large datasets this is slow. Parallel processing with async SQLAlchemy would improve throughput but adds complexity.

**In-process memory for chat sessions.** Session memories are stored in a Python dict. Restarting the server clears all conversation history. A Redis store or database-backed memory would persist sessions across restarts.

**Single-process SQLite.** SQLite WAL mode handles concurrent reads well but write throughput is limited. For production use, a PostgreSQL database with connection pooling would scale better.

**No authentication.** The API has no authentication layer. Anyone who can reach port 8000 can read all alerts and frames. For a real deployment, OAuth2 or API key authentication would be required.

### Potential improvements

- **Real image analysis** — Wire up Gemini Vision to process actual frame images from the drone camera, replacing the text-description simulation.
- **WebSocket streaming** — Push new alerts to the dashboard in real time rather than polling every 15 seconds.
- **Alert notifications** — Send email or SMS when a critical alert fires, using SendGrid or Twilio.
- **Vehicle registry** — A dedicated vehicle registration table with plate numbers and owner names, enabling more precise authorisation checks.
- **Multi-property support** — Namespaced data per property with user accounts.
- **Confidence calibration** — The vehicle matching scorer uses fixed point values. A learned similarity model (e.g. embedding-based) would generalise better across different description styles.
- **Audit log** — Track when alerts are read and resolved, and by whom.

---

## 9. How AI-Assisted Coding Helped Development

**Kiro** (the AI coding assistant used throughout this project) contributed in several concrete ways:

**Incremental phase-by-phase implementation.** Rather than generating the entire codebase at once, the assistant built each component, ran its tests, diagnosed failures, and fixed them before moving to the next phase. This produced working, tested code at every checkpoint.

**SQLAlchemy 2.0 test infrastructure.** The most significant technical challenge was that SQLAlchemy 2.0 removed the `sessionmaker(bind=engine)` routing mechanism that SQLAlchemy 1.x relied on. When 44 API tests failed with "no such table", the assistant diagnosed the root cause (session not bound to the in-memory test connection), identified the correct SQLAlchemy 2.0 test pattern (connection-level session binding with SAVEPOINT restart), and implemented it correctly.

**LangChain tool design.** The assistant designed the six database tools to follow a consistent pattern — parameterised queries, human-readable error strings rather than exceptions, source accumulation via closure — which made all 41 agent tests pass on the first run after the memory-test fixture was corrected.

**Prompt engineering.** The Gemini system instruction and JSON schema description were crafted to prevent hallucination of vehicle attributes, explicitly state the simulated nature of the input, and produce consistent low-temperature output. The instructions were verified against the test suite's "null attribute preservation" test.

**Security-aware code review.** When the `.env` file was inspected and found to contain a real API key, the assistant flagged it immediately with specific remediation steps.

# Drone Security Analyst Agent

A full-stack AI-powered property security monitoring system. Simulated drone frames are analyzed by Groq AI, structured security events are extracted, automatic alerts are generated based on configurable rules, and a conversational AI analyst answers questions grounded in the recorded data.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Features](#3-features)
4. [Technology Stack](#4-technology-stack)
5. [System Architecture and Design Decisions](#5-system-architecture-and-design-decisions)
6. [Database Schema](#6-database-schema)
7. [API Documentation](#7-api-documentation)
8. [Prerequisites](#8-prerequisites)
9. [Installation](#9-installation)
10. [Environment Configuration](#10-environment-configuration)
11. [Running the Backend](#11-running-the-backend)
12. [Running the Frontend](#12-running-the-frontend)
13. [Processing Simulated Drone Frames](#13-processing-simulated-drone-frames)
14. [Running Tests](#14-running-tests)
15. [AI Tools Integrated](#15-ai-tools-integrated)

---

## 1. Project Overview

The Drone Security Analyst Agent demonstrates how large language models can automate property security monitoring. The system processes simulated drone footage, understands what is happening in each frame, tracks objects across multiple frames, fires alerts when security rules are violated, and provides a natural-language chatbot that answers questions about historical footage.

Everything runs locally. No Docker, no external database server, no cloud infrastructure beyond a Groq API key.

---

## 2. Problem Statement

A property owner wants autonomous security monitoring without manually reviewing camera feeds. The system must:

- Analyze each drone frame and identify what is happening — vehicles, people, activities.
- Maintain context across frames so the same vehicle appearing at 8 AM and 2 PM is recognized as one object.
- Fire security alerts automatically when rules are violated, independent of any user interaction.
- Let the owner ask plain-English questions and receive factual answers backed by actual database records.

---

## 3. Features

### Frame Analysis
- Sends frame descriptions to Groq AI (`qwen/qwen3.8-27b`) for structured JSON analysis.
- Extracts objects (vehicles with color/make/model, persons), activities, and event classification.
- All AI output is validated with Pydantic — absent vehicle attributes are stored as null, never invented.

### Cross-Frame Context Management
- Vehicles are matched across frames using a point-based attribute scorer (make=2pts, model=2pts, color=1pt, category=1pt).
- Score ≥2 → CONFIRMED match. Score =1 → POSSIBLE match. This prevents false positives from colour-only overlap.
- Person loitering is tracked by counting prior observations at the same location during restricted hours.
- Reprocessing the same frame is fully idempotent.

### Automatic Security Alert Engine (4 rules)

| Rule | Trigger | Severity |
|---|---|---|
| Suspicious Loitering | Person at restricted location during restricted hours, seen ≥ N times | High |
| Unauthorized Vehicle | Vehicle not in the authorised list, or explicitly unidentified | High |
| Repeated Vehicle Appearance | Same vehicle confirmed seen more than once | Medium |
| Restricted Area Intrusion | Person or vehicle inside a configured restricted zone | Critical |

Alerts are deduplicated using a SHA-256 keyed `dedup_key` stored in a UNIQUE database column.

### AI Security Analyst Chatbot
- LangChain agent with six database retrieval tools — the LLM never executes raw SQL.
- Always queries the database before answering — no invented events.
- Per-session conversational memory with a 10-exchange window.
- Returns source references (frame ID, timestamp, location) alongside every answer.

### React Dashboard
- **Overview** — live stat cards, severity breakdown, monitoring status.
- **Alerts** — filterable list with inline read/resolve controls.
- **Event History** — searchable table with object type, location, and event type filters.
- **Frame Explorer** — simulator panel + indexed frame table with right-side detail drawer.
- **AI Analyst** — conversational chatbot with suggested questions.

---

## 4. Technology Stack

### Backend

| Component | Technology |
|---|---|
| Web framework | FastAPI 0.111 |
| ORM | SQLAlchemy 2.0 |
| Database | Neon PostgreSQL (via SQLAlchemy 2.0 + psycopg2) |
| Data validation | Pydantic v2 |
| AI — frame analysis & chatbot | Groq SDK (`groq`) + `qwen/qwen3.8-27b` |
| AI agent framework | LangChain 0.2 + `langchain-groq` |
| Testing | Pytest + FastAPI TestClient |
| Settings | pydantic-settings |

### Frontend

| Component | Technology |
|---|---|
| Framework | React 18 |
| Build tool | Vite 5 |
| Styling | Tailwind CSS 3 |
| HTTP client | Axios |
| Icons | Lucide React |
| State | React Hooks |

---

## 5. System Architecture and Design Decisions

### Overall structure

```
┌──────────────────────────────────────────────────────────────┐
│                      React Dashboard                         │
│  Overview │ Alerts │ Events │ Frame Explorer │ AI Analyst    │
└───────────────────────────┬──────────────────────────────────┘
                            │  REST (Axios, 15s polling)
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                        FastAPI                               │
│  /api/frames  /api/events  /api/alerts  /api/chat  /api/dashboard │
└──────┬───────────────┬───────────────────────┬──────────────┘
       │               │                       │
       ▼               ▼                       ▼
┌──────────────┐ ┌─────────────────┐  ┌──────────────────────┐
│ Frame        │ │ Event Processor │  │  LangChain Agent     │
│ Analysis     │ │ (context mgmt)  │  │  (Groq LLM + tools)  │
│ (Groq AI)    │ └────────┬────────┘  └──────────┬───────────┘
└──────────────┘          │                      │
                          ▼                      │
                  ┌───────────────┐              │
                  │ Alert Engine  │              │
                  │ (4 rules)     │              │
                  └───────┬───────┘              │
                          │                      │
                          ▼                      ▼
                  ┌──────────────────────────────────┐
                  │      Neon PostgreSQL Database    │
                  │  frames │ events │ alerts         │
                  └──────────────────────────────────┘
```

### Request flow for a single frame

```
POST /api/frames/process
  1. frame_analysis_service.analyze_frame()   → structured AI output
  2. event_processor.process_frame()          → events saved, context checked
  3. alert_engine.evaluate()                  → alerts saved (deduplicated)
  4. db.commit()                              → one atomic transaction
  5. return FrameProcessingResult
```

### Chat request flow

```
POST /api/chat
  1. LangChain agent receives the question
  2. Groq LLM decides which tool(s) to call
  3. Tools execute parameterised SQLAlchemy queries
  4. Groq synthesises a grounded answer from tool results
  5. return ChatResponse(answer, sources, session_id)
```

### Key design decisions

**Single transaction per frame**
The frame analysis, event extraction, and alert evaluation all run inside one database transaction. The FastAPI route commits after all three succeed, or rolls back if any step fails. This keeps each frame processing operation atomic.

**Idempotent reprocessing**
The event processor does a SELECT before INSERT on the frames table. If the frame already exists, it updates it and deletes its old events before creating new ones. This means reprocessing the same frame never creates duplicate rows.

**Alert deduplication at the database level**
Every alert carries a `dedup_key = SHA256("rule:frame_id:event_id")` stored in a UNIQUE column. The database constraint enforces no-duplicate semantics independently of application logic. If the same rule fires twice for the same event, the second INSERT fails silently and returns None.

**Tool-based chatbot — no raw SQL**
The LangChain agent calls six named Python functions (StructuredTools) with typed Pydantic input schemas. The LLM never constructs or executes SQL. This prevents prompt injection from database content affecting query behaviour and keeps the agent deterministic.

**Conservative vehicle matching**
A colour-only match between two vehicles scores 1 point (POSSIBLE). A match on make+colour scores 3 points (CONFIRMED). The repeated-vehicle alert only fires on CONFIRMED matches. This prevents alerting because two different red cars were seen on the same day.

**Separation of concerns**
- Frame analysis understands the input.
- Event processor extracts structure and maintains context.
- Alert engine evaluates rules.
- LangChain agent handles conversation.
- FastAPI routes orchestrate but contain no business logic.
- React handles presentation only.

---

## 6. Database Schema

### frames

| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| frame_id | INTEGER UNIQUE | Drone-assigned frame number |
| timestamp | DATETIME | Capture time (indexed) |
| location | VARCHAR(255) | Location label (indexed) |
| altitude | FLOAT | Metres |
| latitude | FLOAT | GPS latitude |
| longitude | FLOAT | GPS longitude |
| original_description | TEXT | Raw text submitted with the frame |
| gemini_analysis | JSON | Full structured AI output blob |
| processed_at | DATETIME | Server-side write time |

### events

| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| frame_id | INTEGER FK→frames | Cascade delete |
| timestamp | DATETIME | Indexed |
| location | VARCHAR(255) | Indexed |
| object_type | VARCHAR(100) | vehicle / person / unknown (indexed) |
| object_attributes | JSON | color, make, model, category |
| activity | VARCHAR(255) | What the object was doing |
| event_type | VARCHAR(100) | vehicle_entry, loitering, etc. (indexed) |
| description | TEXT | One-sentence AI summary |

### alerts

| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| event_id | INTEGER FK→events | SET NULL on delete |
| alert_type | VARCHAR(100) | Rule name (indexed) |
| severity | VARCHAR(20) | critical / high / medium / low |
| message | TEXT | Human-readable alert text |
| timestamp | DATETIME | Indexed |
| location | VARCHAR(255) | Denormalised for fast queries |
| frame_id | INTEGER | Denormalised for fast queries |
| dedup_key | VARCHAR(512) UNIQUE | Prevents duplicate alerts |
| status | VARCHAR(20) | unread / read / resolved (indexed) |

---

## 7. API Documentation

Interactive Swagger UI is available at `http://localhost:8000/docs` when the server is running.

### Frames

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/frames/process` | Process one frame end-to-end |
| POST | `/api/frames/batch` | Process multiple frames sequentially |
| GET | `/api/frames` | List indexed frames (filterable by location, time) |
| GET | `/api/frames/{frame_id}` | One frame with its extracted events |
| GET | `/api/frames/simulator/frames` | Full 20-frame simulated dataset |
| GET | `/api/frames/simulator/summary` | Lightweight frame list (no descriptions) |

### Events

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/events` | List events (filter by object_type, location, event_type, time) |
| GET | `/api/events/summary` | Aggregate counts by type and location |
| GET | `/api/events/{event_id}` | Single event |

### Alerts

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/alerts` | List alerts (filter by severity, status, type) |
| GET | `/api/alerts/stats` | Counts by severity, status, type |
| GET | `/api/alerts/{alert_id}` | Single alert |
| PATCH | `/api/alerts/{alert_id}` | Update status: unread → read → resolved |

### Chat

**POST** `/api/chat`

Request:
```json
{
  "question": "What vehicles entered the property today?",
  "session_id": "session_abc123"
}
```

Response:
```json
{
  "answer": "Two vehicles were recorded today. The blue Ford F150 appeared twice...",
  "sources": [
    {
      "event_id": 1,
      "frame_id": 2,
      "timestamp": "2026-11-24T08:15:00",
      "location": "Main Gate"
    }
  ],
  "session_id": "session_abc123"
}
```

### Dashboard

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/dashboard/summary` | Overview stats: frame count, event count, alert counts |

---

## 8. Prerequisites

- **Python 3.10 or higher**
- **Node.js 18 or higher**
- **A Groq API key** — free tier, no billing required

Obtain a Groq API key:
1. Go to [https://console.groq.com/keys](https://console.groq.com/keys)
2. Create a free account
3. Click **Create API Key**
4. Copy the key — it starts with `gsk_`

---

## 9. Installation

### Clone the repository

```bash
git clone <repository-url>
cd drone-security-agent
```

### Backend

```bash
cd Backend
pip install -r requirements.txt
```

### Frontend

```bash
cd Frontend
npm install
```

---

## 10. Environment Configuration

Create a file named `.env` inside the `Backend/` folder. This file is listed in `.gitignore` and will never be committed.

```
# Backend/.env

GROQ_API_KEY=gsk_your_actual_key_here
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require&channel_binding=require
```

Two variables are required. All others have working defaults:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Groq model for analysis and chat |
| `FRONTEND_ORIGIN` | `http://localhost:5173` | CORS origin |
| `RESTRICTED_HOUR_START` | `22` | Start of restricted hours (10 PM) |
| `RESTRICTED_HOUR_END` | `6` | End of restricted hours (6 AM) |
| `LOITERING_MIN_FRAMES` | `2` | Sightings before loitering alert fires |
| `AUTHORIZED_VEHICLES` | `blue Ford F150,white Toyota Camry` | Vehicles exempt from unauthorized alert |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

**Getting a Neon database URL:**
1. Go to [https://console.neon.tech](https://console.neon.tech)
2. Create a free project
3. Go to **Connection Details** → copy the **Pooled connection string**
4. It looks like: `postgresql://user:pass@host-pooler.region.aws.neon.tech/dbname?sslmode=require&channel_binding=require`

See `Backend/.env.example` for a complete annotated example.

---

## 11. Running the Backend

Open a terminal in the `Backend/` directory:

```bash
cd Backend
python -m uvicorn app.main:app --reload --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

The Neon PostgreSQL database tables are created automatically on first run via SQLAlchemy's `create_all()`.

**Verify the server is running:**

Open `http://localhost:8000/health` in a browser. You should see:
```json
{"status": "ok", "service": "drone-security-agent"}
```

Open `http://localhost:8000/docs` for the interactive API explorer.

**Common errors:**

| Error | Fix |
|---|---|
| `uvicorn: command not found` | Use `python -m uvicorn` instead of `uvicorn` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `GROQ_API_KEY is not set` | Create `Backend/.env` with your key |
| `Port 8000 already in use` | Add `--port 8001` and update `FRONTEND_ORIGIN` in `.env` |
| Must run from `Backend/` | Always `cd Backend` before running uvicorn |

---

## 12. Running the Frontend

Open a **second terminal** (keep the backend running in the first):

```bash
cd Frontend
npm run dev
```

The dashboard opens at `http://localhost:5173`.

Vite automatically proxies all `/api` requests to `http://localhost:8000`, so no manual CORS configuration is needed during development.

**If the dashboard shows "Backend offline"** — the FastAPI server is not running. Start it in a separate terminal as described above.

---

## 13. Processing Simulated Drone Frames

The system includes 20 pre-built drone frames covering a range of security scenarios. No real drone hardware is needed.

### From the dashboard (recommended)

1. Open `http://localhost:5173`
2. Navigate to **Frame Explorer**
3. In the **Drone Simulator** panel, click **All** to select all 20 frames
4. Click **Process frames**
5. The AI analyzes each frame and stores the results
6. Navigate to **Alerts** and **Event History** to see what was detected

### From the API

Send a batch request to `http://localhost:8000/docs` → `POST /api/frames/batch`, or use curl:

```bash
curl -X POST http://localhost:8000/api/frames/process \
  -H "Content-Type: application/json" \
  -d '{
    "frame_id": 2,
    "timestamp": "2026-11-24T08:15:00",
    "location": "Main Gate",
    "altitude": 22.0,
    "latitude": 18.5204,
    "longitude": 73.8567,
    "description": "A blue Ford F150 is entering the main gate."
  }'
```

### Simulated scenario coverage

| Frame(s) | Scenario | Expected Alert |
|---|---|---|
| 1, 9 | Person at Main Gate at 02:00 and 02:30 | Suspicious loitering (high) |
| 2 | Blue Ford F150 entering — first appearance | None (authorised) |
| 3 | Normal pedestrian, daytime | None |
| 4 | Red sedan parked near Garage | Restricted area intrusion (critical) |
| 5 | Unidentified dark vehicle entering | Unauthorized vehicle (high) |
| 6 | Person inside restricted Garage | Restricted area intrusion (critical) |
| 7, 14 | Blue Ford F150 reappearing | Repeated vehicle (medium) |
| 8 | Routine delivery at Loading Bay | None |
| 11 | Silver SUV, no plates, Side Gate | Unauthorized vehicle (high) |
| 12, 16 | Authorised vehicles exiting | None |
| 13 | Person in Server Room at 22:30 | Restricted area intrusion (critical) |
| 17, 20 | Unknown person during restricted hours | Suspicious loitering (high) |
| 19 | Routine patrol flyover | None |

### Clearing the database

To reset all data and start fresh (drops and recreates all tables on Neon):

```bash
cd Backend
python -c "
from dotenv import load_dotenv
load_dotenv()
from app.database import engine
from app.models.db_models import Base
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)
print('Database cleared.')
"
```

---

## 14. Running Tests

All tests use isolated in-memory databases and mocked Groq API calls. No API key or Neon connection is needed to run tests.

```bash
cd Backend
python -m pytest tests/ -v
```

Run a single file:
```bash
python -m pytest tests/test_alert_engine.py -v
python -m pytest tests/test_api.py -v
python -m pytest tests/test_agent.py -v
```

### Test coverage

| File | Tests | What it covers |
|---|---|---|
| `test_simulator.py` | 30 | Dataset completeness, all accessor functions, immutability |
| `test_gemini_service.py` | 30 | AI output parsing, null attribute preservation, all error types |
| `test_event_processor.py` | 45 | Frame upsert, event creation, vehicle context matching, person loitering |
| `test_alert_engine.py` | 46 | All four rules, boundary conditions, deduplication, status updates |
| `test_api.py` | 50 | All 15 endpoints, full pipeline integration, 422 validation |
| `test_agent.py` | 41 | All six tools, answer_question flow, memory lifecycle |
| **Total** | **242** | Zero failures |

---

## 15. AI Tools Integrated

### Groq — `qwen/qwen3.8-27b`

Groq is used for both frame analysis and the chatbot agent. It was chosen over other providers because:

- **Free tier with no billing setup** — the free Groq tier is generous enough for a development demo.
- **Fast inference** — Groq's LPU hardware returns responses significantly faster than CPU-bound inference, which matters when processing 20 frames sequentially.
- **Tool calling support** — `qwen/qwen3.8-27b` supports LangChain's tool-calling interface, which the chatbot agent depends on.

**Frame analysis usage:**
The `gemini_service.py` module (name kept for schema compatibility) sends a structured prompt to Groq containing the frame metadata and description. The model returns a JSON object identifying objects, activities, event type, and a suspiciousness flag. Temperature is set to 0.6 for factual output. The system instruction explicitly forbids hallucinating vehicle attributes — absent fields must be `null`.

**Chatbot agent usage:**
The LangChain agent uses `ChatGroq` as its LLM. The agent runs a tool-calling loop: Groq decides which of the six database tools to invoke, tools query the Neon PostgreSQL database and return structured JSON, then Groq synthesises a grounded natural-language answer. This keeps every answer traceable to real database records.

**Impact on workflow:**
Switching from Gemini to Groq resolved the daily quota limitation (Gemini free tier: 20 requests/day). Groq's higher free-tier limits allowed processing all 20 simulated frames in a single session during development and testing.

### LangChain 0.2

LangChain provides the agent loop (`create_tool_calling_agent` + `AgentExecutor`) and conversation memory (`ConversationBufferWindowMemory`). Using LangChain meant the tool-calling cycle, iteration limits, error handling, and memory management were handled by the framework rather than hand-coded. This reduced the chatbot implementation to defining the six tool functions and the system prompt.

**Impact on workflow:**
LangChain's `StructuredTool` enforces typed Pydantic schemas on tool inputs, which prevented the LLM from passing malformed arguments to database queries. The `handle_parsing_errors=True` flag on `AgentExecutor` meant malformed tool calls recovered gracefully rather than crashing.

### Kiro (AI Coding Assistant)

Kiro was used as a pair-programming assistant throughout the entire project. Its specific contributions:

- **Phased implementation** — the project was built in 10 incremental phases. Kiro implemented each phase, ran the tests, and fixed failures before moving to the next.
- **SQLAlchemy 2.0 compatibility** — when 44 API integration tests failed with "no such table", Kiro diagnosed the root cause (SA 2.0 removed `sessionmaker(bind=engine)` routing) and implemented the correct connection-level session binding pattern.
- **Model availability discovery** — when model name errors occurred, Kiro queried the Groq API's model list directly and identified the correct model identifier (`qwen/qwen3.8-27b` vs the initially assumed `qwen/qwen3-8b`).
- **Security awareness** — when the `.env` file was inspected and found to contain a real exposed API key, Kiro flagged it immediately with specific steps to revoke and replace it.
- **Test architecture** — all 242 tests were written alongside the implementation, not after. This caught integration issues (transaction boundaries, deduplication edge cases, restricted-hour boundary conditions) during development rather than at the end.

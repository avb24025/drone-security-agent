"""
app/simulator/drone_simulator.py
─────────────────────────────────
Simulated drone telemetry and frame dataset.

Design notes
────────────
• All 20 frames use a fixed date (2026-11-24) so queries by date work
  consistently in tests and demos.
• Timestamps are spread from 00:00 to 23:00 to cover both restricted and
  normal operating hours, exercising every alert rule.
• Locations match the restricted-area and main-gate strings used by the
  alert engine so rule evaluation fires correctly.
• Vehicle attributes (color, make, model) are deliberate so the repeated-
  vehicle and unauthorized-vehicle rules both have something to match.
• The module is stateless: call get_all_frames() or get_frame(n) any number
  of times; the data is never mutated.

Scenario coverage
─────────────────
  Frame  1  – person at Main Gate at 02:00 (restricted hour)        → loitering candidate #1
  Frame  2  – blue Ford F150 entering at Main Gate at 08:00         → vehicle entry
  Frame  3  – normal pedestrian walking during daytime              → no alert
  Frame  4  – red car parked near Garage                            → suspicious park
  Frame  5  – unknown / unidentified vehicle entering               → unauthorized vehicle
  Frame  6  – person in restricted Garage area at 09:00             → restricted intrusion
  Frame  7  – blue Ford F150 seen again at Parking Lot at 14:00     → repeated vehicle
  Frame  8  – normal routine: delivery truck at Loading Bay         → no alert
  Frame  9  – person at Main Gate at 02:30 (restricted hours)       → loitering candidate #2 → alert fires
  Frame 10  – package left unattended at Main Entrance              → suspicious object
  Frame 11  – vehicle with no plates entering at Side Gate          → unauthorized vehicle
  Frame 12  – normal: staff car leaving at 17:00                    → no alert
  Frame 13  – person in Server Room (restricted area) at 22:30      → restricted intrusion
  Frame 14  – blue Ford F150 third appearance at 23:00              → repeated vehicle (3rd)
  Frame 15  – group of people near perimeter fence at 01:00         → loitering candidate
  Frame 16  – white Toyota Camry (authorised) entering at 07:00     → no alert
  Frame 17  – person with backpack at Parking Lot at 03:00          → restricted hour
  Frame 18  – van parked across emergency exit at 11:00             → suspicious parking
  Frame 19  – drone patrol flyover, nothing unusual at 15:00        → no alert
  Frame 20  – motion near storage shed at 00:30                     → restricted hours + unknown person
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import List, Optional

from app.models.schemas import DroneFrame

# ── Property GPS anchor (Pune, MH – fictional coordinates) ───────────────────
_LAT_BASE = 18.5204
_LON_BASE = 73.8567

# ── The 20-frame dataset ──────────────────────────────────────────────────────
_SIMULATED_FRAMES: List[dict] = [
    # 1 ─ Person at Main Gate, restricted hours (02:00)
    {
        "frame_id": 1,
        "timestamp": "2026-11-24T02:00:00",
        "location": "Main Gate",
        "altitude": 20.0,
        "latitude": _LAT_BASE,
        "longitude": _LON_BASE,
        "description": (
            "A person is standing near the main gate. "
            "They appear to be wearing dark clothing and have been stationary for several minutes. "
            "No vehicle is present. Lighting is minimal."
        ),
    },
    # 2 ─ Blue Ford F150 entering (first appearance)
    {
        "frame_id": 2,
        "timestamp": "2026-11-24T08:15:00",
        "location": "Main Gate",
        "altitude": 22.0,
        "latitude": _LAT_BASE + 0.0001,
        "longitude": _LON_BASE + 0.0001,
        "description": (
            "A blue Ford F150 pickup truck is entering the property through the main gate. "
            "The driver is alone. The truck has a standard bed with no visible cargo. "
            "License plate is partially visible."
        ),
    },
    # 3 ─ Normal pedestrian daytime
    {
        "frame_id": 3,
        "timestamp": "2026-11-24T09:30:00",
        "location": "Front Walkway",
        "altitude": 18.0,
        "latitude": _LAT_BASE + 0.0002,
        "longitude": _LON_BASE - 0.0001,
        "description": (
            "A person is walking along the front walkway carrying a laptop bag. "
            "They are moving toward the main building entrance at a normal pace. "
            "This appears to be routine daytime foot traffic."
        ),
    },
    # 4 ─ Red car parked near Garage
    {
        "frame_id": 4,
        "timestamp": "2026-11-24T10:05:00",
        "location": "Garage",
        "altitude": 15.0,
        "latitude": _LAT_BASE - 0.0003,
        "longitude": _LON_BASE + 0.0004,
        "description": (
            "A red sedan is parked directly in front of the garage door. "
            "The vehicle appears to be unattended. No driver or passengers are visible. "
            "The car has been in this position since at least the last patrol sweep."
        ),
    },
    # 5 ─ Unknown / unidentified vehicle entering
    {
        "frame_id": 5,
        "timestamp": "2026-11-24T10:45:00",
        "location": "Main Gate",
        "altitude": 20.0,
        "latitude": _LAT_BASE,
        "longitude": _LON_BASE,
        "description": (
            "An unidentified dark-coloured vehicle is entering the property through the main gate. "
            "The make and model cannot be determined from this angle. "
            "The license plate is obscured. No entry authorisation was observed."
        ),
    },
    # 6 ─ Person in restricted Garage area
    {
        "frame_id": 6,
        "timestamp": "2026-11-24T11:20:00",
        "location": "Garage",
        "altitude": 12.0,
        "latitude": _LAT_BASE - 0.0003,
        "longitude": _LON_BASE + 0.0004,
        "description": (
            "A person is inside the restricted garage area without visible authorisation. "
            "They are examining equipment near the vehicle bay. "
            "No safety vest or identification badge is visible."
        ),
    },
    # 7 ─ Blue Ford F150 reappears (second appearance) at Parking Lot
    {
        "frame_id": 7,
        "timestamp": "2026-11-24T14:00:00",
        "location": "Parking Lot",
        "altitude": 25.0,
        "latitude": _LAT_BASE + 0.0005,
        "longitude": _LON_BASE - 0.0003,
        "description": (
            "The blue Ford F150 pickup truck seen earlier this morning is now parked in the parking lot. "
            "The vehicle matches the same colour, body style and approximate size as the truck that "
            "entered through the main gate at 08:15. The driver is not visible."
        ),
    },
    # 8 ─ Routine delivery truck, no alert
    {
        "frame_id": 8,
        "timestamp": "2026-11-24T14:30:00",
        "location": "Loading Bay",
        "altitude": 20.0,
        "latitude": _LAT_BASE + 0.0010,
        "longitude": _LON_BASE + 0.0008,
        "description": (
            "A large white delivery truck is parked at the loading bay. "
            "Two workers in high-visibility vests are unloading boxes. "
            "A dock staff member is present and appears to be overseeing the delivery. "
            "This looks like routine scheduled delivery activity."
        ),
    },
    # 9 ─ Person at Main Gate again, 02:30 – loitering alert threshold met
    {
        "frame_id": 9,
        "timestamp": "2026-11-24T02:30:00",
        "location": "Main Gate",
        "altitude": 20.0,
        "latitude": _LAT_BASE,
        "longitude": _LON_BASE,
        "description": (
            "The same person observed near the main gate at 02:00 appears to still be in the area. "
            "They are now pacing slowly back and forth near the gate entrance. "
            "No vehicle or companion is present. Behaviour is consistent with loitering."
        ),
    },
    # 10 ─ Unattended package at Main Entrance
    {
        "frame_id": 10,
        "timestamp": "2026-11-24T15:10:00",
        "location": "Main Entrance",
        "altitude": 10.0,
        "latitude": _LAT_BASE + 0.0001,
        "longitude": _LON_BASE - 0.0002,
        "description": (
            "An unattended package or bag has been left at the main entrance. "
            "No person is visible nearby. The package was not present in the previous sweep. "
            "Size and shape are consistent with a medium backpack or duffel bag."
        ),
    },
    # 11 ─ Vehicle with no plates at Side Gate – unauthorized
    {
        "frame_id": 11,
        "timestamp": "2026-11-24T16:00:00",
        "location": "Side Gate",
        "altitude": 20.0,
        "latitude": _LAT_BASE - 0.0010,
        "longitude": _LON_BASE - 0.0010,
        "description": (
            "A silver SUV with no visible license plates is attempting to enter through the side gate. "
            "The driver briefly stopped at the gate before proceeding slowly. "
            "No authorisation credentials or entry badge exchange was observed."
        ),
    },
    # 12 ─ Normal staff car leaving – no alert
    {
        "frame_id": 12,
        "timestamp": "2026-11-24T17:00:00",
        "location": "Main Gate",
        "altitude": 22.0,
        "latitude": _LAT_BASE,
        "longitude": _LON_BASE,
        "description": (
            "A white Toyota Camry is leaving the property through the main gate. "
            "The driver appears to be a staff member based on the visible ID badge. "
            "No unusual behaviour observed. Normal end-of-day departure."
        ),
    },
    # 13 ─ Person in Server Room (restricted) at 22:30
    {
        "frame_id": 13,
        "timestamp": "2026-11-24T22:30:00",
        "location": "Server Room",
        "altitude": 8.0,
        "latitude": _LAT_BASE + 0.0003,
        "longitude": _LON_BASE + 0.0002,
        "description": (
            "An unidentified person has been detected inside the server room, which is a restricted area. "
            "They are accessing equipment near the server racks. "
            "No authorised personnel schedule lists anyone in this area after 22:00."
        ),
    },
    # 14 ─ Blue Ford F150 third appearance at 23:00
    {
        "frame_id": 14,
        "timestamp": "2026-11-24T23:00:00",
        "location": "Main Gate",
        "altitude": 20.0,
        "latitude": _LAT_BASE,
        "longitude": _LON_BASE,
        "description": (
            "The blue Ford F150 pickup truck has reappeared at the main gate late at night. "
            "This is the third observed appearance of a vehicle matching this description today. "
            "The truck is idling just outside the gate with its headlights on."
        ),
    },
    # 15 ─ Group near perimeter fence at 01:00
    {
        "frame_id": 15,
        "timestamp": "2026-11-24T01:00:00",
        "location": "Perimeter Fence",
        "altitude": 30.0,
        "latitude": _LAT_BASE - 0.0015,
        "longitude": _LON_BASE - 0.0012,
        "description": (
            "A group of two to three people are gathered near the perimeter fence on the north side. "
            "They appear to be examining the fence line using a flashlight. "
            "No authorised maintenance work is scheduled for this area overnight."
        ),
    },
    # 16 ─ Authorised white Toyota Camry entering – no alert
    {
        "frame_id": 16,
        "timestamp": "2026-11-24T07:00:00",
        "location": "Main Gate",
        "altitude": 22.0,
        "latitude": _LAT_BASE,
        "longitude": _LON_BASE,
        "description": (
            "A white Toyota Camry is entering the property through the main gate. "
            "The driver displays an access badge at the gate reader. "
            "This is a known staff vehicle that is registered and authorised for property access."
        ),
    },
    # 17 ─ Person with backpack at Parking Lot, 03:00
    {
        "frame_id": 17,
        "timestamp": "2026-11-24T03:00:00",
        "location": "Parking Lot",
        "altitude": 25.0,
        "latitude": _LAT_BASE + 0.0005,
        "longitude": _LON_BASE - 0.0003,
        "description": (
            "A lone individual carrying a large backpack is walking through the parking lot at 03:00. "
            "The area should be empty at this hour. The person is moving slowly between parked vehicles "
            "and looking around frequently."
        ),
    },
    # 18 ─ Van blocking emergency exit
    {
        "frame_id": 18,
        "timestamp": "2026-11-24T11:45:00",
        "location": "Emergency Exit",
        "altitude": 15.0,
        "latitude": _LAT_BASE + 0.0007,
        "longitude": _LON_BASE + 0.0003,
        "description": (
            "A grey cargo van is parked directly across the emergency exit, blocking it completely. "
            "The vehicle has no company markings or signage. "
            "No driver is present. This creates both a safety hazard and a security concern."
        ),
    },
    # 19 ─ Routine drone patrol, nothing unusual
    {
        "frame_id": 19,
        "timestamp": "2026-11-24T15:30:00",
        "location": "Property Overview",
        "altitude": 50.0,
        "latitude": _LAT_BASE + 0.0002,
        "longitude": _LON_BASE + 0.0001,
        "description": (
            "Wide-angle overview of the property during afternoon patrol. "
            "No unusual activity detected. Buildings appear secure with all doors closed. "
            "Two employees are visible walking between buildings on authorised paths. "
            "Perimeter appears intact."
        ),
    },
    # 20 ─ Motion near storage shed at 00:30
    {
        "frame_id": 20,
        "timestamp": "2026-11-24T00:30:00",
        "location": "Storage Shed",
        "altitude": 18.0,
        "latitude": _LAT_BASE - 0.0008,
        "longitude": _LON_BASE + 0.0006,
        "description": (
            "Motion detected near the storage shed at the rear of the property. "
            "A shadowy figure is visible attempting to open the shed door. "
            "No authorised access is scheduled for this area after midnight. "
            "The individual's identity cannot be determined from current altitude."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_all_frames() -> List[DroneFrame]:
    """Return all 20 simulated frames as validated DroneFrame objects."""
    return [DroneFrame(**deepcopy(f)) for f in _SIMULATED_FRAMES]


def get_frame(frame_id: int) -> Optional[DroneFrame]:
    """
    Return a single DroneFrame by its frame_id, or None if not found.

    Args:
        frame_id: The integer frame_id to look up (1-based, 1–20).
    """
    for raw in _SIMULATED_FRAMES:
        if raw["frame_id"] == frame_id:
            return DroneFrame(**deepcopy(raw))
    return None


def get_frames_by_location(location: str) -> List[DroneFrame]:
    """Return all frames whose location matches the given string (case-insensitive)."""
    loc_lower = location.lower()
    return [
        DroneFrame(**deepcopy(f))
        for f in _SIMULATED_FRAMES
        if f["location"].lower() == loc_lower
    ]


def get_frames_in_time_range(start: datetime, end: datetime) -> List[DroneFrame]:
    """Return frames whose timestamps fall within [start, end] inclusive."""
    results = []
    for raw in _SIMULATED_FRAMES:
        ts = datetime.fromisoformat(raw["timestamp"])
        if start <= ts <= end:
            results.append(DroneFrame(**deepcopy(raw)))
    return results


def get_scenario_summary() -> List[dict]:
    """
    Return a lightweight list of {frame_id, timestamp, location, scenario_hint}
    suitable for display in the simulator UI without loading full descriptions.
    """
    return [
        {
            "frame_id": f["frame_id"],
            "timestamp": f["timestamp"],
            "location": f["location"],
            "altitude": f["altitude"],
        }
        for f in _SIMULATED_FRAMES
    ]

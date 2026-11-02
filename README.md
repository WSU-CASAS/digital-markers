# digital-markers

`digital-markers` is an open-source Python library from the CASAS Lab at
Washington State University. It computes four implemented marker families from
wearable sensor data: exertion, gait, location, and behavioral routine.

## Supported markers

Each function accepts a `ParticipantData` instance and returns a `MarkerResult`.
Participant inputs are kept in separate Pandas DataFrames; results include
participant-level markers and, where applicable, daily values and detail tables.

### Exertion

`digital_markers.exertion.compute_exertion_markers(participant)` computes daily
motion-only and heart-rate-plus-motion exertion markers, with participant-level
summaries. Motion data must include timestamps, acceleration, and rotation-rate
axes. Heart-rate data is optional; HR-derived values are unavailable when fewer
than 21 valid HR samples exist.

### Gait

`digital_markers.gait.compute_gait_markers(participant)` computes daily
wrist-derived walking-bout markers, including step counts, cadence, step
intervals, gait frequency, and signal measures. Timestamps and acceleration
axes are required; rotation-rate axes are optional.

### Location

`digital_markers.location.compute_location_markers(participant)` returns
participant-level and daily location measurements. `details["home_locations"]`
and `details["staypoints"]` contain home-location and staypoint tables. Keep
timestamps for all observed sensor rows in `ParticipantData.location`, including
rows without GPS coordinates; latitude and longitude are required on rows with
a GPS fix. A supplied motion stream's timestamps are also used to determine
recorded minutes.

### Behavioral routine

`digital_markers.routine.compute_routine_markers(participant)` computes
model-free routine markers from 10 Hz motion data using fixed acceleration
thresholds. Timestamps and acceleration axes are required; rotation-rate data
and GPS coordinates are optional. Coordinates are used from
`ParticipantData.location` when supplied, or from the motion table when
latitude and longitude are present there. The result includes participant-level
summaries and five-minute interval, state-balance, region, and diagnostic tables
in `details`.

## Installation

```bash
python -m pip install digital-markers
```

For development:

```bash
git clone https://github.com/WSU-CASAS/digital-markers.git
cd digital-markers
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

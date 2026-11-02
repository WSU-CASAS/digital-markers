# digital-markers

`digital-markers` is an open-source Python library for computing digital health markers from wearable sensor data and related participant data.

The project is developed by the CASAS Lab at Washington State University.

## Design principles

- Pandas DataFrames are the primary representation for longitudinal and event-based data.
- Static participant attributes use typed Python dataclasses.
- Different sensor modalities remain in separate DataFrames.
- Externally supplied participant data is kept separate from data derived by this library.
- Participant-provided activity labels are valid inputs and are distinct from model-generated activity labels.
- Marker families can declare required and optional inputs.
- Intermediate/daily results can be retained for reproducibility and inspection.

## Exertion outputs

`compute_exertion_markers(participant)` returns a `MarkerResult` with one row per
motion-valid day in `daily` and one participant-level row in `markers`. Daily
values include separate motion-only and HR-plus-motion exertion, peak HR,
exertion per wear second, HR/motion relationships, and wear time in seconds.
The per-hour-named metrics retain the reference method's convention of dividing
the sum of window exertion by wear time in seconds. The summary table reports
mean, population standard deviation, lag-one autocorrelation, and daily trend
for each daily marker. HR-derived values are unavailable when fewer than 21
valid HR samples exist; motion-only values remain available.

## Location outputs

`compute_location_markers(participant)` returns a `MarkerResult`. The participant
summary is in `markers`, daily measurements are in `daily`, and
`details["home_locations"]` and `details["staypoints"]` contain the other two
tables corresponding to the reference CSV outputs. Keep timestamps for all
observed sensor rows in `ParticipantData.location`, including rows without GPS
coordinates; a supplied `motion` stream's timestamps are also used to determine
recorded minutes. GPS coordinates are required only on rows containing a fix.

## Behavioral routine outputs

`compute_routine_markers(participant)` computes model-free routine markers from
10 Hz motion data using fixed acceleration thresholds. Acceleration and
timestamps are required; rotation-rate data and GPS coordinates are optional.
Coordinates are used from `ParticipantData.location` when supplied, or from the
motion table when latitude and longitude are present there. The result contains
participant-level routine summaries in `markers` and five-minute interval,
state-balance, region, and diagnostic tables in `details`. Routine computation
does not create profile files or use a pretrained model.

## Planned inputs

- Motion, heart rate, location, audio
- Apple-derived health measures and sleep
- EMA
- Behavioral and cognitive tasks
- Participant-provided activity labels
- Sociodemographics
- Baseline assessments
- Diagnosis history

## Planned marker families

- Watch adherence
- Activity level / exertion
- Location and mobility
- Gait
- Speech and language
- Air quality exposure
- Activity recognition

## Installation

Eventually:

```bash
pip install digital-markers
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

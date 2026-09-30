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
- Behavioral routine
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

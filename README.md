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

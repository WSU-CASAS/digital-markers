from dataclasses import dataclass, field
import pandas as pd
from digital_markers.data.assessments import BaselineAssessment
from digital_markers.data.demographics import Demographics

@dataclass
class ParticipantData:
    """Externally supplied participant data.

    Derived data produced by digital-markers belongs in MarkerContext, not here.
    """

    participant_id: str
    timezone: str | None = None

    demographics: Demographics | None = None
    baseline: BaselineAssessment | None = None
    diagnoses: pd.DataFrame | None = None

    motion: pd.DataFrame | None = None
    heart_rate: pd.DataFrame | None = None
    location: pd.DataFrame | None = None
    apple_health: pd.DataFrame | None = None
    sleep: pd.DataFrame | None = None
    ema: pd.DataFrame | None = None

    tasks: dict[str, pd.DataFrame] = field(default_factory=dict)

    # Externally supplied annotations such as participant-provided labels.
    # Suggested columns: start_time, end_time, activity, source
    activities: pd.DataFrame | None = None

    audio: pd.DataFrame | None = None

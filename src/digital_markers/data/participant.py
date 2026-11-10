from dataclasses import dataclass

import pandas as pd

from digital_markers.data.audio import AudioRecording
from digital_markers.data.validation import canonicalize_columns


@dataclass
class ParticipantData:
    """Externally supplied participant data.

    Derived data produced by digital-markers belongs in MarkerContext, not here.
    """

    participant_id: str
    timezone: str | None = None

    motion: pd.DataFrame | None = None
    heart_rate: pd.DataFrame | None = None
    location: pd.DataFrame | None = None
    activity: pd.DataFrame | None = None
    audio: list[AudioRecording] | None = None

    def __post_init__(self) -> None:
        if self.motion is not None:
            self.motion = canonicalize_columns(self.motion)
        if self.heart_rate is not None:
            self.heart_rate = canonicalize_columns(self.heart_rate)
        if self.location is not None:
            self.location = canonicalize_columns(self.location)
        if self.activity is not None:
            self.activity = canonicalize_columns(self.activity)

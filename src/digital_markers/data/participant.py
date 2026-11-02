from dataclasses import dataclass

import pandas as pd


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

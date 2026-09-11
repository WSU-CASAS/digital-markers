from dataclasses import dataclass

@dataclass
class BaselineAssessment:
    tics: float | None = None
    rbans: float | None = None
    moca: float | None = None

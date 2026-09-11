from dataclasses import dataclass, field

import pandas as pd

from digital_markers.context import MarkerContext


@dataclass
class MarkerResult:
    markers: pd.DataFrame
    daily: pd.DataFrame | None = None
    details: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    context: MarkerContext | None = None

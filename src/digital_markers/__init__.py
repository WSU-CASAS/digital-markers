"""Digital health markers derived from wearable and participant data."""

from importlib.metadata import PackageNotFoundError, version

from digital_markers.data.participant import ParticipantData
from digital_markers.results import MarkerResult

try:
    __version__ = version("digital-markers")
except PackageNotFoundError:
    __version__ = "0.1"

__all__ = ["MarkerResult", "ParticipantData", "__version__"]

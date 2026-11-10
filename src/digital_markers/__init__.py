"""Digital health markers derived from wearable and participant data."""

from importlib.metadata import PackageNotFoundError, version

from digital_markers.audio import compute_audio_markers, write_audio_transcripts
from digital_markers.data.audio import AudioRecording, read_audio_directory
from digital_markers.data.participant import ParticipantData
from digital_markers.results import MarkerResult

try:
    __version__ = version("digital-markers")
except PackageNotFoundError:
    __version__ = "0.1"

__all__ = [
    "AudioRecording",
    "MarkerResult",
    "ParticipantData",
    "__version__",
    "compute_audio_markers",
    "read_audio_directory",
    "write_audio_transcripts",
]

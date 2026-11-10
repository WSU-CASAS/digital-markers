from dataclasses import dataclass
from pathlib import Path

import numpy as np

AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4", ".wma"})


@dataclass
class AudioRecording:
    """An audio recording stored in a file or as in-memory samples.

    In-memory samples may be mono (one-dimensional) or have shape
    ``(samples, channels)``. Their sample rate is expressed in Hz. Integer PCM
    samples are normalized during processing; floating-point samples should be
    normalized to the range [-1, 1].
    """

    filename: str
    path: Path | None = None
    samples: np.ndarray | None = None
    sample_rate_hz: int | None = None

    def __post_init__(self) -> None:
        if not self.filename or Path(self.filename).name != self.filename:
            raise ValueError("filename must be a non-empty file name without directory components")
        if (self.path is None) == (self.samples is None):
            raise ValueError("provide exactly one of path or samples")
        if self.path is not None:
            self.path = Path(self.path).expanduser()
            if self.sample_rate_hz is not None:
                raise ValueError("sample_rate_hz is only valid for in-memory recordings")
            return
        if (
            self.sample_rate_hz is None
            or isinstance(self.sample_rate_hz, bool)
            or not isinstance(self.sample_rate_hz, (int, np.integer))
            or self.sample_rate_hz <= 0
        ):
            raise ValueError("in-memory recordings require a positive integer sample_rate_hz")
        samples = np.asarray(self.samples)
        if samples.ndim not in (1, 2) or samples.size == 0:
            raise ValueError("samples must be a non-empty mono or multi-channel array")
        if not np.issubdtype(samples.dtype, np.number) or np.iscomplexobj(samples):
            raise ValueError("samples must contain numeric values")
        if samples.ndim == 2 and samples.shape[1] == 0:
            raise ValueError("multi-channel samples must contain at least one channel")
        if not np.isfinite(samples).all():
            raise ValueError("samples must contain only finite values")
        self.samples = samples


def read_audio_directory(directory: str | Path) -> list[AudioRecording]:
    """Return path-backed recordings for supported files directly in a directory."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {root}")
    return [
        AudioRecording(filename=path.name, path=path)
        for path in sorted(root.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]

"""Band-pass filtering and candidate step-peak detection."""

from functools import lru_cache

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


@lru_cache(maxsize=16)
def cached_bandpass(fs: float) -> np.ndarray:
	nyquist = fs / 2
	return butter(
		4,
		[0.5 / nyquist, min(3.0, nyquist * 0.95) / nyquist],
		btype="bandpass",
		output="sos",
	)


def detect_step_peaks(
	acceleration_magnitude: np.ndarray,
	fs: float,
	*,
	max_cadence: float,
	prominence_scale: float,
) -> np.ndarray:
	try:
		filtered = sosfiltfilt(cached_bandpass(fs), acceleration_magnitude)
	except ValueError:
		return np.array([], dtype=int)

	sigma = np.std(filtered)
	if sigma < 1e-5:
		return np.array([], dtype=int)

	peaks, _ = find_peaks(
		filtered,
		distance=max(1, int(fs * 60 / max_cadence)),
		prominence=prominence_scale * sigma,
	)
	return peaks


__all__ = ["cached_bandpass", "detect_step_peaks"]

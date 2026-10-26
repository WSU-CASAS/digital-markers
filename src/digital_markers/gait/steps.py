"""Step detection from acceleration magnitude."""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def bandpass_filter(
	values: np.ndarray,
	fs: float,
	*,
	low: float = 0.5,
	high: float = 3.0,
	order: int = 4,
) -> np.ndarray:
	if len(values) <= 27:
		return values

	nyquist = 0.5 * fs
	numerator, denominator = butter(
		order, [low / nyquist, high / nyquist], btype="band"
	)
	try:
		return filtfilt(numerator, denominator, values)
	except Exception:
		return values


def detect_steps(acceleration_magnitude: np.ndarray, fs: float) -> np.ndarray:
	if len(acceleration_magnitude) < fs * 2:
		return np.array([])

	filtered = bandpass_filter(acceleration_magnitude, fs)
	standard_deviation = np.std(filtered)
	if standard_deviation == 0:
		return np.array([])

	peaks, _ = find_peaks(
		filtered,
		distance=int(0.4 * fs),
		prominence=standard_deviation * 0.5,
	)
	return peaks / fs


__all__ = ["bandpass_filter", "detect_steps"]

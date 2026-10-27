"""Timestamp segmentation and conservative walking-bout detection."""

import numpy as np

from digital_markers.gait.spectral import bout_spectral_features
from digital_markers.gait.steps import detect_step_peaks


def segment_indices(
	stamps_ns: np.ndarray,
	fs: float,
	max_gap_factor: float,
) -> list[tuple[int, int]]:
	deltas = np.diff(stamps_ns)
	boundaries = np.flatnonzero(
		(deltas <= 0) | (deltas > round(max_gap_factor * 1e9 / fs))
	) + 1
	return list(zip(np.r_[0, boundaries], np.r_[boundaries, len(stamps_ns)]))


def accepted_bouts(
	acceleration: np.ndarray,
	fs: float,
	*,
	min_steps: int,
	min_bout_seconds: float,
	max_step_gap_seconds: float,
	min_cadence: float,
	max_cadence: float,
	max_interval_cv: float,
	prominence_scale: float,
) -> list[dict[str, object]]:
	if len(acceleration) < max(15, int(min_bout_seconds * fs)):
		return []

	magnitude = np.linalg.norm(acceleration, axis=1)
	peaks = detect_step_peaks(
		magnitude,
		fs,
		max_cadence=max_cadence,
		prominence_scale=prominence_scale,
	)
	if len(peaks) < min_steps:
		return []

	gap_seconds = np.diff(peaks) / fs
	breaks = np.flatnonzero(gap_seconds > max_step_gap_seconds) + 1
	bouts = []
	for indices in np.split(peaks, breaks):
		if len(indices) < min_steps:
			continue

		intervals = np.diff(indices) / fs
		duration = (indices[-1] - indices[0]) / fs
		if duration < min_bout_seconds:
			continue

		mean_interval = np.mean(intervals)
		cadence = 60 / mean_interval
		interval_cv = float(np.std(intervals) / mean_interval)
		if not (
			min_cadence <= cadence <= max_cadence
			and interval_cv <= max_interval_cv
		):
			continue

		low = max(0, int(indices[0] - fs))
		high = min(len(magnitude), int(indices[-1] + fs + 1))
		window = magnitude[low:high]
		dominant_frequency, spectral_entropy = bout_spectral_features(window, fs)
		bouts.append(
			{
				"nsteps": len(indices),
				"intervals": intervals,
				"duration": duration,
				"cadence": cadence,
				"dominant_gait_freq": dominant_frequency,
				"spectral_entropy": spectral_entropy,
				"acc_sq_sum": float(np.square(magnitude[low:high]).sum()),
				"acc_n": high - low,
				"lo": low,
				"hi": high,
			}
		)
	return bouts


__all__ = ["accepted_bouts", "segment_indices"]

"""Frequency, symmetry, turning, and nonlinear gait features."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.signal import find_peaks, welch
from scipy.stats import entropy

from digital_markers.gait.steps import bandpass_filter


def safe_float(value: float) -> float:
	return float(value) if np.isfinite(value) else 0.0


def gait_frequency_features(acceleration_magnitude: np.ndarray, fs: float) -> tuple[float, float]:
	frequencies, power = welch(
		acceleration_magnitude,
		fs=fs,
		nperseg=min(len(acceleration_magnitude), int(fs * 4)),
	)
	band = (frequencies >= 0.5) & (frequencies <= 3.0)
	if not np.any(band):
		return 0.0, 0.0

	band_power = power[band]
	if np.sum(band_power) == 0:
		return 0.0, 0.0

	band_frequencies = frequencies[band]
	dominant_frequency = band_frequencies[np.argmax(band_power)]
	spectral_entropy = entropy(band_power / np.sum(band_power))
	return safe_float(dominant_frequency), safe_float(spectral_entropy)


def turning_markers(gyro_z: np.ndarray, fs: float) -> dict[str, float | int]:
	yaw = bandpass_filter(gyro_z, fs, low=0.1, high=1.5)
	turn_rate_mean = np.mean(np.abs(yaw))
	turns, _ = find_peaks(np.abs(yaw), height=np.std(yaw) * 2, distance=int(fs))
	return {"turn_rate": safe_float(turn_rate_mean), "num_turns": int(len(turns))}


def gait_symmetry(step_times: np.ndarray, fs: float) -> float:
	if len(step_times) < 6:
		return 0.0

	step_samples = (step_times * fs).astype(int)
	intervals = np.diff(step_samples)
	if len(intervals) < 4:
		return 0.0

	even = intervals[::2]
	odd = intervals[1::2]
	if len(even) < 2 or len(odd) < 2:
		return 0.0

	denominator = np.mean(intervals)
	if denominator == 0:
		return 0.0
	return safe_float(abs(np.mean(even) - np.mean(odd)) / denominator)


def short_term_lyapunov(
	values: np.ndarray,
	fs: float,
	*,
	max_t: float = 1.0,
) -> float:
	embedding_dimension = 5
	delay = int(0.1 * fs)
	if delay < 1:
		return 0.0

	embedding_count = len(values) - (embedding_dimension - 1) * delay
	if embedding_count < 50:
		return 0.0

	embedded = np.array(
		[values[index : index + embedding_dimension * delay : delay]
		 for index in range(embedding_count)]
	)
	try:
		_, neighbors = cKDTree(embedded).query(embedded, k=2)
		neighbor_indices = np.where(
			neighbors[:, 0] == np.arange(len(neighbors)), neighbors[:, 1], neighbors[:, 0]
		)
	except Exception:
		return 0.0

	divergence = []
	for step in range(1, int(max_t * fs)):
		distances = []
		for index, neighbor in enumerate(neighbor_indices):
			if index + step < len(values) and neighbor + step < len(values):
				difference = abs(values[index + step] - values[neighbor + step])
				if difference > 0:
					distances.append(difference)
		if distances:
			divergence.append(np.mean(np.log(distances)))

	if len(divergence) < 5:
		return 0.0

	slope = np.polyfit(np.arange(len(divergence)) / fs, divergence, 1)[0]
	return safe_float(slope)


__all__ = [
	"gait_frequency_features",
	"gait_symmetry",
	"safe_float",
	"short_term_lyapunov",
	"turning_markers",
]

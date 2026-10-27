"""Spectral features calculated within accepted walking bouts."""

import numpy as np
from scipy.signal import welch


def bout_spectral_features(values: np.ndarray, fs: float) -> tuple[float, float]:
	frequencies, power = welch(values, fs=fs, nperseg=min(len(values), int(fs * 8)))
	band = (frequencies >= 0.5) & (frequencies <= 3.0)
	if not np.any(band) or power[band].sum() <= 0:
		return np.nan, np.nan

	band_power = power[band]
	probabilities = band_power / band_power.sum()
	positive_probabilities = probabilities[probabilities > 0]
	dominant_frequency = frequencies[band][np.argmax(band_power)]
	spectral_entropy = -(positive_probabilities * np.log(positive_probabilities)).sum()
	return float(dominant_frequency), float(spectral_entropy)


__all__ = ["bout_spectral_features"]

"""Compute daily gait features and participant-level summaries."""

import numpy as np
import pandas as pd

from digital_markers.context import MarkerContext
from digital_markers.data.participant import ParticipantData
from digital_markers.data.validation import require_marker_columns
from digital_markers.gait.bouts import count_walking_bouts
from digital_markers.gait.spectral import (
	gait_frequency_features,
	gait_symmetry,
	safe_float,
	short_term_lyapunov,
	turning_markers,
)
from digital_markers.gait.steps import detect_steps
from digital_markers.preprocessing.wear import detect_nonwear
from digital_markers.requirements import MarkerRequirements
from digital_markers.results import MarkerResult


REQUIREMENTS = MarkerRequirements(
	required=("motion",),
	required_columns=(
		"motion.timestamp",
		"motion.acceleration_x",
		"motion.acceleration_y",
		"motion.acceleration_z",
		"motion.rotation_rate_x",
		"motion.rotation_rate_y",
		"motion.rotation_rate_z",
	),
)

MOTION_COLUMNS = (
	"acceleration_x",
	"acceleration_y",
	"acceleration_z",
	"rotation_rate_x",
	"rotation_rate_y",
	"rotation_rate_z",
)
DAILY_FIELDS = (
	"num_steps",
	"num_walking_bouts",
	"wear_time_seconds",
	"cadence_spm",
	"stride_time",
	"acc_rms",
	"gyro_rms",
	"dominant_gait_freq",
	"spectral_entropy",
	"gait_symmetry",
	"lyapunov_exp",
	"turn_rate",
	"num_turns",
)


def vector_magnitude(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
	return np.sqrt(np.sum(frame.loc[:, columns].to_numpy(dtype=float) ** 2, axis=1))


def extract_daily_gait_markers(frame: pd.DataFrame, fs: float) -> dict[str, float | int] | None:
	acceleration_magnitude = vector_magnitude(
		frame, ("acceleration_x", "acceleration_y", "acceleration_z")
	)
	if len(acceleration_magnitude) < fs * 3:
		return None

	gyro_magnitude = vector_magnitude(
		frame, ("rotation_rate_x", "rotation_rate_y", "rotation_rate_z")
	)
	step_times = detect_steps(acceleration_magnitude, fs)
	if len(step_times) < 5:
		return None

	step_intervals = np.diff(step_times)
	if not len(step_intervals):
		return None

	mean_interval = np.mean(step_intervals)
	cadence = 60 / mean_interval if mean_interval > 0 else 0.0
	dominant_frequency, spectral_entropy = gait_frequency_features(acceleration_magnitude, fs)
	turning = turning_markers(frame["rotation_rate_z"].to_numpy(dtype=float), fs)

	return {
		"num_steps": int(len(step_times)),
		"num_walking_bouts": count_walking_bouts(step_times),
		"wear_time_seconds": safe_float(len(frame) / fs),
		"cadence_spm": safe_float(cadence),
		"stride_time": safe_float(mean_interval),
		"acc_rms": safe_float(np.sqrt(np.mean(acceleration_magnitude**2))),
		"gyro_rms": safe_float(np.sqrt(np.mean(gyro_magnitude**2))),
		"dominant_gait_freq": dominant_frequency,
		"spectral_entropy": spectral_entropy,
		"gait_symmetry": gait_symmetry(step_times, fs),
		"lyapunov_exp": short_term_lyapunov(acceleration_magnitude, fs),
		**turning,
	}


def compute_lag1_autocorr(values: np.ndarray) -> float:
	values = np.asarray(values, dtype=float)
	values = values[np.isfinite(values)]
	if len(values) < 3:
		return np.nan

	centered = values - np.mean(values)
	denominator = np.sum(centered**2)
	if denominator == 0:
		return np.nan
	return float(np.sum(centered[:-1] * centered[1:]) / denominator)


def compute_slope(values: np.ndarray) -> float:
	values = np.asarray(values, dtype=float)
	values = values[np.isfinite(values)]
	if len(values) < 2:
		return 0.0
	return float(np.polyfit(np.arange(len(values)), values, 1)[0])


def aggregate_daily_markers(daily: pd.DataFrame) -> dict[str, float | int]:
	ordered = daily.sort_values("date")
	summary: dict[str, float | int] = {"num_valid_days": int(len(ordered))}
	for field in DAILY_FIELDS:
		values = ordered[field].to_numpy(dtype=float)
		values = values[np.isfinite(values)]
		if not len(values):
			summary[f"mean_{field}"] = np.nan
			summary[f"std_{field}"] = np.nan
			summary[f"ac1_{field}"] = np.nan
			summary[f"slope_{field}"] = np.nan
			continue

		summary[f"mean_{field}"] = float(np.mean(values))
		summary[f"std_{field}"] = float(np.std(values, ddof=0))
		summary[f"ac1_{field}"] = compute_lag1_autocorr(values)
		summary[f"slope_{field}"] = compute_slope(values)
	return summary


def compute_gait_markers(data: ParticipantData, *, fs: float = 50.0) -> MarkerResult:
	"""Compute gait markers from one participant's canonical motion dataframe."""
	if data.motion is None or data.motion.empty:
		raise ValueError("Missing required data for gait: motion")
	if fs <= 0:
		raise ValueError("fs must be positive")

	motion = data.motion.copy()
	require_marker_columns({"motion": motion}, REQUIREMENTS.required_columns)
	if not pd.api.types.is_datetime64_any_dtype(motion["timestamp"]):
		raise ValueError("motion.timestamp must have a datetime dtype before calling gait")
	for column in MOTION_COLUMNS:
		if not pd.api.types.is_numeric_dtype(motion[column]):
			raise ValueError(f"motion.{column} must have a numeric dtype before calling gait")

	motion = motion.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
	wear_mask = detect_nonwear(motion)
	worn_motion = motion.loc[wear_mask.to_numpy()].copy()
	worn_motion["date"] = worn_motion["timestamp"].dt.normalize()

	rows = []
	for date, day in worn_motion.groupby("date", sort=True):
		markers = extract_daily_gait_markers(day, fs)
		if markers is not None:
			rows.append({"date": date, **markers})

	daily = pd.DataFrame(rows, columns=("date", *DAILY_FIELDS))
	summary = aggregate_daily_markers(daily)
	markers = pd.DataFrame([summary])
	context = MarkerContext(data=data, derived={"gait": daily.set_index("date").to_dict("index")})
	return MarkerResult(
		markers=markers,
		daily=daily,
		metadata={"marker": "gait", "sampling_rate_hz": fs},
		context=context,
	)


__all__ = [
	"DAILY_FIELDS",
	"REQUIREMENTS",
	"aggregate_daily_markers",
	"compute_gait_markers",
	"extract_daily_gait_markers",
]

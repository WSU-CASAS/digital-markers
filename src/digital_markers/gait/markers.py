"""Compute conservative daily wrist-derived walking-bout markers."""

import numpy as np
import pandas as pd

from digital_markers.context import MarkerContext
from digital_markers.data.participant import ParticipantData
from digital_markers.data.validation import require_marker_columns
from digital_markers.gait.bouts import accepted_bouts, segment_indices
from digital_markers.preprocessing.wear import detect_nonwear
from digital_markers.requirements import MarkerRequirements
from digital_markers.results import MarkerResult

ACC_COLUMNS = ("acceleration_x", "acceleration_y", "acceleration_z")
GYRO_COLUMNS = ("rotation_rate_x", "rotation_rate_y", "rotation_rate_z")
BOUNDS = (
	"num_steps",
	"num_walking_bouts",
	"walking_duration_seconds",
	"wear_time_seconds",
)
MEASURES = (
	"cadence_spm",
	"step_interval_mean",
	"step_interval_sd",
	"step_interval_cv",
	"dominant_gait_freq",
	"spectral_entropy",
	"acc_rms",
	"gyro_rms",
)
DAILY_FIELDS = (*BOUNDS, *MEASURES)
DAILY_COLUMNS = ("date", *DAILY_FIELDS)

REQUIREMENTS = MarkerRequirements(
	required=("motion",),
	required_columns=(
		"motion.timestamp",
		"motion.acceleration_x",
		"motion.acceleration_y",
		"motion.acceleration_z",
	),
	optional_columns=(
		"motion.rotation_rate_x",
		"motion.rotation_rate_y",
		"motion.rotation_rate_z",
	),
)


def build_daily_markers(segments: list[dict[str, object]]) -> pd.DataFrame:
	rows = []
	for date in sorted({str(segment["date"]) for segment in segments}):
		parts = [segment for segment in segments if str(segment["date"]) == date]
		bouts = [bout for part in parts for bout in part["bouts"]]
		wear_seconds = sum(float(part["wear_seconds"]) for part in parts)
		row: dict[str, object] = dict.fromkeys(DAILY_COLUMNS, np.nan)
		row.update(
			date=pd.Timestamp(date),
			wear_time_seconds=wear_seconds,
			num_steps=sum(int(bout["nsteps"]) for bout in bouts),
			num_walking_bouts=len(bouts),
			walking_duration_seconds=sum(float(bout["duration"]) for bout in bouts),
		)
		if bouts:
			intervals = np.concatenate([bout["intervals"] for bout in bouts])
			row["step_interval_mean"] = float(np.mean(intervals))
			row["step_interval_sd"] = float(np.std(intervals))
			row["step_interval_cv"] = float(np.std(intervals) / np.mean(intervals))
			row["cadence_spm"] = float(np.median([bout["cadence"] for bout in bouts]))
			for key in ("dominant_gait_freq", "spectral_entropy"):
				weighted = [
					(bout[key], len(bout["intervals"]))
					for bout in bouts
					if np.isfinite(bout[key])
				]
				if weighted:
					row[key] = float(
						np.average(
							[value for value, _ in weighted],
							weights=[weight for _, weight in weighted],
						)
					)
			for name, prefix in (("acc_rms", "acc"), ("gyro_rms", "gyro")):
				weighted = [
					(bout[f"{prefix}_sq_sum"], bout[f"{prefix}_n"])
					for bout in bouts
					if bout[f"{prefix}_n"] > 0 and np.isfinite(bout[f"{prefix}_sq_sum"])
				]
				if weighted:
					row[name] = float(
						np.sqrt(
							sum(total for total, _ in weighted)
							/ sum(count for _, count in weighted)
						)
					)
		rows.append(row)
	return pd.DataFrame(rows, columns=DAILY_COLUMNS)


def aggregate_participant_markers(daily: pd.DataFrame) -> dict[str, float | int]:
	ordered = daily.sort_values("date")
	row: dict[str, float | int] = {
		"num_recorded_days": len(ordered),
		"num_valid_days": int(ordered["num_walking_bouts"].gt(0).sum()),
	}
	date_values = pd.to_datetime(ordered["date"])
	date_offsets = (date_values - date_values.min()).dt.days.to_numpy()
	consecutive = np.flatnonzero(np.diff(date_offsets) == 1)

	for column in DAILY_FIELDS:
		values = pd.to_numeric(ordered[column], errors="coerce").to_numpy(dtype=float)
		finite = np.isfinite(values)
		row[f"mean_{column}"] = float(np.mean(values[finite])) if finite.any() else np.nan
		row[f"std_{column}"] = float(np.std(values[finite])) if finite.any() else np.nan
		if finite.sum() >= 2 and np.unique(date_offsets[finite]).size >= 2:
			row[f"slope_{column}"] = float(
				np.polyfit(date_offsets[finite], values[finite], 1)[0]
			)
		else:
			row[f"slope_{column}"] = np.nan

		pairs = [
			index
			for index in consecutive
			if finite[index] and finite[index + 1]
		]
		if (
			len(pairs) >= 3
			and np.std(values[pairs]) > 0
			and np.std(values[np.asarray(pairs) + 1]) > 0
		):
			row[f"ac1_{column}"] = float(
				np.corrcoef(values[pairs], values[np.asarray(pairs) + 1])[0, 1]
			)
		else:
			row[f"ac1_{column}"] = np.nan
	return row


def compute_gait_markers(
	data: ParticipantData,
	*,
	fs: float = 10.0,
	wear_threshold: float = 0.5,
	max_gap_factor: float = 2.5,
	min_steps: int = 8,
	min_bout_seconds: float = 4.0,
	max_step_gap_seconds: float = 1.5,
	min_cadence: float = 45.0,
	max_cadence: float = 180.0,
	max_interval_cv: float = 0.4,
	prominence_scale: float = 0.5,
) -> MarkerResult:
	"""Compute conservative bout-based gait markers for one participant."""
	if data.motion is None or data.motion.empty:
		raise ValueError("Missing required data for gait: motion")
	if fs <= 6 or max_gap_factor <= 1 or min_steps < 3:
		raise ValueError("Invalid sampling frequency, gap factor, or minimum steps")

	motion = data.motion.copy()
	require_marker_columns({"motion": motion}, REQUIREMENTS.required_columns)
	if not pd.api.types.is_datetime64_any_dtype(motion["timestamp"]):
		raise ValueError("motion.timestamp must have a datetime dtype before calling gait")
	for column in ACC_COLUMNS:
		if not pd.api.types.is_numeric_dtype(motion[column]):
			raise ValueError(f"motion.{column} must have a numeric dtype before calling gait")

	gyro_columns_available = all(column in motion.columns for column in GYRO_COLUMNS)
	for column in GYRO_COLUMNS:
		if column in motion and not pd.api.types.is_numeric_dtype(motion[column]):
			raise ValueError(f"motion.{column} must have a numeric dtype before calling gait")

	motion = (
		motion.dropna(subset=["timestamp", *ACC_COLUMNS])
		.sort_values("timestamp", kind="stable")
		.reset_index(drop=True)
	)
	wear_motion = motion.copy()
	for column in GYRO_COLUMNS:
		if column not in wear_motion:
			wear_motion[column] = np.nan
	wear_mask = detect_nonwear(wear_motion).to_numpy() > wear_threshold
	motion = motion.loc[wear_mask].reset_index(drop=True)
	if motion.empty:
		daily = pd.DataFrame(columns=DAILY_COLUMNS)
	else:
		timestamps = motion["timestamp"].to_numpy(dtype="datetime64[ns]")
		stamp_ns = timestamps.astype(np.int64)
		dates = timestamps.astype("datetime64[D]")
		acceleration = motion.loc[:, ACC_COLUMNS].to_numpy(dtype=np.float64)
		gyro = (
			motion.loc[:, GYRO_COLUMNS].to_numpy(dtype=np.float64)
			if gyro_columns_available
			else None
		)
		day_boundaries = np.flatnonzero(dates[1:] != dates[:-1]) + 1
		segments = []
		for day_start, day_stop in zip(
			np.r_[0, day_boundaries], np.r_[day_boundaries, len(timestamps)]
		):
			day_stamps = stamp_ns[day_start:day_stop]
			for segment_start, segment_stop in segment_indices(
				day_stamps, fs, max_gap_factor
			):
				if segment_stop - segment_start < max(15, int(min_bout_seconds * fs)):
					continue
				start = day_start + segment_start
				stop = day_start + segment_stop
				segment_acceleration = acceleration[start:stop]
				bouts = accepted_bouts(
					segment_acceleration,
					fs,
					min_steps=min_steps,
					min_bout_seconds=min_bout_seconds,
					max_step_gap_seconds=max_step_gap_seconds,
					min_cadence=min_cadence,
					max_cadence=max_cadence,
					max_interval_cv=max_interval_cv,
					prominence_scale=prominence_scale,
				)
				if gyro is not None and bouts:
					segment_gyro = gyro[start:stop]
					for bout in bouts:
						bout_gyro = segment_gyro[bout["lo"] : bout["hi"]]
						finite = np.isfinite(bout_gyro).all()
						bout["gyro_sq_sum"] = (
							float(np.square(bout_gyro).sum()) if finite else np.nan
						)
						bout["gyro_n"] = len(bout_gyro) if finite else 0
				else:
					for bout in bouts:
						bout["gyro_sq_sum"] = np.nan
						bout["gyro_n"] = 0
				segments.append(
					{
						"date": str(dates[day_start + segment_start]),
						"wear_seconds": (segment_stop - segment_start) / fs,
						"bouts": bouts,
					}
				)
		daily = build_daily_markers(segments)

	markers = pd.DataFrame([aggregate_participant_markers(daily)])
	context = MarkerContext(
		data=data,
		derived={"gait": daily.set_index("date").to_dict("index")},
	)
	return MarkerResult(
		markers=markers,
		daily=daily,
		metadata={"marker": "gait", "participant_id": data.participant_id, "sampling_rate_hz": fs},
		context=context,
	)


__all__ = [
	"BOUNDS",
	"DAILY_COLUMNS",
	"DAILY_FIELDS",
	"MEASURES",
	"REQUIREMENTS",
	"aggregate_participant_markers",
	"build_daily_markers",
	"compute_gait_markers",
]

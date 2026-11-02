"""Compute participant-level and daily location markers."""

import numpy as np
import pandas as pd

from digital_markers.context import MarkerContext
from digital_markers.data.participant import ParticipantData
from digital_markers.data.validation import require_marker_columns
from digital_markers.location.distance import haversine_distance
from digital_markers.location.home import home_report, prepare_fixes
from digital_markers.location.staypoints import (
    MAX_GPS_AGE_HOURS,
    MAX_HOME_GAP_HOURS,
    MAX_SPEED_MPS,
    MIN_STAY_MINUTES,
    cluster_staypoints,
    gps_distance,
    infer_bracketed_home_minutes,
    make_minute_locations,
    make_staypoints,
)
from digital_markers.requirements import MarkerRequirements
from digital_markers.results import MarkerResult

REQUIREMENTS = MarkerRequirements(
    required=("location",),
    optional=("motion",),
    required_columns=("location.timestamp", "location.latitude", "location.longitude"),
    optional_columns=("motion.timestamp",),
)

MIN_POINTS_PER_DAY = 1
MAX_GAP_SECONDS = 86_400

MARKER_STATS = {
    "total_distance": ("mean", "std", "ac1"),
    "max_distance_from_home": ("mean",),
    "mean_distance_from_home": ("mean",),
    "observed_time": ("mean",),
    "unknown_time": ("mean",),
    "away_time": ("mean", "std", "ac1"),
    "time_at_home_ratio": ("mean", "std", "ac1", "slope"),
    "num_distinct_locations": ("mean", "std"),
    "observed_home_ratio": ("mean",),
    "assumed_home_time": ("mean",),
}


def shrinkage_std(values: np.ndarray, reference_variance: float, k: float = 10) -> float:
    """Estimate standard deviation with variance shrinkage toward a reference."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(reference_variance):
        reference_variance = 0.0
    sample_variance = float(np.var(values, ddof=1)) if len(values) >= 2 else 0.0
    weight = (len(values) - 1) / (len(values) - 1 + k) if len(values) > 1 else 0.0
    variance = weight * sample_variance + (1.0 - weight) * reference_variance
    return float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else 0.0


def compute_lag1_autocorr(values: np.ndarray, dates: np.ndarray) -> float:
    """Compute lag-1 autocorrelation from pairs on consecutive calendar days."""
    values = np.asarray(values, dtype=float)
    dates = np.asarray(pd.to_datetime(dates), dtype="datetime64[D]")
    valid = np.isfinite(values) & ~np.isnat(dates)
    values, dates = values[valid], dates[valid]
    if len(values) < 3:
        return np.nan
    order = np.argsort(dates)
    values, dates = values[order], dates[order]
    centered = values - np.mean(values)
    denominator = np.sum(centered**2)
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    consecutive = np.diff(dates) == np.timedelta64(1, "D")
    pair_count = int(np.sum(consecutive))
    if pair_count < 2:
        return np.nan
    numerator = np.sum((centered[:-1] * centered[1:])[consecutive])
    return float((numerator / pair_count) / (denominator / len(values)))


def compute_slope(values: np.ndarray, dates: np.ndarray) -> float:
    """Compute the least-squares slope per elapsed calendar day."""
    values = np.asarray(values, dtype=float)
    dates = np.asarray(pd.to_datetime(dates), dtype="datetime64[D]")
    valid = np.isfinite(values) & ~np.isnat(dates)
    values, dates = values[valid], dates[valid]
    if len(values) < 2:
        return np.nan
    elapsed = (dates - dates.min()).astype("timedelta64[D]").astype(float)
    elapsed -= elapsed.mean()
    denominator = np.dot(elapsed, elapsed)
    if denominator <= 0:
        return np.nan
    return float(np.dot(elapsed, values - values.mean()) / denominator)


def aggregate_daily(daily: pd.DataFrame, participant_id: str) -> dict[str, object]:
    """Aggregate daily location markers into a compact participant summary."""
    row: dict[str, object] = {"id": participant_id, "num_valid_days": len(daily)}
    dates = daily["date"].to_numpy()
    for column, statistics in MARKER_STATS.items():
        if column not in daily:
            raise ValueError(f"Missing daily marker: {column}")
        values = daily[column].to_numpy(dtype=float, copy=False)
        finite = np.isfinite(values)
        valid_values, valid_dates = values[finite], dates[finite]
        if len(valid_values) > 10:
            lower, upper = np.percentile(valid_values, [5, 95])
            keep = (valid_values >= lower) & (valid_values <= upper)
            trimmed, trimmed_dates = valid_values[keep], valid_dates[keep]
        else:
            trimmed, trimmed_dates = valid_values, valid_dates

        if "mean" in statistics:
            row[f"mean_{column}"] = float(trimmed.mean()) if len(trimmed) else np.nan
        if "std" in statistics:
            reference_variance = (
                float(np.var(valid_values, ddof=1)) if len(valid_values) > 1 else 0.0
            )
            row[f"std_{column}"] = (
                shrinkage_std(trimmed, reference_variance) if len(trimmed) else np.nan
            )
        if "ac1" in statistics:
            row[f"ac1_{column}"] = compute_lag1_autocorr(values, dates)
        if "slope" in statistics:
            row[f"slope_{column}"] = compute_slope(trimmed, trimmed_dates)

    max_distance = daily["max_distance_from_home"].to_numpy(dtype=float, copy=False)
    max_distance = max_distance[np.isfinite(max_distance)]
    row["p95_max_distance_from_home"] = (
        float(np.percentile(max_distance, 95)) if len(max_distance) else np.nan
    )
    return row


def daily_table(
    states: pd.DataFrame,
    fixes: pd.DataFrame,
    staypoints: pd.DataFrame,
    inferred_home_minutes: dict[object, int],
) -> pd.DataFrame:
    """Summarize observed and inferred home/away time for each recorded date."""
    if states.empty:
        return pd.DataFrame()
    temporary = states[["date", "known", "away", "dist_from_home", "minute"]].copy()
    temporary["observed_home"] = temporary["known"] & ~temporary["away"]
    grouped = temporary.groupby("date", sort=True)
    daily = (
        grouped.agg(
            n_recorded=("minute", "size"),
            num_points=("known", "sum"),
            n_home=("observed_home", "sum"),
            n_away=("away", "sum"),
            max_distance_from_home=("dist_from_home", "max"),
            mean_distance_from_home=("dist_from_home", "mean"),
            first_time=("minute", "min"),
            last_time=("minute", "max"),
        )
        .reset_index()
    )
    daily["observed_time"] = daily["n_recorded"] * 60.0
    daily["observed_home_time"] = daily["n_home"] * 60.0
    daily["away_time"] = daily["n_away"] * 60.0
    daily["unknown_location_time"] = (
        daily["observed_time"] - daily["observed_home_time"] - daily["away_time"]
    )
    daily["assumed_home_time"] = (
        daily["date"].map(inferred_home_minutes).fillna(0.0).astype(float) * 60.0
    )
    daily["estimated_home_time"] = daily["observed_home_time"] + daily["assumed_home_time"]
    daily["home_time"] = daily["estimated_home_time"]
    daily["unknown_time"] = (
        86_400.0 - daily["observed_home_time"] - daily["away_time"] - daily["assumed_home_time"]
    )
    classified_time = daily["estimated_home_time"] + daily["away_time"]
    daily["time_at_home_ratio"] = daily["estimated_home_time"].div(
        classified_time.where(classified_time > 0)
    )
    daily["observed_home_ratio"] = daily["observed_home_time"] / daily["observed_time"]
    daily["span"] = (daily["last_time"] - daily["first_time"]).dt.total_seconds()

    distances = (
        fixes.groupby("date")["step_dist"].sum(min_count=1)
        if not fixes.empty
        else pd.Series(dtype=float)
    )
    daily["total_distance"] = daily["date"].map(distances)
    if staypoints.empty:
        location_counts = pd.Series(dtype=int)
    else:
        stays = staypoints.assign(date=pd.to_datetime(staypoints["start_time"]).dt.date)
        location_counts = stays.groupby("date")["location_id"].nunique()
    daily["num_distinct_locations"] = daily["date"].map(location_counts).fillna(0).astype(int)
    daily = daily.loc[
        (daily["num_points"] >= MIN_POINTS_PER_DAY) & (daily["span"] < MAX_GAP_SECONDS)
    ].copy()
    first_columns = [
        "date",
        "total_distance",
        "max_distance_from_home",
        "mean_distance_from_home",
        "observed_time",
        "away_time",
        "first_time",
        "last_time",
        "num_points",
        "span",
        "home_time",
        "time_at_home_ratio",
        "num_distinct_locations",
    ]
    return daily[first_columns + [name for name in daily if name not in first_columns]].reset_index(
        drop=True
    )


def radius_gyration(states: pd.DataFrame) -> float:
    """Compute time-weighted radius of gyration across known recording minutes."""
    known = states.loc[states["known"], ["latitude", "longitude"]]
    if len(known) < 2:
        return np.nan
    center_latitude = float(known["latitude"].mean())
    center_longitude = float(known["longitude"].mean())
    distances = haversine_distance(
        known["latitude"].to_numpy(dtype=float),
        known["longitude"].to_numpy(dtype=float),
        center_latitude,
        center_longitude,
    )
    return float(np.sqrt(np.mean(distances**2)))


def location_entropy(
    staypoint_summary: pd.DataFrame,
    daily: pd.DataFrame,
    home_latitude: float,
    home_longitude: float,
) -> float:
    """Compute staypoint-duration entropy, including inferred time at home."""
    assumed_home = float(daily["assumed_home_time"].sum())
    if staypoint_summary.empty:
        return 0.0 if assumed_home > 0 else np.nan
    durations = staypoint_summary["total_duration"].to_numpy(dtype=float).copy()
    durations[~np.isfinite(durations) | (durations < 0)] = 0.0
    if assumed_home > 0:
        distances = haversine_distance(
            staypoint_summary["latitude"].to_numpy(dtype=float),
            staypoint_summary["longitude"].to_numpy(dtype=float),
            home_latitude,
            home_longitude,
        )
        near_home = np.flatnonzero(distances <= 100.0)
        if len(near_home):
            durations[int(near_home[np.argmin(distances[near_home])])] += assumed_home
        else:
            durations = np.append(durations, assumed_home)
    durations = durations[np.isfinite(durations) & (durations > 0)]
    if not len(durations):
        return np.nan
    probabilities = durations / durations.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def _recorded_minutes(data: ParticipantData) -> pd.DatetimeIndex:
    timestamp_frames = [data.location]
    if data.motion is not None:
        timestamp_frames.append(data.motion)
    timestamps = []
    for frame in timestamp_frames:
        if frame is not None and "timestamp" in frame:
            parsed = pd.to_datetime(frame["timestamp"], errors="coerce").dropna()
            if not parsed.empty:
                timestamps.append(parsed.dt.floor("min").to_numpy())
    if not timestamps:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(np.unique(np.concatenate(timestamps))).sort_values()


def compute_location_markers(data: ParticipantData) -> MarkerResult:
    """Compute location markers and return the reference CSVs as DataFrames.

    ``ParticipantData.location`` must retain timestamps for all observed sensor
    rows (including rows without GPS coordinates) to reproduce recording-time
    summaries. A supplied motion stream's timestamps are included as well.
    """
    if data.location is None or data.location.empty:
        raise ValueError("Missing required data for location: location")
    require_marker_columns(
        {"location": data.location},
        REQUIREMENTS.required_columns,
    )

    raw_location = data.location[["timestamp", "latitude", "longitude"]].copy()
    raw_location["timestamp"] = pd.to_datetime(raw_location["timestamp"], errors="coerce")
    raw_location["latitude"] = pd.to_numeric(raw_location["latitude"], errors="coerce")
    raw_location["longitude"] = pd.to_numeric(raw_location["longitude"], errors="coerce")
    fixes = prepare_fixes(raw_location)
    home = pd.DataFrame([home_report(fixes, data.participant_id)])
    minutes = _recorded_minutes(data)

    if len(fixes) < 2 or minutes.empty:
        return MarkerResult(
            markers=pd.DataFrame(),
            daily=pd.DataFrame(),
            details={
                "home_locations": home,
                "staypoints": pd.DataFrame(
                    columns=[
                        "location_id",
                        "latitude",
                        "longitude",
                        "num_visits",
                        "total_duration",
                        "mean_visit_duration",
                        "std_visit_duration",
                    ]
                ),
            },
            metadata={"participant_id": data.participant_id},
            context=MarkerContext(data),
        )

    fixes = gps_distance(fixes)
    states = make_minute_locations(
        minutes,
        fixes,
        float(home.iloc[0]["Home_Latitude"]),
        float(home.iloc[0]["Home_Longitude"]),
    )
    stays = make_staypoints(states)
    staypoint_summary, stays = cluster_staypoints(stays)
    inferred_home = infer_bracketed_home_minutes(states)
    daily = daily_table(states, fixes, stays, inferred_home)
    if daily.empty:
        markers = pd.DataFrame()
    else:
        summary = aggregate_daily(daily, data.participant_id)
        summary["radius_gyration"] = radius_gyration(states)
        summary["location_entropy"] = location_entropy(
            staypoint_summary,
            daily,
            float(home.iloc[0]["Home_Latitude"]),
            float(home.iloc[0]["Home_Longitude"]),
        )
        markers = pd.DataFrame([summary])

    return MarkerResult(
        markers=markers,
        daily=daily,
        details={"home_locations": home, "staypoints": staypoint_summary},
        metadata={
            "participant_id": data.participant_id,
            "max_gps_age_hours": MAX_GPS_AGE_HOURS,
            "max_home_gap_hours": MAX_HOME_GAP_HOURS,
            "max_speed_mps": MAX_SPEED_MPS,
            "minimum_stay_minutes": MIN_STAY_MINUTES,
        },
        context=MarkerContext(data),
    )

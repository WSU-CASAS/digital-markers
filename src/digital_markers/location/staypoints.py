"""Staypoint detection and location-state reconstruction."""

import numpy as np
import pandas as pd

from digital_markers.location.distance import haversine_distance

HOME_THRESHOLD_M = 100.0
MAX_SPEED_MPS = 50.0
LOCATION_RADIUS_M = 100.0
MIN_STAY_MINUTES = 10
MAX_GPS_AGE_HOURS = 16
MAX_HOME_GAP_HOURS = 16


def gps_distance(fixes: pd.DataFrame) -> pd.DataFrame:
    """Compute distances between successive valid fixes within the same day."""
    ordered = (
        fixes.sort_values("timestamp", kind="stable")
        .drop_duplicates("timestamp")
        .copy()
    )
    timestamps = ordered["timestamp"]
    elapsed = timestamps.diff().dt.total_seconds()
    distance = haversine_distance(
        ordered["latitude"].shift().to_numpy(dtype=float),
        ordered["longitude"].shift().to_numpy(dtype=float),
        ordered["latitude"].to_numpy(dtype=float),
        ordered["longitude"].to_numpy(dtype=float),
    )
    valid = (
        (timestamps.dt.normalize() == timestamps.shift().dt.normalize())
        & elapsed.gt(0)
        & (distance / elapsed <= MAX_SPEED_MPS)
    )
    ordered["step_dist"] = np.where(valid, distance, np.nan)
    ordered["date"] = timestamps.dt.date
    return ordered


def make_minute_locations(
    minutes: pd.DatetimeIndex,
    fixes: pd.DataFrame,
    home_latitude: float,
    home_longitude: float,
) -> pd.DataFrame:
    """Forward-fill GPS state within a day for at most 16 hours."""
    columns = ["minute", "date", "latitude", "longitude", "known", "away", "dist_from_home"]
    if minutes.empty:
        return pd.DataFrame(columns=columns)

    minute_rows = pd.DataFrame({"minute": minutes.sort_values()})
    minute_rows["date"] = minute_rows["minute"].dt.date
    fix_rows = fixes[["timestamp", "latitude", "longitude"]].copy()
    fix_rows["minute"] = fix_rows["timestamp"].dt.floor("min")
    fix_rows = (
        fix_rows.sort_values("timestamp", kind="stable")
        .drop_duplicates("minute", keep="last")
    )
    fix_rows["date"] = fix_rows["minute"].dt.date
    states = pd.merge_asof(
        minute_rows.sort_values("minute"),
        fix_rows[["minute", "date", "latitude", "longitude"]]
        .rename(columns={"minute": "fix_minute"})
        .sort_values("fix_minute"),
        left_on="minute",
        right_on="fix_minute",
        by="date",
        direction="backward",
        tolerance=pd.Timedelta(hours=MAX_GPS_AGE_HOURS),
    )
    states["known"] = states["latitude"].notna() & states["longitude"].notna()
    states["dist_from_home"] = haversine_distance(
        states["latitude"].to_numpy(dtype=float),
        states["longitude"].to_numpy(dtype=float),
        home_latitude,
        home_longitude,
    )
    states["away"] = states["known"] & states["dist_from_home"].gt(HOME_THRESHOLD_M)
    return states.drop(columns=["fix_minute"])


def infer_bracketed_home_minutes(states: pd.DataFrame) -> dict[object, int]:
    """Count gaps of up to 16 hours bracketed by known home states."""
    if len(states) < 2:
        return {}
    timestamps = states["minute"].to_numpy(dtype="datetime64[m]")
    gap_minutes = np.diff(timestamps).astype("timedelta64[m]").astype(np.int64) - 1
    gap_indices = np.flatnonzero(
        (gap_minutes > 0) & (gap_minutes <= MAX_HOME_GAP_HOURS * 60)
    )
    if not len(gap_indices):
        return {}
    known_home = states["known"].to_numpy() & ~states["away"].to_numpy()
    eligible = gap_indices[known_home[gap_indices] & known_home[gap_indices + 1]]
    daily: dict[object, int] = {}
    for index in eligible:
        start = timestamps[index] + np.timedelta64(1, "m")
        stop = timestamps[index + 1]
        while start < stop:
            day = start.astype("datetime64[D]")
            end = min(stop, (day + np.timedelta64(1, "D")).astype("datetime64[m]"))
            date = pd.Timestamp(day).date()
            daily[date] = daily.get(date, 0) + int((end - start) / np.timedelta64(1, "m"))
            start = end
    return daily


def _staypoint_segments(
    latitude: np.ndarray,
    longitude: np.ndarray,
    timestamps_ns: np.ndarray,
    dates: np.ndarray,
    radius_m: float,
    minimum_minutes: int,
) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    begin = 0
    for index in range(1, len(latitude) + 1):
        split = index == len(latitude)
        if not split:
            split = (
                dates[index] != dates[index - 1]
                or timestamps_ns[index] - timestamps_ns[index - 1] != 60_000_000_000
            )
        if not split:
            distance = haversine_distance(
                latitude[begin], longitude[begin], latitude[index], longitude[index]
            )
            split = distance > radius_m
        if split:
            if index - begin >= minimum_minutes:
                segments.append((begin, index))
            begin = index
    return segments


def make_staypoints(states: pd.DataFrame) -> pd.DataFrame:
    """Extract contiguous stays lasting at least ten minutes."""
    columns = ["latitude", "longitude", "duration", "start_time", "end_time"]
    known = states.loc[states["known"], ["minute", "latitude", "longitude", "date"]]
    if known.empty:
        return pd.DataFrame(columns=columns)
    known = known.reset_index(drop=True)
    latitude = known["latitude"].to_numpy(dtype=float)
    longitude = known["longitude"].to_numpy(dtype=float)
    minutes = known["minute"].to_numpy()
    segments = _staypoint_segments(
        latitude,
        longitude,
        minutes.astype("datetime64[ns]").astype(np.int64),
        minutes.astype("datetime64[D]").astype(np.int64),
        LOCATION_RADIUS_M,
        MIN_STAY_MINUTES,
    )
    rows = [
        (
            float(np.median(latitude[start:end])),
            float(np.median(longitude[start:end])),
            float((end - start) * 60),
            minutes[start],
            minutes[end - 1] + np.timedelta64(1, "m"),
        )
        for start, end in segments
    ]
    return pd.DataFrame(rows, columns=columns)


def cluster_staypoints(
    staypoints: pd.DataFrame, merge_radius_m: float = LOCATION_RADIUS_M
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster stays with the original first-matching-center update rule."""
    summary_columns = [
        "location_id",
        "latitude",
        "longitude",
        "num_visits",
        "total_duration",
        "mean_visit_duration",
        "std_visit_duration",
    ]
    if staypoints.empty:
        return (
            pd.DataFrame(columns=summary_columns),
            staypoints.assign(location_id=pd.Series(dtype="int64")),
        )

    center_latitudes: list[float] = []
    center_longitudes: list[float] = []
    member_latitudes: list[list[float]] = []
    member_longitudes: list[list[float]] = []
    member_durations: list[list[float]] = []
    location_ids: list[int] = []
    for latitude, longitude, duration in zip(
        staypoints["latitude"].to_numpy(dtype=float),
        staypoints["longitude"].to_numpy(dtype=float),
        staypoints["duration"].to_numpy(dtype=float),
    ):
        if center_latitudes:
            distances = haversine_distance(
                np.asarray(center_latitudes),
                np.asarray(center_longitudes),
                latitude,
                longitude,
            )
            matches = np.flatnonzero(distances <= merge_radius_m)
        else:
            matches = np.empty(0, dtype=int)
        if len(matches):
            location_id = int(matches[0])
            member_latitudes[location_id].append(latitude)
            member_longitudes[location_id].append(longitude)
            member_durations[location_id].append(duration)
            center_latitudes[location_id] = float(np.median(member_latitudes[location_id]))
            center_longitudes[location_id] = float(np.median(member_longitudes[location_id]))
        else:
            location_id = len(center_latitudes)
            center_latitudes.append(float(latitude))
            center_longitudes.append(float(longitude))
            member_latitudes.append([latitude])
            member_longitudes.append([longitude])
            member_durations.append([duration])
        location_ids.append(location_id)

    labeled = staypoints.copy()
    labeled["location_id"] = location_ids
    rows = []
    for location_id, durations in enumerate(member_durations):
        values = np.asarray(durations, dtype=float)
        rows.append(
            {
                "location_id": location_id,
                "latitude": center_latitudes[location_id],
                "longitude": center_longitudes[location_id],
                "num_visits": len(values),
                "total_duration": float(values.sum()),
                "mean_visit_duration": float(values.mean()),
                "std_visit_duration": float(values.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows, columns=summary_columns), labeled

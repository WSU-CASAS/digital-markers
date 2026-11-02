"""Compute model-free behavioral routine markers."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from digital_markers.context import MarkerContext
from digital_markers.data.participant import ParticipantData
from digital_markers.data.validation import require_marker_columns
from digital_markers.requirements import MarkerRequirements
from digital_markers.results import MarkerResult

MOTION_STATES = ("still", "light", "vigorous")
LOCATION_PERIODS = {
    "night": range(6),
    "morning": range(6, 12),
    "afternoon": range(12, 18),
    "evening": range(18, 24),
}
INTERVAL_COLUMNS = (
    "id",
    "time",
    "motion",
    "location",
    "persistence",
    "sustained_fraction",
    "motion_seconds",
    "location_seconds",
    "joint_seconds",
    "observed_minutes",
    "motion_distribution",
    "location_distribution",
    "joint_distribution",
)
REGION_COLUMNS = (
    "id",
    "region_id",
    "latitude",
    "longitude",
    "dwell_minutes",
    "visits",
    "distinct_days",
)
BALANCE_COLUMNS = ("id", "modality", "state", "seconds", "fraction", "valid_seconds")

REQUIREMENTS = MarkerRequirements(
    required=("motion",),
    optional=("location",),
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
        "motion.latitude",
        "motion.longitude",
        "location.timestamp",
        "location.latitude",
        "location.longitude",
    ),
)


@dataclass(frozen=True)
class _Settings:
    rate: int = 10
    min_minute_seconds: float = 30
    min_5min_seconds: float = 150
    min_5min_location_seconds: float = 120
    min_hour_minutes: float = 30
    min_valid_hours_day: int = 6
    min_shared_hours: int = 6
    min_segment_shared_hours: int = 2
    max_routine_regions: int = 6
    sequence_shift_hours: int = 1
    gyro_still: float = 0.07
    acc_light: float = 0.025
    acc_vigorous: float = 0.20
    low_motion_flag: float = 0.003
    sustained_minutes: int = 3
    region_radius_m: float = 125
    region_merge_m: float = 150
    familiar_min_visits: int = 3
    familiar_min_days: int = 2
    transit_max_gap_minutes: float = 5
    transit_min_displacement_m: float = 100
    transit_min_speed_mps: float = 0.5
    transit_max_speed_mps: float = 55
    dwell_minutes: float = 7
    gps_gap_minutes: float = 2
    gps_max_age_minutes: float = 1


_EARTH_RADIUS_M = 6_371_008.8
_ACCELERATION_COLUMNS = ("acceleration_x", "acceleration_y", "acceleration_z")
_GYROSCOPE_COLUMNS = ("rotation_rate_x", "rotation_rate_y", "rotation_rate_z")


def _local_clock(series: pd.Series, timezone: str | None) -> pd.Series:
    timestamps = series
    if isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        timestamps = timestamps.dt.tz_convert(timezone or "UTC").dt.tz_localize(None)
    return timestamps


def _validate_motion(data: ParticipantData) -> tuple[pd.DataFrame, bool]:
    if data.motion is None or data.motion.empty:
        raise ValueError("Missing required data for routine: motion")

    motion = data.motion.copy()
    require_marker_columns({"motion": motion}, REQUIREMENTS.required_columns)
    if not pd.api.types.is_datetime64_any_dtype(motion["timestamp"]):
        raise ValueError("motion.timestamp must have a datetime dtype before calling routine")
    for column in _ACCELERATION_COLUMNS:
        if not pd.api.types.is_numeric_dtype(motion[column]):
            raise ValueError(f"motion.{column} must have a numeric dtype before calling routine")

    gyro_present = [column in motion.columns for column in _GYROSCOPE_COLUMNS]
    if any(gyro_present) and not all(gyro_present):
        raise ValueError(
            "routine requires all three rotation-rate columns when gyro data is present"
        )
    if all(gyro_present):
        for column in _GYROSCOPE_COLUMNS:
            if not pd.api.types.is_numeric_dtype(motion[column]):
                raise ValueError(
                    f"motion.{column} must have a numeric dtype before calling routine"
                )
    else:
        for column in _GYROSCOPE_COLUMNS:
            motion[column] = np.nan

    motion["timestamp"] = _local_clock(motion["timestamp"], data.timezone)
    motion = (
        motion.dropna(subset=["timestamp"])
        .sort_values("timestamp", kind="stable")
        .drop_duplicates("timestamp", keep="first")
        .reset_index(drop=True)
    )
    if motion.empty:
        raise ValueError("routine motion data has no valid timestamps")
    return motion, all(gyro_present)


def _gps_source(data: ParticipantData, motion: pd.DataFrame) -> pd.DataFrame:
    location = data.location
    source: pd.DataFrame | None = None
    if location is not None and not location.empty:
        has_latitude = "latitude" in location.columns
        has_longitude = "longitude" in location.columns
        if has_latitude != has_longitude:
            raise ValueError("routine location data must contain both latitude and longitude")
        if has_latitude:
            require_marker_columns(
                {"location": location},
                ("location.timestamp", "location.latitude", "location.longitude"),
            )
            source = location[["timestamp", "latitude", "longitude"]].copy()

    if source is None:
        has_latitude = "latitude" in motion.columns
        has_longitude = "longitude" in motion.columns
        if has_latitude != has_longitude:
            raise ValueError("routine motion data must contain both latitude and longitude")
        if has_latitude:
            source = motion[["timestamp", "latitude", "longitude"]].copy()

    if source is None:
        return pd.DataFrame(columns=["time", "lat", "lon"])

    if not pd.api.types.is_datetime64_any_dtype(source["timestamp"]):
        raise ValueError("location.timestamp must have a datetime dtype before calling routine")
    for column in ("latitude", "longitude"):
        if not pd.api.types.is_numeric_dtype(source[column]):
            raise ValueError(f"location.{column} must have a numeric dtype before calling routine")

    source = source.rename(columns={"timestamp": "time", "latitude": "lat", "longitude": "lon"})
    source["time"] = _local_clock(source["time"], data.timezone)
    source = source.dropna(subset=["time", "lat", "lon"]).sort_values("time", kind="stable")
    return source.drop_duplicates("time", keep="first").reset_index(drop=True)


def _minute_features(motion: pd.DataFrame, settings: _Settings) -> pd.DataFrame:
    times = motion["timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64")
    acceleration = motion.loc[:, _ACCELERATION_COLUMNS].to_numpy(dtype=np.float64)
    gyroscope = motion.loc[:, _GYROSCOPE_COLUMNS].to_numpy(dtype=np.float64)

    baseline = (
        pd.DataFrame(acceleration).rolling(21, center=True, min_periods=5).median().to_numpy()
    )
    dynamic_acceleration = np.linalg.norm(acceleration - baseline, axis=1)
    rotation = np.linalg.norm(gyroscope, axis=1)

    minute_ids = times // 60_000_000_000
    unique_minutes, group_index = np.unique(minute_ids, return_inverse=True)
    group_count = len(unique_minutes)

    tenths = times // 100_000_000
    distinct_tenths = np.r_[True, tenths[1:] != tenths[:-1]]
    coverage = np.minimum(
        60.0,
        np.bincount(group_index[distinct_tenths], minlength=group_count) / settings.rate,
    )

    dynamic_valid = np.isfinite(dynamic_acceleration)
    rotation_valid = np.isfinite(rotation)
    dynamic_count = np.bincount(group_index[dynamic_valid], minlength=group_count)
    rotation_count = np.bincount(group_index[rotation_valid], minlength=group_count)
    dynamic_squared = np.bincount(
        group_index[dynamic_valid],
        weights=dynamic_acceleration[dynamic_valid] ** 2,
        minlength=group_count,
    )
    rotation_squared = np.bincount(
        group_index[rotation_valid],
        weights=rotation[rotation_valid] ** 2,
        minlength=group_count,
    )
    dynamic_rms = np.sqrt(
        np.divide(
            dynamic_squared,
            dynamic_count,
            out=np.full(group_count, np.nan),
            where=dynamic_count > 0,
        )
    )
    rotation_rms = np.sqrt(
        np.divide(
            rotation_squared,
            rotation_count,
            out=np.full(group_count, np.nan),
            where=rotation_count > 0,
        )
    )
    dynamic_p90 = (
        pd.Series(dynamic_acceleration)
        .groupby(group_index, sort=False)
        .quantile(0.9)
        .reindex(range(group_count))
        .to_numpy()
    )
    valid = (coverage >= settings.min_minute_seconds) & (
        dynamic_count >= settings.min_minute_seconds * settings.rate * 0.7
    )
    index = pd.DatetimeIndex(pd.to_datetime(unique_minutes, unit="m"), name="time")
    return pd.DataFrame(
        {
            "seconds": coverage,
            "valid_motion": valid,
            "acc_rms": dynamic_rms,
            "acc_p90": dynamic_p90,
            "gyro_rms": rotation_rms,
        },
        index=index,
    )


def _classify_motion(minutes: pd.DataFrame, settings: _Settings) -> pd.DataFrame:
    result = minutes.copy()
    valid = result["valid_motion"].to_numpy(dtype=bool, copy=True)
    acceleration_rms = result["acc_rms"].to_numpy(dtype=float)
    rotation_rms = result["gyro_rms"].to_numpy(dtype=float)
    valid &= np.isfinite(acceleration_rms)

    labels = np.full(len(result), "unknown", dtype=object)
    labels[valid & (acceleration_rms < settings.acc_light)] = "still"
    labels[
        valid
        & (acceleration_rms >= settings.acc_light)
        & (acceleration_rms < settings.acc_vigorous)
    ] = "light"
    labels[valid & (acceleration_rms >= settings.acc_vigorous)] = "vigorous"
    result["motion"] = labels
    result["acc_component"] = acceleration_rms / settings.acc_light
    result["gyro_component"] = rotation_rms / settings.gyro_still
    result["intensity"] = acceleration_rms

    active = valid & (labels != "still")
    sustained = np.zeros(len(result), dtype=bool)
    timestamps = result.index.as_unit("ns").asi8
    start: int | None = None
    for i in range(len(result) + 1):
        connected = (
            i < len(result)
            and active[i]
            and (start is None or i == start or timestamps[i] - timestamps[i - 1] == 60_000_000_000)
        )
        if not connected:
            if start is not None and i - start >= settings.sustained_minutes:
                sustained[start:i] = True
            start = i if i < len(result) and active[i] else None
        elif start is None:
            start = i

    result["persistence"] = np.where(
        ~valid,
        "unknown",
        np.where(sustained, "sustained", "intermittent"),
    )
    return result


def _motion_diagnostics(
    minutes: pd.DataFrame, participant_id: str, settings: _Settings
) -> dict[str, Any]:
    valid = minutes.loc[minutes["motion"] != "unknown"]
    row: dict[str, Any] = {
        "id": participant_id,
        "valid_minutes": len(valid),
        "valid_motion_seconds": float(valid["seconds"].sum()),
        "acc_light_threshold": settings.acc_light,
        "acc_vigorous_threshold": settings.acc_vigorous,
        "classification": "acceleration_led",
    }
    for field in ("acc_rms", "gyro_rms", "acc_component", "gyro_component", "intensity"):
        values = valid[field].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        for percentile in (10, 25, 50, 75, 90, 95):
            row[f"{field}_p{percentile}"] = (
                float(np.percentile(values, percentile)) if len(values) else np.nan
            )
    for state in MOTION_STATES:
        row[f"fraction_{state}"] = (
            float(np.mean(valid["motion"] == state)) if len(valid) else np.nan
        )
    row["fraction_sustained"] = (
        float(np.mean(valid["persistence"] == "sustained")) if len(valid) else np.nan
    )
    row["low_motion_signal_flag"] = bool(
        len(valid) and np.nanmedian(valid["acc_rms"]) < settings.low_motion_flag
    )
    return row


def _haversine(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> Any:
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    latitude_delta = lat2_rad - lat1_rad
    longitude_delta = np.radians(np.asarray(lon2) - np.asarray(lon1))
    haversine = (
        np.sin(latitude_delta / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(longitude_delta / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))


def _gps_fix_minutes(gps: pd.DataFrame) -> pd.DataFrame:
    fixes = gps.loc[
        gps["lat"].between(-90, 90) & gps["lon"].between(-180, 180),
        ["time", "lat", "lon"],
    ].copy()
    fixes = fixes.loc[~((fixes["lat"] == 0) & (fixes["lon"] == 0))]
    if fixes.empty:
        return pd.DataFrame(columns=["time", "lat", "lon"])
    fixes["minute"] = fixes["time"].dt.floor("min")
    return fixes.groupby("minute", as_index=False).agg(
        time=("time", "median"),
        lat=("lat", "median"),
        lon=("lon", "median"),
    )


def _build_regions(
    fixes: pd.DataFrame, settings: _Settings, participant_id: str
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    if fixes.empty:
        empty = pd.DataFrame(columns=["time", "lat", "lon", "location", "region_id"])
        diagnostics = {
            "id": participant_id,
            "gps_fixes": 0,
            "observed_transit_edges": 0,
            "unsupported_region_changes": 0,
            "labeled_fix_fraction": np.nan,
            "region_count": 0,
        }
        return [], empty, diagnostics

    points = fixes.sort_values("time").reset_index(drop=True).copy()
    timestamps = points["time"].dt.as_unit("ns").astype("int64").to_numpy(dtype=np.float64) / 1e9
    latitude = points["lat"].to_numpy(dtype=float)
    longitude = points["lon"].to_numpy(dtype=float)
    episodes: list[dict[str, Any]] = []
    start = 0
    while start < len(points):
        stop = start + 1
        while (
            stop < len(points)
            and 0 < timestamps[stop] - timestamps[stop - 1] <= settings.gps_gap_minutes * 60
        ):
            distance = _haversine(
                latitude[stop],
                longitude[stop],
                np.median(latitude[start:stop]),
                np.median(longitude[start:stop]),
            )
            if distance > settings.region_radius_m:
                break
            stop += 1
        duration = (timestamps[stop - 1] - timestamps[start]) / 60 if stop > start else 0
        if stop - start >= 2 and duration >= settings.dwell_minutes:
            minutes = float(
                np.minimum(np.diff(timestamps[start:stop]), settings.gps_gap_minutes * 60).sum()
                / 60
            )
            episodes.append(
                {
                    "start": start,
                    "end": stop,
                    "lat": float(np.median(latitude[start:stop])),
                    "lon": float(np.median(longitude[start:stop])),
                    "minutes": minutes,
                    "day": points["time"].iloc[start].date(),
                }
            )
        start = stop

    groups = [[i] for i in range(len(episodes))]

    def can_merge(left: list[int], right: list[int]) -> bool:
        for left_index in left:
            for right_index in right:
                left_episode = episodes[left_index]
                right_episode = episodes[right_index]
                distance = _haversine(
                    left_episode["lat"],
                    left_episode["lon"],
                    right_episode["lat"],
                    right_episode["lon"],
                )
                if float(distance) > settings.region_merge_m:
                    return False
                if (
                    left_episode["start"] < right_episode["end"]
                    and right_episode["start"] < left_episode["end"]
                ):
                    return False
        return True

    while True:
        best: tuple[float, int, int] | None = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if can_merge(groups[i], groups[j]):
                    left_episode = episodes[groups[i][0]]
                    right_episode = episodes[groups[j][0]]
                    distance = float(
                        _haversine(
                            left_episode["lat"],
                            left_episode["lon"],
                            right_episode["lat"],
                            right_episode["lon"],
                        )
                    )
                    candidate = (distance, i, j)
                    if best is None or candidate < best:
                        best = candidate
        if best is None:
            break
        _, i, j = best
        groups[i].extend(groups.pop(j))

    regions: list[dict[str, Any]] = []
    assigned: dict[int, int] = {}
    for group in groups:
        members = [episodes[index] for index in group]
        weights = np.array([episode["minutes"] for episode in members])
        regions.append(
            {
                "lat": float(np.average([ep["lat"] for ep in members], weights=weights)),
                "lon": float(np.average([ep["lon"] for ep in members], weights=weights)),
                "minutes": float(weights.sum()),
                "visits": len(members),
                "distinct_days": len({ep["day"] for ep in members}),
            }
        )
        for index in group:
            assigned[index] = len(regions) - 1

    order = sorted(
        range(len(regions)),
        key=lambda index: (-regions[index]["minutes"], -regions[index]["visits"]),
    )
    rank = {old: new for new, old in enumerate(order)}
    records = [
        {
            "id": participant_id,
            "region_id": rank[index] + 1,
            "latitude": regions[index]["lat"],
            "longitude": regions[index]["lon"],
            "dwell_minutes": regions[index]["minutes"],
            "visits": regions[index]["visits"],
            "distinct_days": regions[index]["distinct_days"],
        }
        for index in order
    ]

    labels = np.full(len(points), "unknown", dtype=object)
    region_ids = np.full(len(points), -1, dtype=int)
    prior_visits: dict[int, int] = {}
    prior_days: dict[int, set[object]] = {}
    for episode_index, episode in sorted(enumerate(episodes), key=lambda item: item[1]["start"]):
        group_index = assigned[episode_index]
        previous_visits = prior_visits.get(group_index, 0)
        days = prior_days.get(group_index, set())
        if previous_visits == 0:
            label = "novel"
        elif rank[group_index] == 0:
            label = "primary"
        elif (
            previous_visits >= settings.familiar_min_visits
            and len(days) >= settings.familiar_min_days
        ):
            label = "familiar_secondary"
        else:
            label = "other_visited"
        labels[episode["start"] : episode["end"]] = label
        region_ids[episode["start"] : episode["end"]] = rank[group_index] + 1
        prior_visits[group_index] = previous_visits + 1
        prior_days[group_index] = days | {episode["day"]}

    observed_transit_edges = 0
    for index in range(1, len(points)):
        elapsed = timestamps[index] - timestamps[index - 1]
        if not (0 < elapsed <= settings.transit_max_gap_minutes * 60):
            continue
        distance = float(
            _haversine(
                latitude[index - 1],
                longitude[index - 1],
                latitude[index],
                longitude[index],
            )
        )
        speed = distance / elapsed
        if (
            distance >= settings.transit_min_displacement_m
            and settings.transit_min_speed_mps <= speed <= settings.transit_max_speed_mps
        ):
            observed_transit_edges += 1
            if region_ids[index] == -1:
                labels[index] = "transit"
            if region_ids[index - 1] == -1:
                labels[index - 1] = "transit"

    supported = np.flatnonzero(region_ids >= 0)
    unsupported_changes = 0
    for left, right in pairwise(supported):
        if (
            region_ids[left] != region_ids[right]
            and timestamps[right] - timestamps[left] > settings.transit_max_gap_minutes * 60
        ):
            unsupported_changes += 1

    points["location"] = labels
    points["region_id"] = region_ids
    diagnostics = {
        "id": participant_id,
        "gps_fixes": len(points),
        "observed_transit_edges": observed_transit_edges,
        "unsupported_region_changes": unsupported_changes,
        "labeled_fix_fraction": float(np.mean(labels != "unknown")),
        "region_count": len(records),
    }
    return records, points, diagnostics


def _align_location(minutes: pd.DataFrame, gps: pd.DataFrame, settings: _Settings) -> pd.DataFrame:
    result = minutes.copy()
    result["region_id"] = -1
    result["location"] = "unknown"
    if gps.empty:
        return result

    left = result.drop(columns=["region_id", "location"]).reset_index().sort_values("time")
    right = (
        gps[["time", "region_id", "location"]]
        .rename(columns={"time": "gps_time"})
        .sort_values("gps_time")
    )
    left["time"] = left["time"].astype(right["gps_time"].dtype)
    matched = pd.merge_asof(
        left,
        right,
        left_on="time",
        right_on="gps_time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=settings.gps_max_age_minutes),
    )
    result["region_id"] = matched["region_id"].fillna(-1).to_numpy(dtype=int)
    result["location"] = matched["location"].fillna("unknown").to_numpy()
    return result


def _five_minute_intervals(
    minutes: pd.DataFrame, settings: _Settings, participant_id: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stamp, group in minutes.groupby(minutes.index.floor("5min"), sort=True):
        valid_motion = group.loc[group["motion"].isin(MOTION_STATES)]
        motion_seconds = float(valid_motion["seconds"].sum())
        located = group.loc[group["region_id"] > 0]
        location_seconds = float(located["seconds"].sum())
        paired = group.loc[group["motion"].isin(MOTION_STATES) & (group["region_id"] > 0)]
        joint_seconds = float(paired["seconds"].sum())

        motion_distribution = dict.fromkeys(MOTION_STATES, 0.0)
        if motion_seconds >= settings.min_5min_seconds:
            motion_distribution.update(
                valid_motion.groupby("motion")["seconds"].sum().div(motion_seconds).to_dict()
            )
        else:
            motion_seconds = 0.0

        location_distribution: dict[int, float] = {}
        if location_seconds >= settings.min_5min_location_seconds:
            location_distribution = {
                int(region_id): float(seconds / location_seconds)
                for region_id, seconds in located.groupby("region_id")["seconds"].sum().items()
            }
        else:
            location_seconds = 0.0

        joint_distribution: dict[str, float] = {}
        if (
            motion_seconds
            and location_seconds
            and joint_seconds >= min(settings.min_5min_seconds, settings.min_5min_location_seconds)
        ):
            joint_distribution = {
                f"{state}:{int(region_id)}": float(seconds / joint_seconds)
                for (state, region_id), seconds in paired.groupby(["motion", "region_id"])[
                    "seconds"
                ]
                .sum()
                .items()
            }
        else:
            joint_seconds = 0.0

        sustained_seconds = valid_motion.loc[
            valid_motion["persistence"] == "sustained", "seconds"
        ].sum()
        sustained_fraction = float(sustained_seconds / motion_seconds) if motion_seconds else np.nan
        rows.append(
            {
                "id": participant_id,
                "time": stamp,
                "motion": (
                    max(motion_distribution, key=motion_distribution.get)
                    if motion_seconds
                    else "unknown"
                ),
                "location": (
                    max(location_distribution, key=location_distribution.get)
                    if location_seconds
                    else "unknown"
                ),
                "persistence": (
                    "sustained"
                    if motion_seconds and sustained_seconds / motion_seconds >= 0.5
                    else "intermittent"
                    if motion_seconds
                    else "unknown"
                ),
                "sustained_fraction": sustained_fraction,
                "motion_seconds": motion_seconds,
                "location_seconds": location_seconds,
                "joint_seconds": joint_seconds,
                "observed_minutes": int(group["valid_motion"].sum()),
                "motion_distribution": (
                    json.dumps(motion_distribution) if motion_seconds else "{}"
                ),
                "location_distribution": json.dumps(location_distribution),
                "joint_distribution": json.dumps(joint_distribution),
            }
        )
    return pd.DataFrame(rows, columns=INTERVAL_COLUMNS)


def _js_similarity_rows(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / first.sum(axis=-1, keepdims=True)
    second = second / second.sum(axis=-1, keepdims=True)
    midpoint = (first + second) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        first_kl = np.where(first > 0, first * np.log2(first / midpoint), 0).sum(axis=-1)
        second_kl = np.where(second > 0, second * np.log2(second / midpoint), 0).sum(axis=-1)
    return np.clip(1 - np.sqrt(np.maximum(0, (first_kl + second_kl) / 2)), 0, 1)


def _sequence_similarity(
    profile_a: np.ndarray,
    profile_b: np.ndarray,
    hours: np.ndarray,
    max_shift: int,
) -> float:
    if not len(hours):
        return np.nan
    distance = 1 - _js_similarity_rows(profile_a[:, None, :], profile_b[None, :, :])
    cost = np.full((len(hours) + 1, len(hours) + 1), np.inf)
    steps = np.zeros((len(hours) + 1, len(hours) + 1), dtype=np.int32)
    cost[0, 0] = 0
    for i in range(1, len(hours) + 1):
        for j in range(1, len(hours) + 1):
            if abs(int(hours[i - 1]) - int(hours[j - 1])) > max_shift:
                continue
            candidates = (
                (cost[i - 1, j - 1], steps[i - 1, j - 1]),
                (cost[i - 1, j], steps[i - 1, j]),
                (cost[i, j - 1], steps[i, j - 1]),
            )
            best, length = min(candidates, key=lambda item: item[0])
            if np.isfinite(best):
                cost[i, j] = best + distance[i - 1, j - 1]
                steps[i, j] = length + 1
    if steps[-1, -1]:
        return float(np.clip(1 - cost[-1, -1] / steps[-1, -1], 0, 1))
    return np.nan


def _summarize(
    intervals: pd.DataFrame,
    settings: _Settings,
    participant_id: str,
    region_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if intervals.empty:
        return {}, []

    intervals = intervals.copy()
    intervals["date"] = intervals["time"].dt.floor("D")
    intervals["hour"] = intervals["time"].dt.hour
    dates = np.sort(intervals["date"].unique())
    day_count = len(dates)
    top_regions = max(1, min(settings.max_routine_regions, region_count))
    location_dimension = top_regions + 1
    dimensions = {
        "motion": len(MOTION_STATES),
        "location": location_dimension,
        "joint": len(MOTION_STATES) * location_dimension,
    }
    profiles = {
        name: np.full((day_count, 24, size), np.nan, dtype=np.float32)
        for name, size in dimensions.items()
    }
    seconds = {name: np.zeros((day_count, 24), dtype=np.float32) for name in dimensions}
    sustained = np.full((day_count, 24), np.nan, dtype=np.float32)

    def region_bin(region_id: int) -> int:
        return region_id - 1 if 1 <= region_id <= top_regions else top_regions

    for (date, hour), chunk in intervals.groupby(["date", "hour"], sort=True):
        day_index = int(np.searchsorted(dates, np.datetime64(date)))
        totals = {name: np.zeros(size, dtype=np.float64) for name, size in dimensions.items()}
        persistence_sum = 0.0
        persistence_denominator = 0.0
        for row in chunk.itertuples():
            for name, field in (
                ("motion", "motion_distribution"),
                ("location", "location_distribution"),
                ("joint", "joint_distribution"),
            ):
                weight = float(getattr(row, f"{name}_seconds"))
                if not weight:
                    continue
                distribution = json.loads(getattr(row, field))
                for key, probability in distribution.items():
                    if name == "motion":
                        index = MOTION_STATES.index(key)
                    elif name == "location":
                        index = region_bin(int(key))
                    else:
                        state, region_id = key.split(":")
                        index = MOTION_STATES.index(state) * location_dimension + region_bin(
                            int(region_id)
                        )
                    totals[name][index] += weight * probability
                seconds[name][day_index, hour] += weight
            if row.motion_seconds and np.isfinite(row.sustained_fraction):
                persistence_sum += row.motion_seconds * row.sustained_fraction
                persistence_denominator += row.motion_seconds
        if persistence_denominator:
            sustained[day_index, hour] = persistence_sum / persistence_denominator
        for name in dimensions:
            total = totals[name].sum()
            if seconds[name][day_index, hour] >= settings.min_hour_minutes * 60 and total > 0:
                profiles[name][day_index, hour] = totals[name] / total

    result: dict[str, Any] = {
        "id": participant_id,
        "n_observed_days": day_count,
        "n_intervals": len(intervals),
    }

    for name, profile in profiles.items():
        mask = np.isfinite(profile[:, :, 0])
        eligible = mask.sum(axis=1) >= settings.min_valid_hours_day
        eligible_days = np.flatnonzero(eligible)
        result[f"{name}_valid_days"] = int(eligible.sum())
        result[f"{name}_valid_hours"] = int(mask.sum())

        pairs: list[float] = []
        sequences: list[float] = []
        lag1: list[float] = []
        lag7: list[float] = []
        specificity: list[float] = []
        weighted_pairs: list[float] = []
        overlapping_hours: list[int] = []
        overlap_coverage: list[float] = []
        segment_scores = {period: [] for period in LOCATION_PERIODS}

        for hour in range(24):
            days = np.flatnonzero(eligible & mask[:, hour])
            values: list[float] = []
            for position, day_index in enumerate(days[:-1]):
                other_days = days[position + 1 :]
                first = np.broadcast_to(
                    profile[day_index, hour], (len(other_days), profile.shape[2])
                )
                values.extend(_js_similarity_rows(first, profile[other_days, hour]).tolist())
            prefix = f"{name}_hour_{hour:02d}"
            result[f"{prefix}_similarity_mean"] = float(np.mean(values)) if values else np.nan
            result[f"{prefix}_similarity_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            )
            result[f"{prefix}_day_pairs"] = len(values)

        within_day: list[float] = []
        for day_index in eligible_days:
            hours = np.flatnonzero(mask[day_index])
            adjacent = hours[:-1][np.diff(hours) == 1]
            if len(adjacent):
                distances = 1 - _js_similarity_rows(
                    profile[day_index, adjacent], profile[day_index, adjacent + 1]
                )
                within_day.extend(distances.tolist())
        result[f"{name}_within_day_structure_mean"] = (
            float(np.mean(within_day)) if within_day else np.nan
        )
        result[f"{name}_within_day_structure_sd"] = (
            float(np.std(within_day, ddof=1)) if len(within_day) > 1 else np.nan
        )
        result[f"{name}_within_day_adjacent_pairs"] = len(within_day)
        result[f"{name}_mean_valid_hours_day"] = (
            float(mask[eligible].sum(axis=1).mean()) if eligible.any() else np.nan
        )
        result[f"{name}_mean_hour_coverage"] = (
            float(np.mean(seconds[name][mask] / 3600)) if mask.any() else np.nan
        )

        for position, first_day in enumerate(eligible_days):
            for second_day in eligible_days[position + 1 :]:
                shared = np.flatnonzero(mask[first_day] & mask[second_day])
                if len(shared) < settings.min_shared_hours:
                    continue
                hourly = _js_similarity_rows(
                    profile[first_day, shared], profile[second_day, shared]
                )
                score = float(hourly.mean())
                pairs.append(score)
                overlapping_hours.append(len(shared))
                weights = np.clip(
                    np.minimum(
                        seconds[name][first_day, shared],
                        seconds[name][second_day, shared],
                    )
                    / 3600,
                    0,
                    1,
                )
                weighted_pairs.append(
                    float(np.average(hourly, weights=weights)) if weights.sum() > 0 else np.nan
                )
                overlap_coverage.append(float(np.mean(weights)))
                for period, hour_range in LOCATION_PERIODS.items():
                    selected = np.isin(shared, list(hour_range))
                    if selected.sum() >= settings.min_segment_shared_hours:
                        segment_scores[period].append(float(hourly[selected].mean()))
                sequences.append(
                    _sequence_similarity(
                        profile[first_day, shared],
                        profile[second_day, shared],
                        shared,
                        settings.sequence_shift_hours,
                    )
                )
                gap = int((dates[second_day] - dates[first_day]) / np.timedelta64(1, "D"))
                if gap == 1:
                    lag1.append(score)
                if gap == 7:
                    lag7.append(score)
                if len(shared) >= 2:
                    shifted = np.roll(shared, 1)
                    mismatch = _js_similarity_rows(
                        profile[first_day, shared], profile[second_day, shifted]
                    ).mean()
                    specificity.append(score - float(mismatch))

        result[f"{name}_day_similarity_coverage_normalized_mean"] = (
            float(np.nanmean(weighted_pairs)) if weighted_pairs else np.nan
        )
        result[f"{name}_day_similarity_coverage_normalized_sd"] = (
            float(np.nanstd(weighted_pairs, ddof=1)) if len(weighted_pairs) > 1 else np.nan
        )
        result[f"{name}_mean_shared_hours"] = (
            float(np.mean(overlapping_hours)) if overlapping_hours else np.nan
        )
        result[f"{name}_mean_shared_hour_coverage"] = (
            float(np.mean(overlap_coverage)) if overlap_coverage else np.nan
        )

        for period, values in segment_scores.items():
            prefix = f"{name}_{period}"
            result[f"{prefix}_similarity_mean"] = float(np.mean(values)) if values else np.nan
            result[f"{prefix}_similarity_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            )
            result[f"{prefix}_day_pairs"] = len(values)

        result[f"{name}_day_similarity_mean"] = float(np.mean(pairs)) if pairs else np.nan
        result[f"{name}_day_similarity_sd"] = (
            float(np.std(pairs, ddof=1)) if len(pairs) > 1 else np.nan
        )
        result[f"{name}_day_similarity_lag1"] = float(np.mean(lag1)) if lag1 else np.nan
        result[f"{name}_day_similarity_lag7"] = float(np.mean(lag7)) if lag7 else np.nan
        finite_sequences = np.asarray(sequences, dtype=float)
        finite_sequences = finite_sequences[np.isfinite(finite_sequences)]
        result[f"{name}_sequence_similarity_mean"] = (
            float(finite_sequences.mean()) if len(finite_sequences) else np.nan
        )
        result[f"{name}_sequence_similarity_sd"] = (
            float(finite_sequences.std(ddof=1)) if len(finite_sequences) > 1 else np.nan
        )
        result[f"{name}_time_specificity"] = float(np.mean(specificity)) if specificity else np.nan
        result[f"{name}_day_pairs"] = len(pairs)

        distances: list[float] = []
        elapsed_days: list[float] = []
        for day_index in eligible_days:
            similarities: list[float] = []
            for hour in np.flatnonzero(mask[day_index]):
                others = np.flatnonzero(
                    eligible & mask[:, hour] & (np.arange(day_count) != day_index)
                )
                if len(others) < 2:
                    continue
                reference = profile[others, hour].mean(axis=0)
                similarity = _js_similarity_rows(
                    profile[day_index, hour][None, :], reference[None, :]
                )[0]
                similarities.append(float(similarity))
            if len(similarities) >= settings.min_shared_hours:
                distances.append(1 - float(np.mean(similarities)))
                elapsed_days.append(float((dates[day_index] - dates[0]) / np.timedelta64(1, "D")))

        result[f"{name}_typical_distance_mean"] = float(np.mean(distances)) if distances else np.nan
        result[f"{name}_typical_distance_p90"] = (
            float(np.quantile(distances, 0.9)) if distances else np.nan
        )
        result[f"{name}_fraction_atypical"] = np.nan
        result[f"{name}_drift_slope"] = np.nan
        if len(distances) >= 5:
            values = np.asarray(distances)
            median = np.median(values)
            mad = 1.4826 * np.median(np.abs(values - median))
            if mad > 1e-8:
                result[f"{name}_fraction_atypical"] = float(np.mean((values - median) / mad > 2.5))
            if len(set(elapsed_days)) > 1:
                result[f"{name}_drift_slope"] = float(np.polyfit(elapsed_days, distances, 1)[0])
        result[f"{name}_unknown_intervals"] = int((intervals[f"{name}_seconds"] == 0).sum())

    result["sustained_hourly_mean"] = (
        float(np.nanmean(sustained)) if np.isfinite(sustained).any() else np.nan
    )

    balance: list[dict[str, Any]] = []
    for modality in ("motion", "location"):
        weights: defaultdict[str, float] = defaultdict(float)
        for row in intervals.itertuples():
            duration = float(getattr(row, f"{modality}_seconds"))
            if duration <= 0:
                continue
            distribution = json.loads(getattr(row, f"{modality}_distribution"))
            for key, fraction in distribution.items():
                label = key if modality == "motion" else f"region_{key}"
                weights[label] += duration * fraction
        total = sum(weights.values())
        labels = list(MOTION_STATES) if modality == "motion" else sorted(weights)
        for label in labels:
            value = weights[label]
            balance.append(
                {
                    "id": participant_id,
                    "modality": modality,
                    "state": label,
                    "seconds": float(value),
                    "fraction": float(value / total) if total else np.nan,
                    "valid_seconds": total,
                }
            )
        fractions = (
            np.asarray([value / total for value in weights.values() if value > 0])
            if total
            else np.array([])
        )
        result[f"{modality}_balance_entropy"] = (
            float(-(fractions * np.log2(fractions)).sum() / np.log2(len(weights)))
            if len(weights) > 1
            else np.nan
        )
        result[f"{modality}_dominant_fraction"] = (
            float(max(weights.values()) / total) if total else np.nan
        )
    return result, balance


def compute_routine_markers(data: ParticipantData) -> MarkerResult:
    """Compute fixed-threshold routine distributions and regularity features.

    Acceleration is required. Rotation-rate columns and GPS coordinates are
    optional; GPS can be supplied in ``ParticipantData.location`` or alongside
    motion samples. Result details contain five-minute intervals, state balance,
    inferred stay regions, and diagnostic tables. No model or profiles are used.
    """
    settings = _Settings()
    motion, has_gyroscope = _validate_motion(data)
    gps = _gps_source(data, motion)
    minutes = _classify_motion(_minute_features(motion, settings), settings)
    motion_diagnostics = _motion_diagnostics(minutes, data.participant_id, settings)
    fixes = _gps_fix_minutes(gps)
    regions, labeled_fixes, location_diagnostics = _build_regions(
        fixes, settings, data.participant_id
    )
    minutes = _align_location(minutes, labeled_fixes, settings)
    intervals = _five_minute_intervals(minutes, settings, data.participant_id)
    summary, balance = _summarize(intervals, settings, data.participant_id, len(regions))
    markers = pd.DataFrame([summary]) if summary else pd.DataFrame()
    state_balance = pd.DataFrame(balance, columns=BALANCE_COLUMNS)
    region_table = pd.DataFrame(regions, columns=REGION_COLUMNS)
    context = MarkerContext(
        data=data,
        derived={"routine_intervals": intervals.set_index("time").to_dict("index")},
    )
    return MarkerResult(
        markers=markers,
        details={
            "intervals": intervals,
            "state_balance": state_balance,
            "regions": region_table,
            "motion_diagnostics": pd.DataFrame([motion_diagnostics]),
            "location_diagnostics": pd.DataFrame([location_diagnostics]),
        },
        metadata={
            "marker": "routine",
            "participant_id": data.participant_id,
            "sampling_rate_hz": settings.rate,
            "has_gyroscope": has_gyroscope,
            "has_location": not fixes.empty,
            "acc_light_threshold": settings.acc_light,
            "acc_vigorous_threshold": settings.acc_vigorous,
        },
        context=context,
    )


__all__ = ["REQUIREMENTS", "compute_routine_markers"]

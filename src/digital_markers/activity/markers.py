"""Compute daily activity-label markers and participant-level summaries."""

import re

import numpy as np
import pandas as pd

from digital_markers.context import MarkerContext
from digital_markers.data.participant import ParticipantData
from digital_markers.data.validation import require_marker_columns
from digital_markers.requirements import MarkerRequirements
from digital_markers.results import MarkerResult

PERIODS = {
    "night": (0, 6),
    "morning": (6, 12),
    "afternoon": (12, 18),
    "evening": (18, 24),
}
UNKNOWN_VALUES = {
    "",
    "unknown",
    "unlabeled",
    "unlabelled",
    "none",
    "null",
    "nan",
    "na",
    "n/a",
    "missing",
}
SLOT_NS = 100_000_000
REQUIREMENTS = MarkerRequirements(
    required=("activity",),
    required_columns=("activity.timestamp", "activity.activity"),
)


def _state_name(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    state = str(value).strip().lower()
    return "unknown" if state in UNKNOWN_VALUES else state


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "state"


def _state_slugs(states: list[str]) -> dict[str, str]:
    slugs: dict[str, str] = {}
    used: set[str] = set()
    for state in states:
        base = _slugify(state)
        name = base
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        slugs[state] = name
    return slugs


def _state_entropy(weights: list[float]) -> float:
    values = np.asarray(weights, dtype=float)
    values = values[values > 0]
    if len(values) < 2:
        return 0.0 if len(values) == 1 else np.nan
    probabilities = values / values.sum()
    return float(-(probabilities * np.log2(probabilities)).sum() / np.log2(len(values)))


def _activity_seconds(data: ParticipantData) -> pd.DataFrame:
    if data.activity is None or data.activity.empty:
        raise ValueError("Missing required data for activity: activity")

    activity = data.activity.copy()
    require_marker_columns({"activity": activity}, REQUIREMENTS.required_columns)
    if not pd.api.types.is_datetime64_any_dtype(activity["timestamp"]):
        raise ValueError("activity.timestamp must have a datetime dtype before calling activity")

    timestamps = activity["timestamp"]
    if isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        timestamps = timestamps.dt.tz_convert(data.timezone or "UTC").dt.tz_localize(None)
    valid = timestamps.notna()
    if not valid.any():
        raise ValueError("activity motion data has no valid timestamps")

    samples = pd.DataFrame(
        {
            "timestamp": timestamps.loc[valid].reset_index(drop=True),
            "state": activity.loc[valid, "activity"].map(_state_name).reset_index(drop=True),
        }
    )
    samples["slot"] = (
        samples["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64) // SLOT_NS
    )
    samples = samples.drop_duplicates("slot", keep="first")
    samples["second"] = samples["slot"] // 10
    seconds = samples.groupby(["second", "state"], sort=False).size().rename("slots").reset_index()
    return seconds


def _calendar_ac1(values: np.ndarray, dates: pd.Series) -> float:
    x = np.asarray(values, dtype=float)
    time = np.asarray(pd.to_datetime(dates), dtype="datetime64[D]")
    valid = np.isfinite(x) & ~np.isnat(time)
    x, time = x[valid], time[valid]
    if len(x) < 3:
        return np.nan
    order = np.argsort(time)
    x, time = x[order], time[order]
    pairs = np.flatnonzero(np.diff(time) == np.timedelta64(1, "D"))
    if len(pairs) < 3:
        return np.nan
    before, after = x[pairs], x[pairs + 1]
    if np.std(before) == 0 or np.std(after) == 0:
        return np.nan
    return float(np.corrcoef(before, after)[0, 1])


def _calendar_slope(values: np.ndarray, dates: pd.Series) -> float:
    x = np.asarray(values, dtype=float)
    time = np.asarray(pd.to_datetime(dates), dtype="datetime64[D]")
    valid = np.isfinite(x) & ~np.isnat(time)
    x, time = x[valid], time[valid]
    if len(x) < 2 or len(np.unique(time)) < 2:
        return np.nan
    elapsed = (time - time.min()).astype("timedelta64[D]").astype(float)
    centered = elapsed - elapsed.mean()
    return float(np.dot(centered, x - x.mean()) / np.dot(centered, centered))


def _daily_rows(
    seconds: pd.DataFrame,
    states: list[str],
    slugs: dict[str, str],
    include_hourly: bool,
) -> pd.DataFrame:
    frame = seconds.copy()
    frame["date"] = pd.to_datetime(frame["second"], unit="s").dt.date
    frame["hour"] = (frame["second"] // 3600) % 24
    dominant = frame.sort_values(
        ["second", "slots", "state"], ascending=[True, False, True], kind="stable"
    ).drop_duplicates("second")
    rows = []

    for date, daily in frame.groupby("date", sort=True):
        total = float(daily["slots"].sum() / 10.0)
        known_slots = daily.loc[daily["state"].ne("unknown"), "slots"].sum()
        known_seconds = float(known_slots / 10.0)
        if total <= 0:
            continue

        row: dict[str, object] = {
            "date": pd.Timestamp(date),
            "observed_seconds": total,
            "unknown_fraction": (total - known_seconds) / total,
            "num_states": int(
                daily.loc[daily["state"].ne("unknown") & daily["slots"].gt(0), "state"].nunique()
            ),
        }
        by_second = dominant.loc[dominant["date"].eq(date)].sort_values("second")
        observed_seconds = by_second["second"].to_numpy(dtype=np.int64)
        labels = by_second["state"].to_numpy()
        adjacent = np.diff(observed_seconds) == 1
        changes = labels[1:] != labels[:-1]
        known_transitions = (
            adjacent & changes & (labels[1:] != "unknown") & (labels[:-1] != "unknown")
        )
        row["transitions"] = int(known_transitions.sum())
        row["transitions_per_hour"] = (
            row["transitions"] / (known_seconds / 3600.0) if known_seconds else np.nan
        )

        counts = daily.groupby("state")["slots"].sum().to_dict()
        row["activity_entropy"] = _state_entropy(
            [counts.get(state, 0) for state in states if state != "unknown"]
        )
        row["dominant_fraction"] = (
            max((counts.get(state, 0) for state in states if state != "unknown"), default=0)
            / known_slots
            if known_slots
            else np.nan
        )

        starts = np.r_[True, (np.diff(observed_seconds) != 1) | (labels[1:] != labels[:-1])]
        groups = np.cumsum(starts)
        bouts = pd.DataFrame({"group": groups, "state": labels})
        bout_lengths = (
            bouts.groupby(["group", "state"], sort=False).size().reset_index(name="duration")
        )

        for state in states:
            if state == "unknown":
                continue
            key = slugs[state]
            state_slots = counts.get(state, 0)
            state_bouts = bout_lengths.loc[bout_lengths["state"].eq(state), "duration"]
            row[f"{key}_fraction"] = state_slots / known_slots if known_slots else np.nan
            row[f"{key}_bouts"] = len(state_bouts)
            row[f"{key}_bout_mean_seconds"] = (
                float(state_bouts.mean()) if len(state_bouts) else np.nan
            )
            for period, (start, end) in PERIODS.items():
                period_data = daily.loc[daily["hour"].between(start, end - 1)]
                period_slots = period_data["slots"].sum()
                period_state_slots = period_data.loc[period_data["state"].eq(state), "slots"].sum()
                row[f"{key}_{period}_fraction"] = (
                    float(period_state_slots / period_slots) if period_slots else np.nan
                )
                row[f"{key}_{period}_share"] = (
                    float(period_state_slots / state_slots) if state_slots else np.nan
                )
            if include_hourly:
                for hour in range(24):
                    hour_data = daily.loc[daily["hour"].eq(hour)]
                    hour_slots = hour_data["slots"].sum()
                    state_hour_slots = hour_data.loc[hour_data["state"].eq(state), "slots"].sum()
                    row[f"{key}_hour_{hour:02d}_fraction"] = (
                        float(state_hour_slots / hour_slots) if hour_slots else np.nan
                    )
        rows.append(row)

    return pd.DataFrame(rows)


def _aggregate_daily(daily: pd.DataFrame) -> pd.DataFrame:
    result: dict[str, object] = {
        "num_recorded_days": len(daily),
        "mean_observed_seconds": float(daily["observed_seconds"].mean()),
        "mean_unknown_fraction": float(daily["unknown_fraction"].mean()),
    }
    for column in daily.select_dtypes(include=[np.number]).columns:
        if column in {"observed_seconds", "unknown_fraction"}:
            continue
        values = daily[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        result[f"mean_{column}"] = float(finite.mean()) if len(finite) else np.nan
        result[f"std_{column}"] = float(finite.std(ddof=0)) if len(finite) else np.nan
        result[f"ac1_{column}"] = _calendar_ac1(values, daily["date"])
        result[f"slope_{column}"] = _calendar_slope(values, daily["date"])
    return pd.DataFrame([result])


def compute_activity_markers(
    data: ParticipantData,
    *,
    include_hourly: bool = False,
) -> MarkerResult:
    """Compute activity-label summaries from motion samples expected at 10 Hz.

    Timezone-aware timestamps are converted to ``data.timezone`` (or UTC when
    omitted) before daily grouping. Naive timestamps are treated as local time.
    """
    seconds = _activity_seconds(data)
    states = ["unknown", *sorted(set(seconds["state"]) - {"unknown"})]
    slugs = _state_slugs(states)
    daily = _daily_rows(seconds, states, slugs, include_hourly)
    markers = _aggregate_daily(daily)
    context = MarkerContext(
        data=data,
        derived={"activity": daily.set_index("date").to_dict("index")},
    )
    return MarkerResult(
        markers=markers,
        daily=daily,
        metadata={
            "marker": "activity",
            "participant_id": data.participant_id,
            "sampling_rate_hz": 10,
            "states": states,
        },
        context=context,
    )


__all__ = ["REQUIREMENTS", "compute_activity_markers"]

"""Compute exertion markers from participant motion and optional heart-rate data."""

import numpy as np
import pandas as pd

from digital_markers.context import MarkerContext
from digital_markers.data.participant import ParticipantData
from digital_markers.data.validation import require_marker_columns
from digital_markers.preprocessing.wear import detect_nonwear
from digital_markers.requirements import MarkerRequirements
from digital_markers.results import MarkerResult

REQUIREMENTS = MarkerRequirements(
    required=("motion",),
    optional=("heart_rate",),
    required_columns=(
        "motion.timestamp",
        "motion.acceleration_x",
        "motion.acceleration_y",
        "motion.acceleration_z",
        "motion.rotation_rate_x",
        "motion.rotation_rate_y",
        "motion.rotation_rate_z",
        "heart_rate.timestamp",
        "heart_rate.heart_rate",
    ),
)

WINDOW_SECONDS = 30
HR_WEIGHT = 0.3
MOTION_WEIGHT = 0.7
MIN_HR = 35
MAX_HR = 220
MOTION_ROBUST_Z_MAX = 6.0
HR_ROBUST_Z_MAX = 4.0
MIN_MOTION_SAMPLES_PER_WINDOW = 5
MIN_HR_SAMPLES_PER_WINDOW = 1
MIN_HR_VALUES_FOR_MARKERS = 21
MOTION_SMOOTH_SECONDS = 5
HR_SMOOTH_SECONDS = 15

MOTION_COLUMNS = (
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "rotation_rate_x",
    "rotation_rate_y",
    "rotation_rate_z",
)
MOTION_REQUIRED_COLUMNS = ("timestamp", *MOTION_COLUMNS)
HEART_RATE_REQUIRED_COLUMNS = ("timestamp", "heart_rate")

DAILY_FIELDS = (
    "daily_motion_exertion",
    "daily_hr_motion_exertion",
    "daily_peak_hr",
    "motion_exertion_per_hour",
    "hr_motion_exertion_per_hour",
    "hr_motion_corr",
    "hr_motion_gain",
    "hr_residual_mean",
    "motion_wear_time",
    "hr_motion_wear_time",
)
HR_FIELDS = (
    "daily_hr_motion_exertion",
    "daily_peak_hr",
    "hr_motion_exertion_per_hour",
    "hr_motion_corr",
    "hr_motion_gain",
    "hr_residual_mean",
    "hr_motion_wear_time",
)
HR_EXERTION_FIELDS = (
    "daily_hr_motion_exertion",
    "hr_motion_exertion_per_hour",
    "hr_motion_corr",
    "hr_motion_gain",
    "hr_residual_mean",
    "hr_motion_wear_time",
)
EXCLUDED_AGGREGATE_FIELDS = {
    "motion_exertion_per_hour",
    "hr_motion_exertion_per_hour",
    "motion_wear_time",
    "hr_motion_wear_time",
}


def robust_zscore(values: pd.Series) -> pd.Series:
    """Compute a median-absolute-deviation robust z-score."""
    values = pd.Series(values, dtype=float)
    median = values.median(skipna=True)
    mad = np.nanmedian(np.abs(values - median))
    if pd.isna(mad) or mad == 0:
        return pd.Series(0.0, index=values.index)
    return 0.6745 * (values - median) / mad


def compute_motion_energy(motion: pd.DataFrame) -> pd.Series:
    """Compute acceleration magnitude plus half the rotation-rate magnitude."""
    acceleration = np.sqrt(
        motion["acceleration_x"] ** 2
        + motion["acceleration_y"] ** 2
        + motion["acceleration_z"] ** 2
    )
    rotation = np.sqrt(
        motion["rotation_rate_x"] ** 2
        + motion["rotation_rate_y"] ** 2
        + motion["rotation_rate_z"] ** 2
    )
    return acceleration + 0.5 * rotation


def clean_heart_rate(
    heart_rate: pd.DataFrame,
    *,
    min_hr: float = MIN_HR,
    max_hr: float = MAX_HR,
    robust_z_max: float = HR_ROBUST_Z_MAX,
) -> pd.DataFrame:
    """Remove implausible and participant-level robust HR outliers."""
    if "heart_rate" not in heart_rate.columns:
        return heart_rate

    cleaned = heart_rate.copy()
    values = pd.to_numeric(cleaned["heart_rate"], errors="coerce")
    values = values.mask((values < min_hr) | (values > max_hr))
    values = values.mask(robust_zscore(values).abs() > robust_z_max)
    cleaned["heart_rate"] = values
    return cleaned


def _smooth_signals(
    motion: pd.DataFrame,
    heart_rate: pd.DataFrame | None,
    *,
    use_short_smoothing: bool = True,
    motion_smooth_seconds: int = MOTION_SMOOTH_SECONDS,
    hr_smooth_seconds: int = HR_SMOOTH_SECONDS,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not use_short_smoothing:
        return motion, heart_rate

    smoothed_motion = motion.copy().set_index("timestamp")
    smoothed_motion["motion_energy"] = smoothed_motion["motion_energy"].rolling(
        f"{motion_smooth_seconds}s", min_periods=1
    ).median()
    smoothed_motion = smoothed_motion.reset_index()

    if heart_rate is None:
        return smoothed_motion, None

    smoothed_hr = heart_rate.copy().set_index("timestamp")
    smoothed_hr["heart_rate"] = smoothed_hr["heart_rate"].rolling(
        f"{hr_smooth_seconds}s", min_periods=1
    ).median()
    return smoothed_motion, smoothed_hr.reset_index()


def exertion_raw(
    window: pd.DataFrame,
    hr_mean: float | None = None,
    hr_std: float | None = None,
    *,
    min_motion_samples_per_window: int = MIN_MOTION_SAMPLES_PER_WINDOW,
    min_hr_samples_per_window: int = MIN_HR_SAMPLES_PER_WINDOW,
    hr_weight: float = HR_WEIGHT,
    motion_weight: float = MOTION_WEIGHT,
) -> tuple[float, float, float, float]:
    """Compute motion-only and combined exertion for one time window."""
    motion = np.nan
    heart_rate_component = np.nan

    if "motion_energy" in window:
        motion_values = window["motion_energy"].dropna()
        if len(motion_values) >= min_motion_samples_per_window:
            raw_motion = motion_values.mean()
            if np.isfinite(raw_motion):
                motion = float(np.tanh(raw_motion))
    elif set(MOTION_COLUMNS).issubset(window.columns):
        raw_motion = compute_motion_energy(window).mean()
        if np.isfinite(raw_motion):
            motion = float(np.tanh(raw_motion))

    if (
        "heart_rate" in window
        and window["heart_rate"].notna().any()
        and hr_mean is not None
        and hr_std is not None
        and hr_std > 0
    ):
        hr_values = window["heart_rate"].dropna()
        if len(hr_values) >= min_hr_samples_per_window:
            raw_hr = hr_values.mean()
            z_score = (raw_hr - hr_mean) / hr_std
            heart_rate_component = float((np.tanh(z_score) + 1.0) / 2.0)

    combined = np.nan
    if np.isfinite(heart_rate_component) and np.isfinite(motion):
        combined = hr_weight * heart_rate_component + motion_weight * motion
    return motion, combined, heart_rate_component, motion


def _summarize_daily_values(values: pd.Series, dates: pd.Series) -> dict[str, float]:
    numbers = values.to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(numbers)
    clean = numbers[valid]
    if not len(clean):
        return {"mean": np.nan, "std": np.nan, "ac1": np.nan, "slope": np.nan}

    dated_values = pd.Series(
        numbers,
        index=pd.DatetimeIndex(pd.to_datetime(dates)).normalize(),
    )
    dated_values = dated_values[np.isfinite(dated_values.to_numpy())]
    dated_values = dated_values.groupby(level=0).mean().sort_index()
    ac1 = np.nan
    if len(dated_values) >= 3:
        dated_numbers = dated_values.to_numpy()
        centered = dated_numbers - dated_numbers.mean()
        denominator = np.sum(centered**2)
        consecutive = np.diff(dated_values.index.to_numpy()) == np.timedelta64(1, "D")
        consecutive_count = int(consecutive.sum())
        if denominator > 0 and consecutive_count >= 2:
            numerator = np.sum((centered[:-1] * centered[1:])[consecutive])
            ac1 = float((numerator / consecutive_count) / (denominator / len(dated_numbers)))

    slope = (
        float(np.polyfit(np.arange(len(numbers))[valid], clean, 1)[0])
        if len(clean) >= 2
        else 0.0
    )
    return {
        "mean": float(clean.mean()),
        "std": float(np.std(clean, ddof=0)),
        "ac1": ac1,
        "slope": slope,
    }


def compute_exertion_markers(
    data: ParticipantData,
    *,
    time_window: str | None = None,
    window_seconds: int = WINDOW_SECONDS,
    hr_weight: float = HR_WEIGHT,
    motion_weight: float = MOTION_WEIGHT,
    min_hr: float = MIN_HR,
    max_hr: float = MAX_HR,
    motion_robust_z_max: float = MOTION_ROBUST_Z_MAX,
    hr_robust_z_max: float = HR_ROBUST_Z_MAX,
    min_motion_samples_per_window: int = MIN_MOTION_SAMPLES_PER_WINDOW,
    min_hr_samples_per_window: int = MIN_HR_SAMPLES_PER_WINDOW,
    min_hr_values_for_markers: int = MIN_HR_VALUES_FOR_MARKERS,
    filter_signals: bool = False,
    use_short_smoothing: bool | None = None,
    motion_smooth_seconds: int = MOTION_SMOOTH_SECONDS,
    hr_smooth_seconds: int = HR_SMOOTH_SECONDS,
) -> MarkerResult:
    """Compute exertion dataframes from a participant's motion and heart-rate frames."""
    if data.motion is None or data.motion.empty:
        raise ValueError("Missing required data for exertion: motion")

    motion = data.motion.copy()
    heart_rate = (
        data.heart_rate.copy()
        if data.heart_rate is not None and not data.heart_rate.empty
        else None
    )
    frames = {"motion": motion}
    if heart_rate is not None:
        frames["heart_rate"] = heart_rate
    require_marker_columns(
        frames,
        REQUIREMENTS.required_columns,
        optional_streams=REQUIREMENTS.optional,
    )

    for name, frame in frames.items():
        if not pd.api.types.is_datetime64_any_dtype(frame["timestamp"]):
            raise ValueError(f"{name}.timestamp must have a datetime dtype before calling exertion")
    for column in MOTION_COLUMNS:
        if not pd.api.types.is_numeric_dtype(motion[column]):
            raise ValueError(f"motion.{column} must have a numeric dtype before calling exertion")
    if heart_rate is not None and not pd.api.types.is_numeric_dtype(heart_rate["heart_rate"]):
        raise ValueError("heart_rate.heart_rate must have a numeric dtype before calling exertion")

    wear_mask = detect_nonwear(motion, rate=10, window_minutes=1)
    motion = motion.loc[wear_mask.to_numpy(), MOTION_REQUIRED_COLUMNS].copy()
    motion = motion.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    motion["motion_energy"] = compute_motion_energy(motion)

    if filter_signals:
        motion_z = robust_zscore(motion["motion_energy"])
        motion.loc[motion_z.abs() > motion_robust_z_max, "motion_energy"] = np.nan

    if heart_rate is not None:
        heart_rate = heart_rate.loc[:, HEART_RATE_REQUIRED_COLUMNS].copy()
        heart_rate = (
            heart_rate.dropna(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        heart_rate["heart_rate"] = pd.to_numeric(heart_rate["heart_rate"], errors="coerce")
        if filter_signals:
            heart_rate = clean_heart_rate(
                heart_rate,
                min_hr=min_hr,
                max_hr=max_hr,
                robust_z_max=hr_robust_z_max,
            )
        valid_hr = heart_rate["heart_rate"].dropna()
    else:
        valid_hr = pd.Series(dtype=float)

    has_sufficient_hr = len(valid_hr) >= min_hr_values_for_markers
    if has_sufficient_hr:
        hr_mean = float(valid_hr.mean())
        hr_std = float(valid_hr.std(ddof=1))
    else:
        hr_mean = hr_std = None

    has_hr_stream = heart_rate is not None
    if heart_rate is not None:
        heart_rate["heart_rate_peak"] = heart_rate["heart_rate"]

    smoothing = filter_signals if use_short_smoothing is None else use_short_smoothing
    motion, heart_rate = _smooth_signals(
        motion,
        heart_rate,
        use_short_smoothing=smoothing,
        motion_smooth_seconds=motion_smooth_seconds,
        hr_smooth_seconds=hr_smooth_seconds,
    )

    motion_signals = motion[["timestamp", "motion_energy"]]
    if heart_rate is None:
        merged = motion_signals.copy()
        merged["heart_rate"] = np.nan
        merged["heart_rate_peak"] = np.nan
    else:
        merged = pd.concat(
            [
                motion_signals,
                heart_rate[["timestamp", "heart_rate", "heart_rate_peak"]],
            ],
            ignore_index=True,
        )
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    rows = []
    windowed_rows = []
    windowed_index = []
    if not merged.empty:
        for timestamp, group in merged.set_index("timestamp").groupby(
            pd.Grouper(freq=time_window or f"{window_seconds}s")
        ):
            motion_exertion, hr_motion_exertion, hr_component, motion_component = exertion_raw(
                group,
                hr_mean=hr_mean,
                hr_std=hr_std,
                min_motion_samples_per_window=min_motion_samples_per_window,
                min_hr_samples_per_window=min_hr_samples_per_window,
                hr_weight=hr_weight,
                motion_weight=motion_weight,
            )
            if np.isfinite(hr_motion_exertion):
                exertion = hr_motion_exertion
            elif np.isfinite(motion_exertion):
                exertion = motion_exertion
            else:
                exertion = np.nan
            windowed_rows.append(
                {
                    "motion_exertion": motion_exertion,
                    "hr_motion_exertion": hr_motion_exertion,
                    "hr_component": hr_component,
                    "motion_component": motion_component,
                    "heart_rate_peak": group["heart_rate_peak"].max(),
                    "exertion": exertion,
                }
            )
            windowed_index.append(timestamp)

    windowed = pd.DataFrame(
        windowed_rows,
        index=pd.DatetimeIndex(windowed_index, name="timestamp"),
        columns=(
            "motion_exertion",
            "hr_motion_exertion",
            "hr_component",
            "motion_component",
            "heart_rate_peak",
            "exertion",
        ),
    )
    if not has_sufficient_hr:
        for column in ("hr_motion_exertion", "hr_component"):
            windowed[column] = pd.Series([None] * len(windowed), index=windowed.index, dtype=object)
    if not has_hr_stream:
        windowed["heart_rate_peak"] = pd.Series(
            [None] * len(windowed), index=windowed.index, dtype=object
        )

    for date, day in windowed.groupby(windowed.index.normalize()):
        motion_values = day["motion_exertion"].dropna()
        if motion_values.empty:
            continue
        hr_values = day["hr_motion_exertion"].dropna() if has_sufficient_hr else pd.Series(dtype=float)
        paired = day.loc[
            day["hr_motion_exertion"].notna(),
            ["hr_component", "motion_component"],
        ]

        hr_motion_corr = hr_motion_gain = hr_residual_mean = np.nan
        if len(paired) >= 3:
            hr_array = paired["hr_component"].to_numpy(dtype=float)
            motion_array = paired["motion_component"].to_numpy(dtype=float)
            hr_motion_corr = float(np.corrcoef(hr_array, motion_array)[0, 1])
            slope, intercept = np.polyfit(motion_array, hr_array, 1)
            hr_motion_gain = float(slope)
            residuals = hr_array - (slope * motion_array + intercept)
            positive_residuals = residuals[residuals > 0]
            hr_residual_mean = float(positive_residuals.mean()) if len(positive_residuals) else 0.0

        motion_wear_time = len(motion_values) * window_seconds
        hr_wear_time = len(hr_values) * window_seconds
        peak_values = day["heart_rate_peak"].dropna()
        peak_values = peak_values[(peak_values >= min_hr) & (peak_values <= max_hr)]
        rows.append(
            {
                "date": date,
                "daily_motion_exertion": float(motion_values.mean()),
                "daily_hr_motion_exertion": float(hr_values.mean()) if len(hr_values) else None,
                "daily_peak_hr": (
                    float(peak_values.max()) if has_hr_stream and len(peak_values) else None
                ),
                "motion_exertion_per_hour": float(motion_values.sum() / motion_wear_time),
                "hr_motion_exertion_per_hour": (
                    float(hr_values.sum() / hr_wear_time) if hr_wear_time else None
                ),
                "hr_motion_corr": hr_motion_corr if len(paired) >= 3 else None,
                "hr_motion_gain": hr_motion_gain if len(paired) >= 3 else None,
                "hr_residual_mean": hr_residual_mean if len(paired) >= 3 else None,
                "motion_wear_time": motion_wear_time,
                "hr_motion_wear_time": hr_wear_time if has_sufficient_hr else None,
            }
        )

    daily = pd.DataFrame(rows, columns=("date", *DAILY_FIELDS))
    daily["daily_peak_hr"] = daily["daily_peak_hr"].astype(object).where(
        daily["daily_peak_hr"].notna(), None
    )
    if has_sufficient_hr:
        for field in HR_FIELDS:
            daily[field] = daily[field].astype(object).where(daily[field].notna(), None)
    else:
        for field in HR_EXERTION_FIELDS:
            daily[field] = pd.Series([None] * len(daily), dtype=object)

    summary = {}
    for field in DAILY_FIELDS:
        if field in EXCLUDED_AGGREGATE_FIELDS:
            continue
        stats = _summarize_daily_values(daily[field], daily["date"])
        for statistic, value in stats.items():
            summary[f"{statistic}_{field}"] = (
                None
                if (
                    field in HR_EXERTION_FIELDS and not has_sufficient_hr
                    or field == "daily_peak_hr" and not has_hr_stream
                )
                else value
            )

    markers = pd.DataFrame([summary])
    context = MarkerContext(data=data, derived={"exertion": windowed["exertion"].to_dict()})
    return MarkerResult(
        markers=markers,
        daily=daily,
        details={"windowed_exertion": windowed},
        metadata={"marker": "exertion"},
        context=context,
    )


__all__ = [
    "REQUIREMENTS",
    "clean_heart_rate",
    "compute_exertion_markers",
    "compute_motion_energy",
    "exertion_raw",
]

import numpy as np
import pandas as pd

from digital_markers.context import MarkerContext
from digital_markers.data.participant import ParticipantData
from digital_markers.data.validation import require_marker_columns
from digital_markers.requirements import MarkerRequirements
from digital_markers.results import MarkerResult

MOTION_REQUIRED_COLUMNS = ("timestamp", "acceleration_x", "acceleration_y", "acceleration_z")
HEART_RATE_REQUIRED_COLUMNS = ("timestamp", "heart_rate")

REQUIREMENTS = MarkerRequirements(
    required=("motion",),
    optional=("heart_rate",),
    required_columns=(
        "motion.timestamp",
        "motion.acceleration_x",
        "motion.acceleration_y",
        "motion.acceleration_z",
    ),
    optional_columns=("heart_rate.timestamp", "heart_rate.heart_rate"),
)

WINDOW_SECONDS = 30
HR_WEIGHT = 0.3
MOTION_WEIGHT = 0.7
MIN_HR = 35
MAX_HR = 220
MOTION_ROBUST_Z_MAX = 6.0
HR_ROBUST_Z_MAX = 4.0
MIN_MOTION_SAMPLES_PER_WINDOW = 1
MIN_HR_SAMPLES_PER_WINDOW = 1
MIN_HR_VALUES_FOR_MARKERS = 21
USE_SHORT_SMOOTHING = True
MOTION_SMOOTH_SECONDS = 5
HR_SMOOTH_SECONDS = 15


def robust_zscore(x: pd.Series) -> pd.Series:
    """Robust z-score using median absolute deviation."""
    med = x.median(skipna=True)
    mad = np.nanmedian(np.abs(x - med))

    if pd.isna(mad) or mad == 0:
        return pd.Series(0.0, index=x.index)

    return 0.6745 * (x - med) / mad


def compute_motion_energy(df: pd.DataFrame) -> pd.Series:
    """Compute acceleration magnitude from canonical axis columns."""
    return np.sqrt(
        df["acceleration_x"] ** 2
        + df["acceleration_y"] ** 2
        + df["acceleration_z"] ** 2
    )


def clean_heart_rate(
    df: pd.DataFrame,
    *,
    min_hr: float = MIN_HR,
    max_hr: float = MAX_HR,
    robust_z_max: float = HR_ROBUST_Z_MAX,
) -> pd.DataFrame:
    """Remove physiologically implausible and extreme participant-level HR values."""
    if "heart_rate" not in df.columns:
        return df

    cleaned = df.copy()
    hr = cleaned["heart_rate"].copy()
    hr[(hr < min_hr) | (hr > max_hr)] = np.nan
    z = robust_zscore(hr)
    hr[np.abs(z) > robust_z_max] = np.nan
    cleaned["heart_rate"] = hr
    return cleaned


def _smooth_signals(
    motion: pd.DataFrame,
    heart_rate: pd.DataFrame | None,
    *,
    use_short_smoothing: bool = USE_SHORT_SMOOTHING,
    motion_smooth_seconds: int = MOTION_SMOOTH_SECONDS,
    hr_smooth_seconds: int = HR_SMOOTH_SECONDS,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not use_short_smoothing:
        return motion, heart_rate

    motion = motion.copy().set_index("timestamp")
    motion["motion_energy"] = motion["motion_energy"].rolling(
        f"{motion_smooth_seconds}s", min_periods=1
    ).median()
    motion = motion.reset_index()
    if heart_rate is None:
        return motion, None

    heart_rate = heart_rate.copy().set_index("timestamp")
    heart_rate["heart_rate"] = heart_rate["heart_rate"].rolling(
        f"{hr_smooth_seconds}s", min_periods=1
    ).median()
    return motion, heart_rate.reset_index()


def _summarize_daily_values(day_values: pd.Series) -> dict[str, float | int | None]:
    clean = day_values.dropna()
    if clean.empty:
        return {"mean": np.nan, "std": np.nan, "ac1": np.nan, "slope": np.nan}

    x = np.arange(len(clean), dtype=float)
    slope = float(np.polyfit(x, clean.to_numpy(), 1)[0]) if len(clean) > 1 else np.nan
    ac1 = float(clean.autocorr(lag=1)) if len(clean) > 2 else np.nan
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if len(clean) > 1 else np.nan,
        "ac1": ac1,
        "slope": slope,
    }


def exertion_raw(
    window: pd.DataFrame,
    hr_mean: float | None = None,
    hr_std: float | None = None,
    *,
    min_motion_samples_per_window: int = MIN_MOTION_SAMPLES_PER_WINDOW,
    min_hr_samples_per_window: int = MIN_HR_SAMPLES_PER_WINDOW,
    hr_weight: float = HR_WEIGHT,
    motion_weight: float = MOTION_WEIGHT,
):
    """Compute window-level motion and heart-rate exertion components."""
    motion = np.nan
    hr: float | None = None

    if "motion_energy" in window.columns:
        motion_series = window["motion_energy"].dropna()
        if len(motion_series) >= min_motion_samples_per_window:
            raw_motion = motion_series.mean()
            if np.isfinite(raw_motion):
                motion = np.tanh(raw_motion)
    if (
        "heart_rate" in window.columns
        and window["heart_rate"].notna().any()
        and hr_mean is not None
        and hr_std is not None
        and hr_std > 0
    ):
        hr_series = window["heart_rate"].dropna()
        if len(hr_series) >= min_hr_samples_per_window:
            raw_hr = hr_series.mean()
            z = (raw_hr - hr_mean) / hr_std
            hr = (np.tanh(z) + 1.0) / 2.0

    motion_exertion = motion if np.isfinite(motion) else np.nan
    if hr is not None and np.isfinite(hr) and np.isfinite(motion):
        hr_motion_exertion = hr_weight * hr + motion_weight * motion
    else:
        hr_motion_exertion = None

    return motion_exertion, hr_motion_exertion, hr, motion


def compute_exertion_markers(
    data: ParticipantData,
    *,
    time_window: str = "30s",
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
    use_short_smoothing: bool = USE_SHORT_SMOOTHING,
    motion_smooth_seconds: int = MOTION_SMOOTH_SECONDS,
    hr_smooth_seconds: int = HR_SMOOTH_SECONDS,
) -> MarkerResult:
    """Compute exertion markers."""
    required = {"motion": data.motion}
    missing = [name for name, value in required.items() if value is None or value.empty]
    if missing:
        raise ValueError(f"Missing required data for exertion: {', '.join(missing)}")

    motion = data.motion.copy()
    heart_rate = data.heart_rate.copy() if data.heart_rate is not None and not data.heart_rate.empty else None
    frames = {"motion": motion}
    if heart_rate is not None:
        frames["heart_rate"] = heart_rate
    require_marker_columns(frames, REQUIREMENTS.required_columns)
    if heart_rate is not None:
        require_marker_columns(frames, REQUIREMENTS.optional_columns)

    for name, frame in frames.items():
        if not pd.api.types.is_datetime64_any_dtype(frame["timestamp"]):
            raise ValueError(f"{name}.timestamp must have a datetime dtype before calling the exertion marker")

    for name in ("acceleration_x", "acceleration_y", "acceleration_z"):
        if not pd.api.types.is_numeric_dtype(motion[name]):
            raise ValueError(f"motion.{name} must have a numeric dtype before calling the exertion marker")
    if heart_rate is not None and not pd.api.types.is_numeric_dtype(heart_rate["heart_rate"]):
        raise ValueError("heart_rate.heart_rate must have a numeric dtype before calling the exertion marker")

    motion = motion.loc[:, MOTION_REQUIRED_COLUMNS].copy()
    if heart_rate is not None:
        heart_rate = heart_rate.loc[:, HEART_RATE_REQUIRED_COLUMNS].copy()

    motion = motion.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if heart_rate is not None:
        heart_rate = heart_rate.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    motion["motion_energy"] = compute_motion_energy(motion)
    rz = robust_zscore(motion["motion_energy"])
    motion.loc[np.abs(rz) > motion_robust_z_max, "motion_energy"] = np.nan

    if heart_rate is not None:
        heart_rate = clean_heart_rate(
            heart_rate,
            min_hr=min_hr,
            max_hr=max_hr,
            robust_z_max=hr_robust_z_max,
        )
        heart_rate["heart_rate_peak"] = heart_rate["heart_rate"]
        valid_hr = heart_rate["heart_rate"].dropna()
    else:
        valid_hr = pd.Series(dtype=float)

    has_sufficient_hr = len(valid_hr) >= min_hr_values_for_markers
    if has_sufficient_hr:
        hr_mean = float(valid_hr.mean())
        hr_std = float(valid_hr.std(ddof=0))
    else:
        hr_mean = None
        hr_std = None

    motion, heart_rate = _smooth_signals(
        motion,
        heart_rate,
        use_short_smoothing=use_short_smoothing,
        motion_smooth_seconds=motion_smooth_seconds,
        hr_smooth_seconds=hr_smooth_seconds,
    )
    if heart_rate is not None:
        merged = motion[["timestamp", "motion_energy"]].merge(
            heart_rate[["timestamp", "heart_rate", "heart_rate_peak"]],
            how="left",
            on="timestamp",
        )
    else:
        merged = motion[["timestamp", "motion_energy"]].copy()
        merged["heart_rate"] = np.nan
        merged["heart_rate_peak"] = np.nan
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    def _window_summary(group: pd.DataFrame):
        motion_exertion, hr_motion_exertion, hr_component, motion_component = exertion_raw(
            group,
            hr_mean=hr_mean,
            hr_std=hr_std,
            min_motion_samples_per_window=min_motion_samples_per_window,
            min_hr_samples_per_window=min_hr_samples_per_window,
            hr_weight=hr_weight,
            motion_weight=motion_weight,
        )
        if hr_motion_exertion is not None and np.isfinite(hr_motion_exertion):
            exertion_value = hr_motion_exertion
        elif np.isfinite(motion_exertion):
            exertion_value = motion_exertion
        else:
            exertion_value = np.nan
        return pd.Series(
            {
                "motion_exertion": motion_exertion,
                "hr_motion_exertion": hr_motion_exertion,
                "hr_component": hr_component,
                "motion_component": motion_component,
                "heart_rate_peak": group["heart_rate_peak"].max(),
                "exertion": exertion_value,
            }
        )

    windowed = (
        merged.set_index("timestamp")
        .resample(time_window)
        .apply(_window_summary)
        .dropna(how="all")
    )

    if not has_sufficient_hr:
        for column in ("hr_motion_exertion", "hr_component", "heart_rate_peak"):
            windowed[column] = pd.Series([None] * len(windowed), index=windowed.index, dtype=object)

    daily = windowed.groupby(windowed.index.normalize()).agg(
        exertion=("exertion", "mean"),
        wear_time=("motion_component", lambda s: int(len(s) * window_seconds)),
        heart_rate_peak=("heart_rate_peak", lambda s: float(s.max()) if s.notna().any() else None),
    )
    daily.index.name = "date"
    daily = daily.reset_index()
    if not has_sufficient_hr:
        daily["heart_rate_peak"] = pd.Series([None] * len(daily), dtype=object)
    daily = daily.sort_values("date").reset_index(drop=True)

    daily_summary = _summarize_daily_values(daily["exertion"])
    hourly = windowed["exertion"].resample("h").mean() if not windowed.empty else pd.Series(dtype=float)
    hourly_summary = _summarize_daily_values(hourly) if not hourly.empty else {"mean": np.nan, "std": np.nan, "ac1": np.nan, "slope": np.nan}

    summary = {
        "num_valid_days": int(daily["exertion"].notna().sum()),
        "mean_daily_exertion": daily_summary["mean"],
        "std_daily_exertion": daily_summary["std"],
        "ac1_daily_exertion": daily_summary["ac1"],
        "slope_daily_exertion": daily_summary["slope"],
        "mean_exertion_per_hour": hourly_summary["mean"],
        "std_exertion_per_hour": hourly_summary["std"],
        "ac1_exertion_per_hour": hourly_summary["ac1"],
        "slope_exertion_per_hour": hourly_summary["slope"],
        "mean_hr_motion_corr": np.nan if has_sufficient_hr else None,
        "mean_hr_motion_gain": np.nan if has_sufficient_hr else None,
        "mean_hr_residual_mean": np.nan if has_sufficient_hr else None,
        "mean_wear_time": float(daily["wear_time"].mean()) if not daily.empty else np.nan,
        "heart_rate_peak": (
            float(daily["heart_rate_peak"].max())
            if has_sufficient_hr and daily["heart_rate_peak"].notna().any()
            else None
        ),
    }

    markers = pd.DataFrame([summary])
    context = MarkerContext(data=data, derived={"exertion": windowed["exertion"].to_dict()})
    return MarkerResult(markers=markers, daily=daily, details={"windowed_exertion": windowed}, metadata={"marker": "exertion"}, context=context)


__all__ = ["compute_exertion_markers", "REQUIREMENTS", "clean_heart_rate", "compute_motion_energy", "exertion_raw"]

import numpy as np
import pandas as pd

from digital_markers.context import MarkerContext
from digital_markers.data.participant import ParticipantData
from digital_markers.requirements import MarkerRequirements
from digital_markers.results import MarkerResult

REQUIREMENTS = MarkerRequirements(required=("motion", "heart_rate"), optional=("demographics.age",))

WINDOW_SECONDS = 30
HR_WEIGHT = 0.3
MOTION_WEIGHT = 0.7
MIN_HR = 35
MAX_HR = 220
MOTION_ROBUST_Z_MAX = 6.0
HR_ROBUST_Z_MAX = 4.0
MIN_MOTION_SAMPLES_PER_WINDOW = 1
MIN_HR_SAMPLES_PER_WINDOW = 1
USE_SHORT_SMOOTHING = True
MOTION_SMOOTH_SECONDS = 5
HR_SMOOTH_SECONDS = 15


def _coerce_timestamp_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce")


def robust_zscore(x: pd.Series) -> pd.Series:
    """Robust z-score using median absolute deviation."""
    x = pd.Series(x, dtype=float)
    med = x.median(skipna=True)
    mad = np.nanmedian(np.abs(x - med))

    if pd.isna(mad) or mad == 0:
        return pd.Series(0.0, index=x.index)

    return 0.6745 * (x - med) / mad


def compute_motion_energy(df: pd.DataFrame) -> pd.Series:
    """Match the legacy alevel motion energy formulation."""
    if {
        "user_acceleration_x",
        "user_acceleration_y",
        "user_acceleration_z",
        "rotation_rate_x",
        "rotation_rate_y",
        "rotation_rate_z",
    }.issubset(df.columns):
        acc = np.sqrt(
            df["user_acceleration_x"] ** 2
            + df["user_acceleration_y"] ** 2
            + df["user_acceleration_z"] ** 2
        )
        rot = np.sqrt(
            df["rotation_rate_x"] ** 2
            + df["rotation_rate_y"] ** 2
            + df["rotation_rate_z"] ** 2
        )
        return acc + 0.5 * rot

    x = pd.to_numeric(df.get("x", 0), errors="coerce")
    y = pd.to_numeric(df.get("y", 0), errors="coerce")
    z = pd.to_numeric(df.get("z", 0), errors="coerce")
    return np.sqrt(x**2 + y**2 + z**2)


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
    hr = pd.to_numeric(cleaned["heart_rate"], errors="coerce")
    hr[(hr < min_hr) | (hr > max_hr)] = np.nan
    z = robust_zscore(hr)
    hr[np.abs(z) > robust_z_max] = np.nan
    cleaned["heart_rate"] = hr
    return cleaned


def _smooth_signals(
    motion: pd.DataFrame,
    heart_rate: pd.DataFrame,
    *,
    use_short_smoothing: bool = USE_SHORT_SMOOTHING,
    motion_smooth_seconds: int = MOTION_SMOOTH_SECONDS,
    hr_smooth_seconds: int = HR_SMOOTH_SECONDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not use_short_smoothing:
        return motion, heart_rate

    motion = motion.copy().set_index("timestamp")
    heart_rate = heart_rate.copy().set_index("timestamp")
    motion["motion_energy"] = motion["motion_energy"].rolling(
        f"{motion_smooth_seconds}s", min_periods=1
    ).median()
    heart_rate["heart_rate"] = heart_rate["heart_rate"].rolling(
        f"{hr_smooth_seconds}s", min_periods=1
    ).median()
    return motion.reset_index(), heart_rate.reset_index()


def _summarize_daily_values(day_values: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(day_values, errors="coerce").dropna()
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
    """Compute the legacy alevel window-level exertion decomposition."""
    motion = np.nan
    hr = np.nan

    if "motion_energy" in window.columns:
        motion_series = pd.to_numeric(window["motion_energy"], errors="coerce").dropna()
        if len(motion_series) >= min_motion_samples_per_window:
            raw_motion = motion_series.mean()
            if np.isfinite(raw_motion):
                motion = np.tanh(raw_motion)
    else:
        motion_cols = {
            "user_acceleration_x",
            "user_acceleration_y",
            "user_acceleration_z",
            "rotation_rate_x",
            "rotation_rate_y",
            "rotation_rate_z",
        }
        if motion_cols.issubset(window.columns):
            raw_motion = compute_motion_energy(window).mean()
            if np.isfinite(raw_motion):
                motion = np.tanh(raw_motion)

    if (
        "heart_rate" in window.columns
        and window["heart_rate"].notna().any()
        and hr_mean is not None
        and hr_std is not None
        and hr_std > 0
    ):
        hr_series = pd.to_numeric(window["heart_rate"], errors="coerce").dropna()
        if len(hr_series) >= min_hr_samples_per_window:
            raw_hr = hr_series.mean()
            z = (raw_hr - hr_mean) / hr_std
            hr = (np.tanh(z) + 1.0) / 2.0

    motion_exertion = motion if np.isfinite(motion) else np.nan
    if np.isfinite(hr) and np.isfinite(motion):
        hr_motion_exertion = hr_weight * hr + motion_weight * motion
    else:
        hr_motion_exertion = np.nan

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
    use_short_smoothing: bool = USE_SHORT_SMOOTHING,
    motion_smooth_seconds: int = MOTION_SMOOTH_SECONDS,
    hr_smooth_seconds: int = HR_SMOOTH_SECONDS,
) -> MarkerResult:
    """Compute exertion markers using the legacy alevel formulation."""
    required = {"motion": data.motion, "heart_rate": data.heart_rate}
    missing = [name for name, value in required.items() if value is None or value.empty]
    if missing:
        raise ValueError(f"Missing required data for exertion: {', '.join(missing)}")

    motion = data.motion.copy()
    heart_rate = data.heart_rate.copy()

    if "timestamp" not in motion.columns:
        raise ValueError("motion data must include a 'timestamp' column")
    if "timestamp" not in heart_rate.columns:
        raise ValueError("heart_rate data must include a 'timestamp' column")

    motion["timestamp"] = _coerce_timestamp_series(motion["timestamp"])
    heart_rate["timestamp"] = _coerce_timestamp_series(heart_rate["timestamp"])
    motion = motion.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    heart_rate = heart_rate.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    motion["motion_energy"] = compute_motion_energy(motion)
    rz = robust_zscore(motion["motion_energy"])
    motion.loc[np.abs(rz) > motion_robust_z_max, "motion_energy"] = np.nan

    heart_rate = clean_heart_rate(
        heart_rate,
        min_hr=min_hr,
        max_hr=max_hr,
        robust_z_max=hr_robust_z_max,
    )
    motion, heart_rate = _smooth_signals(
        motion,
        heart_rate,
        use_short_smoothing=use_short_smoothing,
        motion_smooth_seconds=motion_smooth_seconds,
        hr_smooth_seconds=hr_smooth_seconds,
    )
    hr_mean = float(pd.to_numeric(heart_rate["heart_rate"], errors="coerce").mean())
    hr_std = float(pd.to_numeric(heart_rate["heart_rate"], errors="coerce").std(ddof=0))

    merged = motion[["timestamp", "motion_energy"]].merge(
        heart_rate[["timestamp", "heart_rate"]], how="left", on="timestamp"
    )
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
        if np.isfinite(hr_motion_exertion):
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
                "exertion": exertion_value,
            }
        )

    windowed = (
        merged.set_index("timestamp")
        .resample(time_window)
        .apply(_window_summary)
        .dropna(how="all")
    )

    daily = windowed.groupby(windowed.index.date).agg(
        exertion=("exertion", "mean"),
        wear_time=("motion_component", lambda s: int(len(s) * window_seconds)),
        heart_rate_peak=("hr_component", lambda s: float(pd.to_numeric(s, errors="coerce").max()) if s.notna().any() else np.nan),
    ).reset_index()
    daily = daily.rename(columns={"index": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    daily_summary = _summarize_daily_values(pd.to_numeric(daily["exertion"], errors="coerce"))
    hourly = windowed["exertion"].resample("h").mean() if not windowed.empty else pd.Series(dtype=float)
    hourly_summary = _summarize_daily_values(pd.to_numeric(hourly, errors="coerce")) if not hourly.empty else {"mean": np.nan, "std": np.nan, "ac1": np.nan, "slope": np.nan}

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
        "mean_hr_motion_corr": np.nan,
        "mean_hr_motion_gain": np.nan,
        "mean_hr_residual_mean": np.nan,
        "mean_wear_time": float(daily["wear_time"].mean()) if not daily.empty else np.nan,
        "heart_rate_peak": float(pd.to_numeric(daily["heart_rate_peak"], errors="coerce").max()) if daily["heart_rate_peak"].notna().any() else np.nan,
    }

    markers = pd.DataFrame([summary])
    context = MarkerContext(data=data, derived={"exertion": windowed["exertion"].to_dict()})
    return MarkerResult(markers=markers, daily=daily, details={"windowed_exertion": windowed}, metadata={"marker": "exertion"}, context=context)


__all__ = ["compute_exertion_markers", "REQUIREMENTS", "clean_heart_rate", "compute_motion_energy", "exertion_raw"]

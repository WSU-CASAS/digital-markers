import numpy as np
import pandas as pd

from digital_markers.context import MarkerContext
from digital_markers.data.participant import ParticipantData
from digital_markers.requirements import MarkerRequirements
from digital_markers.results import MarkerResult

REQUIREMENTS = MarkerRequirements(required=("motion", "heart_rate"), optional=("demographics.age",))


def _coerce_timestamp_series(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce")
    return timestamps


def _compute_motion_magnitude(df: pd.DataFrame) -> pd.Series:
    if "accel_magnitude" in df.columns:
        return pd.to_numeric(df["accel_magnitude"], errors="coerce")

    x = pd.to_numeric(df.get("x", 0), errors="coerce")
    y = pd.to_numeric(df.get("y", 0), errors="coerce")
    z = pd.to_numeric(df.get("z", 0), errors="coerce")
    return np.sqrt(x**2 + y**2 + z**2)


def _filter_implausible_hr(series: pd.Series) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce")
    return cleaned.where((cleaned >= 35) & (cleaned <= 220))


def _robust_mad_outlier_mask(values: pd.Series, threshold: float = 5.0) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce")
    clean = clean.dropna()
    if clean.empty:
        return values.isna()

    median = clean.median()
    mad = (clean - median).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(False, index=values.index)

    z = 0.6745 * (clean - median) / mad
    mask = z.abs() > threshold
    result = pd.Series(False, index=values.index)
    result.loc[clean.index] = mask
    return result


def _normalize_hr(hr: pd.Series) -> pd.Series:
    values = pd.to_numeric(hr, errors="coerce")
    if values.dropna().empty:
        return values

    mean = values.mean()
    std = values.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=values.index)
    z = (values - mean) / std
    return np.tanh(z)


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


def compute_exertion_markers(data: ParticipantData, *, time_window: str = "30s") -> MarkerResult:
    """Compute a minimal end-to-end exertion summary for a single participant.

    This implementation intentionally favors a clear, reproducible template over a full
    research-grade algorithm, while still matching the project architecture: validate data,
    clean timestamps, merge motion and HR, compute a daily exertion signal, and summarize it.
    """
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

    motion["motion_mag"] = _compute_motion_magnitude(motion)
    motion["motion_mag"] = motion["motion_mag"].replace([np.inf, -np.inf], np.nan)
    motion["motion_mag"] = motion["motion_mag"].clip(lower=0)

    heart_rate["heart_rate"] = _filter_implausible_hr(heart_rate["heart_rate"])
    heart_rate["outlier"] = _robust_mad_outlier_mask(heart_rate["heart_rate"])
    heart_rate.loc[heart_rate["outlier"], "heart_rate"] = np.nan
    heart_rate["hr_norm"] = _normalize_hr(heart_rate["heart_rate"])

    merged = motion[["timestamp", "motion_mag"]].merge(
        heart_rate[["timestamp", "heart_rate", "hr_norm"]], how="left", on="timestamp"
    )

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged["hr_norm"] = merged["hr_norm"].fillna(0.0)
    merged["motion_mag"] = merged["motion_mag"].fillna(0.0)

    windows = merged.set_index("timestamp").resample(time_window).agg({
        "motion_mag": "mean",
        "hr_norm": "mean",
        "heart_rate": "mean",
    })
    windows = windows.dropna(subset=["motion_mag", "hr_norm", "heart_rate"], how="all")
    windows["exertion"] = windows["motion_mag"] * (0.5 + 0.5 * (windows["hr_norm"].clip(-1.0, 1.0) + 1.0) / 2.0)
    windows["exertion"] = windows["exertion"].replace([np.inf, -np.inf], np.nan)

    daily = windows.groupby(windows.index.date).agg(
        exertion=("exertion", "mean"),
        wear_time=("motion_mag", lambda s: int(len(s) * 30)),
        heart_rate_peak=("heart_rate", "max"),
    ).reset_index()
    daily = daily.rename(columns={"index": "date"})
    daily.columns = ["date", "exertion", "wear_time", "heart_rate_peak"]

    if daily.empty:
        daily = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "exertion": pd.Series(dtype="float64"), "wear_time": pd.Series(dtype="int64"), "heart_rate_peak": pd.Series(dtype="float64")})

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    daily_exertion = pd.to_numeric(daily["exertion"], errors="coerce")
    daily_summary = _summarize_daily_values(daily_exertion)

    if not (windows["exertion"].dropna().empty):
        hourly = windows["exertion"].resample("h").mean()
        hourly_summary = _summarize_daily_values(hourly)
    else:
        hourly_summary = {"mean": np.nan, "std": np.nan, "ac1": np.nan, "slope": np.nan}

    motion_corr = pd.Series(
        merged["motion_mag"].replace(0, np.nan).dropna().to_numpy(),
        index=merged.index,
    )
    hr_corr = pd.Series(
        merged["heart_rate"].dropna().to_numpy(),
        index=merged.index[: len(merged["heart_rate"].dropna())],
    )
    if not motion_corr.empty and not hr_corr.empty:
        corr_series = pd.concat([motion_corr, hr_corr], axis=1)
        corr_series = corr_series.dropna()
        if len(corr_series) > 1:
            mean_hr_motion_corr = float(corr_series.iloc[:, 0].corr(corr_series.iloc[:, 1]))
            slope = np.polyfit(corr_series.iloc[:, 0].to_numpy(), corr_series.iloc[:, 1].to_numpy(), 1)[0]
        else:
            mean_hr_motion_corr = np.nan
            slope = np.nan
    else:
        mean_hr_motion_corr = np.nan
        slope = np.nan

    residual = merged["heart_rate"].fillna(0) - (0.5 * merged["motion_mag"].fillna(0))
    residual_mean = float(residual.mean()) if not residual.empty else np.nan

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
        "mean_hr_motion_corr": mean_hr_motion_corr,
        "mean_hr_motion_gain": slope,
        "mean_hr_residual_mean": residual_mean,
        "mean_wear_time": float(daily["wear_time"].mean()) if not daily.empty else np.nan,
        "heart_rate_peak": float(daily["heart_rate_peak"].max()) if daily["heart_rate_peak"].notna().any() else np.nan,
    }

    markers = pd.DataFrame([summary])
    context = MarkerContext(data=data, derived={"exertion": windows["exertion"].to_dict()})
    return MarkerResult(markers=markers, daily=daily, details={"windowed_exertion": windows}, metadata={"marker": "exertion"}, context=context)


__all__ = ["compute_exertion_markers", "REQUIREMENTS"]

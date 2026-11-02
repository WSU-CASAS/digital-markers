"""Home-location inference and evidence reporting."""

import numpy as np
import pandas as pd


def prepare_fixes(fixes: pd.DataFrame) -> pd.DataFrame:
    """Clean location fixes using the geographic rules used by the reference."""
    prepared = fixes[["timestamp", "latitude", "longitude"]].copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
    prepared["latitude"] = pd.to_numeric(prepared["latitude"], errors="coerce")
    prepared["longitude"] = pd.to_numeric(prepared["longitude"], errors="coerce")
    valid = (
        prepared["timestamp"].notna()
        & prepared["latitude"].between(-90, 90)
        & prepared["longitude"].between(-180, 180)
        & ~((prepared["latitude"].abs() < 1e-6) & (prepared["longitude"].abs() < 1e-6))
    )
    prepared = (
        prepared.loc[valid]
        .sort_values("timestamp", kind="stable")
        .drop_duplicates("timestamp", keep="first")
        .reset_index(drop=True)
    )
    prepared["date"] = prepared["timestamp"].dt.date
    return prepared


def infer_home_location(fixes: pd.DataFrame, decimals: int = 3) -> tuple[float, float]:
    """Infer home from the most consistent rounded nighttime GPS location."""
    hours = fixes["timestamp"].dt.hour.to_numpy()
    candidate_mask = (hours >= 21) | (hours < 6)
    if not np.any(candidate_mask):
        candidate_mask = (hours <= 8) | (hours >= 20)
    if np.any(candidate_mask):
        candidates = fixes.loc[candidate_mask, ["timestamp", "latitude", "longitude"]].copy()
    else:
        first = fixes.groupby("date", sort=False)[["latitude", "longitude"]].first()
        last = fixes.groupby("date", sort=False)[["latitude", "longitude"]].last()
        candidates = pd.concat([first, last], ignore_index=True)

    candidates["lat_q"] = candidates["latitude"].round(decimals)
    candidates["lon_q"] = candidates["longitude"].round(decimals)
    if "timestamp" in candidates:
        candidates["night"] = (candidates["timestamp"] - pd.Timedelta(hours=6)).dt.date
    else:
        candidates["night"] = 0

    nightly_mode = (
        candidates.groupby(["night", "lat_q", "lon_q"], sort=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("rows", ascending=False, kind="stable")
        .drop_duplicates("night")
    )
    counts = (
        nightly_mode.groupby(["lat_q", "lon_q"], sort=False)
        .agg(nights=("night", "size"), rows=("rows", "sum"))
        .reset_index()
        .sort_values(["nights", "rows"], ascending=False, kind="stable")
    )
    winning_lat = counts.iloc[0]["lat_q"]
    winning_lon = counts.iloc[0]["lon_q"]
    home_mask = (candidates["lat_q"] == winning_lat) & (candidates["lon_q"] == winning_lon)
    return (
        float(candidates.loc[home_mask, "latitude"].median()),
        float(candidates.loc[home_mask, "longitude"].median()),
    )


def home_report(fixes: pd.DataFrame, participant_id: str) -> dict[str, object]:
    """Return the inferred home location and supporting evidence."""
    prepared = prepare_fixes(fixes)
    row: dict[str, object] = {
        "id": participant_id,
        "Home_Latitude": np.nan,
        "Home_Longitude": np.nan,
        "GPS_Fixes": len(prepared),
        "Observed_Days": int(prepared["date"].nunique()),
        "Night_Days": 0,
        "Home_Nights": 0,
        "Home_Night_Share": np.nan,
        "Home_Method": "insufficient",
        "Home_Quality": "insufficient",
        "Input_Files": 1,
    }
    if prepared.empty:
        return row

    home_lat, home_lon = infer_home_location(prepared)
    row["Home_Latitude"], row["Home_Longitude"] = home_lat, home_lon
    hours = prepared["timestamp"].dt.hour
    mask = (hours >= 21) | (hours < 6)
    method = "nighttime"
    if not mask.any():
        mask = (hours <= 8) | (hours >= 20)
        method = "extended_hours"
    if not mask.any():
        row["Home_Method"] = "daily_endpoints"
        row["Home_Quality"] = "low"
        return row

    night = prepared.loc[mask].copy()
    night["night"] = (night["timestamp"] - pd.Timedelta(hours=6)).dt.date
    night["lat_q"] = night["latitude"].round(3)
    night["lon_q"] = night["longitude"].round(3)
    row["Night_Days"] = int(night["night"].nunique())
    counts = (
        night.groupby(["night", "lat_q", "lon_q"], sort=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("rows", ascending=False, kind="stable")
        .drop_duplicates("night")
    )
    winning = (round(home_lat, 3), round(home_lon, 3))
    home_nights = int(((counts["lat_q"] == winning[0]) & (counts["lon_q"] == winning[1])).sum())
    share = home_nights / len(counts) if len(counts) else np.nan
    row["Home_Nights"] = home_nights
    row["Home_Night_Share"] = share
    row["Home_Method"] = method
    row["Home_Quality"] = (
        "high"
        if home_nights >= 5 and share >= 0.5 and method == "nighttime"
        else "moderate"
        if home_nights >= 2 and method == "nighttime"
        else "low"
    )
    return row

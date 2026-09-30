import numpy as np
import pandas as pd

from digital_markers.data.validation import require_columns

WEAR_THRESHOLD = 0.5
DEFAULT_RATE = 10
DEFAULT_WINDOW_MINUTES = 1
DEFAULT_ACCEL_THRESHOLD = 0.01
DEFAULT_ROTATION_THRESHOLD = 0.03
WEAR_COLUMNS = (
    "timestamp",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "rotation_rate_x",
    "rotation_rate_y",
    "rotation_rate_z",
)


def _safe_nanstd(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return np.nan
    return float(np.nanstd(values, ddof=0))


def detect_nonwear(
    motion: pd.DataFrame,
    *,
    rate: int = DEFAULT_RATE,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
    accel_threshold: float = DEFAULT_ACCEL_THRESHOLD,
    rotation_threshold: float = DEFAULT_ROTATION_THRESHOLD,
) -> pd.Series:
    """Return a timestamp-aligned boolean mask where True indicates worn."""
    require_columns(motion, set(WEAR_COLUMNS), "motion")
    if not pd.api.types.is_datetime64_any_dtype(motion["timestamp"]):
        raise ValueError("motion.timestamp must have a datetime dtype")
    if rate <= 0 or window_minutes <= 0:
        raise ValueError("rate and window_minutes must be positive")

    if motion.empty:
        return pd.Series(index=motion.index, dtype=bool, name="wear")

    ordered = motion.loc[:, WEAR_COLUMNS].copy()
    ordered["_position"] = np.arange(len(ordered))
    ordered = ordered.sort_values("timestamp", kind="stable")

    acceleration_magnitude = np.sqrt(
        ordered["acceleration_x"] ** 2
        + ordered["acceleration_y"] ** 2
        + ordered["acceleration_z"] ** 2
    )
    rotation_magnitude = np.sqrt(
        ordered["rotation_rate_x"] ** 2
        + ordered["rotation_rate_y"] ** 2
        + ordered["rotation_rate_z"] ** 2
    )

    window_size = max(1, int(rate * window_minutes * 60))
    worn = np.ones(len(ordered), dtype=bool)
    acceleration_values = acceleration_magnitude.to_numpy(dtype=float)
    rotation_values = rotation_magnitude.to_numpy(dtype=float)

    for start in range(0, len(ordered), window_size):
        stop = min(start + window_size, len(ordered))
        accel_std = _safe_nanstd(acceleration_values[start:stop])
        rotation_std = _safe_nanstd(rotation_values[start:stop])
        no_signal = np.isnan(accel_std) and np.isnan(rotation_std)
        low_variability = (
            not np.isnan(accel_std)
            and not np.isnan(rotation_std)
            and accel_std < accel_threshold
            and rotation_std < rotation_threshold
        )
        if no_signal or low_variability:
            worn[start:stop] = False

    worn_by_position = np.empty(len(ordered), dtype=bool)
    worn_by_position[ordered["_position"].to_numpy()] = worn
    return pd.Series(worn_by_position, index=motion.index, name="wear")


def compute_wear_prob(
    motion: pd.DataFrame,
    *,
    rate: int = DEFAULT_RATE,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
    accel_threshold: float = DEFAULT_ACCEL_THRESHOLD,
    rotation_threshold: float = DEFAULT_ROTATION_THRESHOLD,
) -> pd.Series:
    """Return binary wear probability: 1.0 worn, 0.0 non-wear."""
    return detect_nonwear(
        motion,
        rate=rate,
        window_minutes=window_minutes,
        accel_threshold=accel_threshold,
        rotation_threshold=rotation_threshold,
    ).astype("float32").rename("probability")


__all__ = ["WEAR_COLUMNS", "compute_wear_prob", "detect_nonwear"]

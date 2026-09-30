import re
from collections.abc import Mapping

import pandas as pd


_COLUMN_ALIASES = {
    "date_time": "timestamp",
    "datetime": "timestamp",
    "stamp": "timestamp",
    "time_stamp": "timestamp",
    "accel_x": "acceleration_x",
    "accel_y": "acceleration_y",
    "accel_z": "acceleration_z",
    "user_acceleration_x": "acceleration_x",
    "user_acceleration_y": "acceleration_y",
    "user_acceleration_z": "acceleration_z",
    "rotation_x": "rotation_rate_x",
    "rotation_y": "rotation_rate_y",
    "rotation_z": "rotation_rate_z",
    "gyro_x": "rotation_rate_x",
    "gyro_y": "rotation_rate_y",
    "gyro_z": "rotation_rate_z",
    "heartrate": "heart_rate",
    "hr": "heart_rate",
    "bpm": "heart_rate",
}


def _normalize_column_name(name: object) -> str:
    normalized = re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower())
    return normalized.strip("_")


def canonicalize_columns(
    df: pd.DataFrame,
    *,
    aliases: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Normalize column names and map common sensor aliases to canonical names."""
    column_aliases = _COLUMN_ALIASES.copy()
    if aliases:
        column_aliases.update(
            {_normalize_column_name(alias): _normalize_column_name(name) for alias, name in aliases.items()}
        )

    columns = [
        column_aliases.get(normalized, normalized)
        for column in df.columns
        if (normalized := _normalize_column_name(column))
    ]
    if len(columns) != len(df.columns):
        raise ValueError("Column names cannot be empty after canonicalization")

    duplicated = pd.Index(columns)[pd.Index(columns).duplicated()].unique().tolist()
    if duplicated:
        names = ", ".join(sorted(duplicated))
        raise ValueError(f"Column canonicalization creates duplicate names: {names}")

    result = df.copy()
    result.columns = columns
    return result


def require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(sorted(missing))}")


def require_marker_columns(
    frames: Mapping[str, pd.DataFrame],
    required_columns: tuple[str, ...],
    *,
    optional_streams: tuple[str, ...] = (),
) -> None:
    """Validate required columns, skipping absent optional streams."""
    missing_by_stream: dict[str, set[str]] = {}
    for requirement in required_columns:
        stream, separator, column = requirement.partition(".")
        if not separator or not stream or not column:
            raise ValueError(f"Invalid marker column requirement: {requirement!r}")
        frame = frames.get(stream)
        if frame is None and stream in optional_streams:
            continue
        if frame is None or column not in frame.columns:
            missing_by_stream.setdefault(stream, set()).add(column)

    if missing_by_stream:
        messages = [
            f"{stream} data is missing required columns: {', '.join(sorted(columns))}"
            for stream, columns in sorted(missing_by_stream.items())
        ]
        raise ValueError("; ".join(messages))

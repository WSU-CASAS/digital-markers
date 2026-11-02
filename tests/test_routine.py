import numpy as np
import pandas as pd
import pytest

from digital_markers import ParticipantData
from digital_markers.routine import compute_routine_markers


def _routine_motion(days: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    locations = []
    for day in pd.date_range("2026-01-01", periods=days, freq="D"):
        for hour in range(10, 16):
            for minute in range(60):
                seconds = np.arange(310) / 10
                timestamps = (
                    day
                    + pd.Timedelta(hours=hour, minutes=minute)
                    + pd.to_timedelta(seconds, unit="s")
                )
                amplitude = (0.005, 0.05, 0.3)[hour % 3]
                frames.append(
                    pd.DataFrame(
                        {
                            "timestamp": timestamps,
                            "acceleration_x": amplitude * np.sin(2 * np.pi * 0.5 * seconds),
                            "acceleration_y": 0.0,
                            "acceleration_z": 0.0,
                        }
                    )
                )
        locations.append(
            pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        day + pd.Timedelta(hours=10),
                        periods=11,
                        freq="min",
                    ),
                    "latitude": 46.72,
                    "longitude": -117.18,
                }
            )
        )
    return pd.concat(frames, ignore_index=True), pd.concat(locations, ignore_index=True)


def test_routine_markers_return_reference_tables_and_use_optional_gps():
    motion, location = _routine_motion()
    result = compute_routine_markers(
        ParticipantData(participant_id="P001", motion=motion, location=location)
    )

    assert result.markers.loc[0, "n_observed_days"] == 2
    assert result.markers.loc[0, "motion_valid_days"] == 2
    assert result.markers.loc[0, "motion_valid_hours"] == 12
    assert result.markers.loc[0, "motion_day_similarity_mean"] == pytest.approx(1.0)
    assert len(result.details["intervals"]) == 144
    assert result.details["regions"]["region_id"].tolist() == [1]
    assert result.details["location_diagnostics"].loc[0, "region_count"] == 1
    assert result.details["intervals"]["location_seconds"].gt(0).any()
    assert result.details["intervals"]["joint_seconds"].gt(0).any()
    assert result.details["state_balance"].groupby("modality")["fraction"].sum().eq(1).all()


def test_routine_markers_work_without_location_or_gyroscope():
    timestamps = pd.date_range("2026-01-01 10:00", periods=600, freq="100ms")
    motion = pd.DataFrame(
        {
            "timestamp": timestamps,
            "acceleration_x": np.sin(np.arange(len(timestamps)) / 10) * 0.05,
            "acceleration_y": 0.0,
            "acceleration_z": 0.0,
        }
    )

    result = compute_routine_markers(ParticipantData(participant_id="P001", motion=motion))

    assert result.markers.loc[0, "n_observed_days"] == 1
    assert result.details["regions"].empty
    assert not result.details["intervals"]["location_seconds"].any()
    assert result.metadata["has_gyroscope"] is False
    assert result.metadata["has_location"] is False


def test_routine_markers_accept_gps_columns_on_motion():
    timestamps = pd.date_range("2026-01-01 10:00", periods=11, freq="min")
    motion = pd.DataFrame(
        {
            "timestamp": timestamps,
            "acceleration_x": np.zeros(len(timestamps)),
            "acceleration_y": np.zeros(len(timestamps)),
            "acceleration_z": np.zeros(len(timestamps)),
            "latitude": 46.72,
            "longitude": -117.18,
        }
    )

    result = compute_routine_markers(ParticipantData(participant_id="P001", motion=motion))

    assert result.metadata["has_location"] is True
    assert len(result.details["regions"]) == 1


def test_routine_markers_require_acceleration():
    with pytest.raises(ValueError, match="Missing required data for routine"):
        compute_routine_markers(ParticipantData(participant_id="P001"))

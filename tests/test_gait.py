import numpy as np
import pandas as pd
import pytest

from digital_markers import ParticipantData
from digital_markers.gait import compute_gait_markers
from digital_markers.gait.bouts import accepted_bouts, segment_indices
from digital_markers.gait.markers import aggregate_participant_markers


def _walking_motion(
    start: str,
    *,
    seconds: int = 20,
    fs: int = 10,
) -> pd.DataFrame:
    time = np.arange(seconds * fs) / fs
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start,
                periods=len(time),
                freq=pd.Timedelta(seconds=1 / fs),
            ),
            "acceleration_x": 0.5 + 0.4 * np.sin(2 * np.pi * 1.2 * time),
            "acceleration_y": np.zeros(len(time)),
            "acceleration_z": np.zeros(len(time)),
            "rotation_rate_x": np.zeros(len(time)),
            "rotation_rate_y": np.zeros(len(time)),
            "rotation_rate_z": 0.2 * np.sin(2 * np.pi * 0.5 * time),
        }
    )


def test_segment_indices_splits_gaps_and_non_monotonic_timestamps():
    stamps = np.array([0, 100_000_000, 200_000_000, 1_000_000_000, 900_000_000])

    assert segment_indices(stamps, fs=10, max_gap_factor=2.5) == [(0, 3), (3, 4), (4, 5)]


def test_accepted_bouts_apply_conservative_cadence_and_duration_rules():
    motion = _walking_motion("2026-01-01")
    acceleration = motion[["acceleration_x", "acceleration_y", "acceleration_z"]].to_numpy()
    bouts = accepted_bouts(
        acceleration,
        10,
        min_steps=8,
        min_bout_seconds=4,
        max_step_gap_seconds=1.5,
        min_cadence=45,
        max_cadence=180,
        max_interval_cv=0.4,
        prominence_scale=0.5,
    )

    assert len(bouts) == 1
    assert bouts[0]["nsteps"] >= 8
    assert bouts[0]["duration"] >= 4
    assert 45 <= bouts[0]["cadence"] <= 180


def test_package_outputs_reference_daily_fields_and_recording_day_counts():
    motion = pd.concat(
        [_walking_motion("2026-01-01"), _walking_motion("2026-01-02")],
        ignore_index=True,
    )
    result = compute_gait_markers(ParticipantData(participant_id="P001", motion=motion))

    assert list(result.daily.columns) == [
        "date",
        "num_steps",
        "num_walking_bouts",
        "walking_duration_seconds",
        "wear_time_seconds",
        "cadence_spm",
        "step_interval_mean",
        "step_interval_sd",
        "step_interval_cv",
        "dominant_gait_freq",
        "spectral_entropy",
        "acc_rms",
        "gyro_rms",
    ]
    assert len(result.daily) == 2
    assert result.daily["num_walking_bouts"].eq(1).all()
    assert result.markers["num_recorded_days"].iloc[0] == 2
    assert result.markers["num_valid_days"].iloc[0] == 2
    assert result.metadata["sampling_rate_hz"] == 10


def test_gait_requires_acceleration_and_accepts_missing_gyro():
    with pytest.raises(ValueError, match="Missing required data"):
        compute_gait_markers(ParticipantData(participant_id="P001"))

    motion = _walking_motion("2026-01-01").drop(
        columns=["rotation_rate_x", "rotation_rate_y", "rotation_rate_z"]
    )
    result = compute_gait_markers(ParticipantData(participant_id="P001", motion=motion))

    assert result.daily["num_walking_bouts"].iloc[0] == 1
    assert np.isnan(result.daily["gyro_rms"].iloc[0])


def test_participant_aggregation_uses_actual_consecutive_dates():
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04", "2026-01-05"]),
            "num_steps": [10, 20, 40, 50],
            "num_walking_bouts": [1, 1, 1, 1],
            "walking_duration_seconds": [10, 20, 40, 50],
            "wear_time_seconds": [100, 100, 100, 100],
            "cadence_spm": [60, 70, 80, 90],
            "step_interval_mean": [1, 1, 1, 1],
            "step_interval_sd": [0.1, 0.1, 0.1, 0.1],
            "step_interval_cv": [0.1, 0.1, 0.1, 0.1],
            "dominant_gait_freq": [1, 1, 1, 1],
            "spectral_entropy": [1, 1, 1, 1],
            "acc_rms": [0.1, 0.1, 0.1, 0.1],
            "gyro_rms": [0.2, 0.2, 0.2, 0.2],
        }
    )

    result = aggregate_participant_markers(daily)

    assert result["num_recorded_days"] == 4
    assert result["num_valid_days"] == 4
    assert result["mean_num_steps"] == 30
    assert result["slope_num_steps"] == 10
    assert np.isnan(result["ac1_num_steps"])
import numpy as np
import pandas as pd
import pytest

from digital_markers import ParticipantData
from digital_markers.gait import compute_gait_markers
from digital_markers.gait.bouts import count_walking_bouts
from digital_markers.gait.markers import aggregate_daily_markers
from digital_markers.gait.steps import detect_steps


def _walking_motion(day: str, *, duration_seconds: int = 24, fs: int = 50) -> pd.DataFrame:
    time = np.arange(duration_seconds * fs) / fs
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(day, periods=len(time), freq=pd.Timedelta(seconds=1 / fs)),
            "acceleration_x": 0.5 + 0.4 * np.sin(2 * np.pi * 1.8 * time),
            "acceleration_y": 0.0,
            "acceleration_z": 0.0,
            "rotation_rate_x": 0.0,
            "rotation_rate_y": 0.0,
            "rotation_rate_z": 0.2 * np.sin(2 * np.pi * 0.5 * time),
        }
    )


def test_gait_markers_return_package_daily_and_aggregate_results():
    motion = pd.concat(
        [_walking_motion("2026-01-01"), _walking_motion("2026-01-02")],
        ignore_index=True,
    )

    result = compute_gait_markers(ParticipantData(participant_id="P001", motion=motion))

    assert len(result.daily) == 2
    assert result.daily["num_steps"].min() >= 5
    assert result.daily["num_walking_bouts"].eq(1).all()
    assert result.daily["cadence_spm"].between(80, 120).all()
    assert result.markers["num_valid_days"].iloc[0] == 2
    np.testing.assert_allclose(
        result.markers["mean_num_steps"].iloc[0], result.daily["num_steps"].mean()
    )
    assert result.metadata["marker"] == "gait"


def test_gait_markers_validate_required_motion_columns_and_types():
    with pytest.raises(ValueError, match="Missing required data"):
        compute_gait_markers(ParticipantData(participant_id="P001"))

    motion = _walking_motion("2026-01-01")
    motion["timestamp"] = motion["timestamp"].astype(str)
    with pytest.raises(ValueError, match="datetime dtype"):
        compute_gait_markers(ParticipantData(participant_id="P001", motion=motion))


def test_step_detection_and_walking_bout_boundaries():
    motion = _walking_motion("2026-01-01")
    magnitude = np.sqrt(motion["acceleration_x"].to_numpy() ** 2)
    step_times = detect_steps(magnitude, 50)

    assert len(step_times) >= 5
    assert count_walking_bouts(np.array([0, 0.5, 1, 1.5, 2, 5, 5.5, 6, 6.5, 7])) == 2


def test_daily_aggregation_matches_reference_statistics():
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3),
            "num_steps": [10, 20, 30],
            "num_walking_bouts": [1, 2, 3],
        }
    )

    result = aggregate_daily_markers(daily)

    assert result["num_valid_days"] == 3
    assert result["mean_num_steps"] == 20
    assert result["std_num_steps"] == pytest.approx(np.sqrt(200 / 3))
    assert result["ac1_num_steps"] == 0
    assert result["slope_num_steps"] == 10
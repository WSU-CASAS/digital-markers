import pandas as pd

from digital_markers import ParticipantData
from digital_markers.exertion import compute_exertion_markers
from digital_markers.exertion.markers import _smooth_signals


def test_exertion_signals_use_legacy_time_based_median_smoothing():
    motion_timestamps = pd.date_range("2026-01-01", periods=4, freq="1s")
    heart_rate_timestamps = pd.date_range("2026-01-01", periods=4, freq="5s")
    motion = pd.DataFrame({"timestamp": motion_timestamps, "motion_energy": [0.0, 1.0, 100.0, 1.0]})
    heart_rate = pd.DataFrame({"timestamp": heart_rate_timestamps, "heart_rate": [60.0, 80.0, 100.0, 70.0]})

    smoothed_motion, smoothed_heart_rate = _smooth_signals(motion, heart_rate)

    assert smoothed_motion["motion_energy"].tolist() == [0.0, 0.5, 1.0, 1.0]
    assert smoothed_heart_rate["heart_rate"].tolist() == [60.0, 70.0, 80.0, 80.0]


def test_exertion_markers_are_computed_from_motion_and_hr():
    motion = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 00:00:00", periods=6, freq="30s"),
            "x": [0.1, 0.2, 0.5, 1.0, 0.8, 0.3],
            "y": [0.0, 0.1, 0.3, 0.4, 0.5, 0.2],
            "z": [0.0, 0.0, 0.1, 0.3, 0.2, 0.1],
        }
    )
    heart_rate = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 00:00:00", periods=6, freq="30s"),
            "heart_rate": [60.0, 68.0, 76.0, 90.0, 82.0, 66.0],
        }
    )

    data = ParticipantData(participant_id="P001", motion=motion, heart_rate=heart_rate)
    result = compute_exertion_markers(data)

    assert result.markers["num_valid_days"].iloc[0] == 1
    assert result.markers["mean_daily_exertion"].iloc[0] > 0
    assert result.daily["exertion"].notna().all()


def test_exertion_marker_tuning_can_be_overridden():
    timestamps = pd.date_range("2026-01-01", periods=2, freq="30s")
    motion = pd.DataFrame({"timestamp": timestamps, "x": [0.1, 0.5], "y": [0.0, 0.2], "z": [0.0, 0.1]})
    heart_rate = pd.DataFrame({"timestamp": timestamps, "heart_rate": [60.0, 80.0]})
    data = ParticipantData(participant_id="P001", motion=motion, heart_rate=heart_rate)

    result = compute_exertion_markers(
        data,
        window_seconds=45,
        hr_weight=1.0,
        motion_weight=0.0,
        min_hr=30,
        max_hr=200,
        motion_robust_z_max=7.0,
        hr_robust_z_max=5.0,
        min_motion_samples_per_window=1,
        min_hr_samples_per_window=1,
        use_short_smoothing=False,
        motion_smooth_seconds=3,
        hr_smooth_seconds=10,
    )

    windowed = result.details["windowed_exertion"]
    assert windowed["exertion"].iloc[0] == windowed["hr_component"].iloc[0]
    assert result.daily["wear_time"].iloc[0] == len(windowed) * 45

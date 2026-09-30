import pandas as pd

from digital_markers import ParticipantData
from digital_markers.data.validation import canonicalize_columns
from digital_markers.exertion import compute_exertion_markers
from digital_markers.exertion.markers import _smooth_signals


def test_rotation_columns_use_rotation_rate_canonical_names():
    motion = pd.DataFrame({"Rotation X": [0.1], "Gyro Y": [0.2], "rotation_z": [0.3]})

    canonical = canonicalize_columns(motion)

    assert list(canonical.columns) == ["rotation_rate_x", "rotation_rate_y", "rotation_rate_z"]


def test_exertion_signals_use_time_based_median_smoothing():
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
            "acceleration_x": [0.1, 0.2, 0.5, 1.0, 0.8, 0.3],
            "acceleration_y": [0.0, 0.1, 0.3, 0.4, 0.5, 0.2],
            "acceleration_z": [0.0, 0.0, 0.1, 0.3, 0.2, 0.1],
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
    timestamps = pd.date_range("2026-01-01", periods=21, freq="30s")
    motion = pd.DataFrame(
        {
            "timestamp": timestamps,
            "acceleration_x": [0.1 + index * 0.02 for index in range(21)],
            "acceleration_y": [0.0] * 21,
            "acceleration_z": [0.0] * 21,
        }
    )
    heart_rate = pd.DataFrame({"timestamp": timestamps, "heart_rate": list(range(60, 101, 2))})
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


def test_exertion_canonicalizes_required_columns_and_ignores_extras():
    timestamps = pd.date_range("2026-01-01", periods=3, freq="30s")
    motion = pd.DataFrame(
        {
            "Timestamp": timestamps,
            "User Acceleration X": [0.1, 0.5, 0.8],
            "Acceleration Y": [0.0, 0.2, 0.3],
            "Acceleration Z": [0.0, 0.1, 0.2],
            "Rotation X": [100.0, -100.0, 50.0],
            "Rotation Y": [-100.0, 100.0, -50.0],
            "Rotation Z": [50.0, -50.0, 100.0],
            "Yaw": [10.0, 20.0, 30.0],
        }
    )
    heart_rate = pd.DataFrame({"Time Stamp": timestamps, "HR": [60.0, 75.0, 90.0]})
    motion = canonicalize_columns(motion)
    heart_rate = canonicalize_columns(heart_rate)
    result = compute_exertion_markers(
        ParticipantData(participant_id="P001", motion=motion, heart_rate=heart_rate)
    )

    expected_motion = pd.DataFrame(
        {
            "timestamp": timestamps,
            "acceleration_x": [0.1, 0.5, 0.8],
            "acceleration_y": [0.0, 0.2, 0.3],
            "acceleration_z": [0.0, 0.1, 0.2],
        }
    )
    expected_heart_rate = pd.DataFrame({"timestamp": timestamps, "heart_rate": [60.0, 75.0, 90.0]})
    expected = compute_exertion_markers(
        ParticipantData(participant_id="P001", motion=expected_motion, heart_rate=expected_heart_rate)
    )

    pd.testing.assert_frame_equal(result.details["windowed_exertion"], expected.details["windowed_exertion"])


def test_exertion_reports_missing_required_canonical_columns():
    timestamps = pd.date_range("2026-01-01", periods=2, freq="30s")
    motion = pd.DataFrame({"timestamp": timestamps, "acceleration_x": [0.1, 0.5]})
    heart_rate = pd.DataFrame({"timestamp": timestamps, "heart_rate": [60.0, 75.0]})

    try:
        compute_exertion_markers(
            ParticipantData(participant_id="P001", motion=motion, heart_rate=heart_rate)
        )
    except ValueError as exc:
        assert "motion data is missing required columns: acceleration_y, acceleration_z" in str(exc)
    else:
        raise AssertionError("Expected missing motion columns to raise ValueError")


def test_exertion_runs_without_heart_rate_and_returns_none_for_hr_markers():
    timestamps = pd.date_range("2026-01-01", periods=4, freq="30s")
    motion = pd.DataFrame(
        {
            "timestamp": timestamps,
            "acceleration_x": [0.1, 0.3, 0.6, 0.9],
            "acceleration_y": [0.0, 0.1, 0.2, 0.3],
            "acceleration_z": [0.0, 0.0, 0.1, 0.2],
        }
    )

    result = compute_exertion_markers(ParticipantData(participant_id="P001", motion=motion))

    assert result.markers["mean_daily_exertion"].iloc[0] > 0
    assert result.markers["heart_rate_peak"].iloc[0] is None
    assert result.markers["mean_hr_motion_corr"].iloc[0] is None
    assert result.markers["mean_hr_motion_gain"].iloc[0] is None
    assert result.markers["mean_hr_residual_mean"].iloc[0] is None
    assert result.daily["heart_rate_peak"].map(lambda value: value is None).all()
    assert result.details["windowed_exertion"]["hr_motion_exertion"].map(lambda value: value is None).all()
    assert result.details["windowed_exertion"]["hr_component"].map(lambda value: value is None).all()


def test_exertion_returns_none_for_hr_markers_below_sample_threshold():
    timestamps = pd.date_range("2026-01-01", periods=20, freq="30s")
    motion = pd.DataFrame(
        {
            "timestamp": timestamps,
            "acceleration_x": [0.1 + index * 0.02 for index in range(20)],
            "acceleration_y": [0.0] * 20,
            "acceleration_z": [0.0] * 20,
        }
    )
    heart_rate = pd.DataFrame({"timestamp": timestamps, "heart_rate": list(range(60, 100, 2))})

    result = compute_exertion_markers(
        ParticipantData(participant_id="P001", motion=motion, heart_rate=heart_rate)
    )

    assert result.markers["mean_daily_exertion"].iloc[0] > 0
    assert result.markers["heart_rate_peak"].iloc[0] is None
    assert result.daily["heart_rate_peak"].map(lambda value: value is None).all()
    assert result.details["windowed_exertion"]["hr_motion_exertion"].map(lambda value: value is None).all()
    assert result.details["windowed_exertion"]["hr_component"].map(lambda value: value is None).all()


def test_optional_heart_rate_stream_still_requires_its_required_columns():
    timestamps = pd.date_range("2026-01-01", periods=2, freq="30s")
    motion = pd.DataFrame(
        {
            "timestamp": timestamps,
            "acceleration_x": [0.1, 0.5],
            "acceleration_y": [0.0, 0.2],
            "acceleration_z": [0.0, 0.1],
        }
    )
    heart_rate = pd.DataFrame({"timestamp": timestamps})

    try:
        compute_exertion_markers(
            ParticipantData(participant_id="P001", motion=motion, heart_rate=heart_rate)
        )
    except ValueError as exc:
        assert "heart_rate data is missing required columns: heart_rate" in str(exc)
    else:
        raise AssertionError("Expected supplied heart-rate data to require a heart_rate column")


def test_exertion_rejects_unprepared_timestamp_strings():
    timestamps = ["2026-01-01 00:00:00", "2026-01-01 00:00:30"]
    motion = pd.DataFrame(
        {
            "timestamp": timestamps,
            "acceleration_x": [0.1, 0.5],
            "acceleration_y": [0.0, 0.2],
            "acceleration_z": [0.0, 0.1],
        }
    )
    heart_rate = pd.DataFrame({"timestamp": pd.to_datetime(timestamps), "heart_rate": [60.0, 75.0]})

    try:
        compute_exertion_markers(
            ParticipantData(participant_id="P001", motion=motion, heart_rate=heart_rate)
        )
    except ValueError as exc:
        assert "motion.timestamp must have a datetime dtype" in str(exc)
    else:
        raise AssertionError("Expected string timestamps to be rejected")

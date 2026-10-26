import numpy as np
import pandas as pd

from digital_markers import ParticipantData
from digital_markers.data.validation import canonicalize_columns
from digital_markers.exertion import compute_exertion_markers
from digital_markers.exertion.markers import (
    _smooth_signals,
    _summarize_daily_values,
    compute_motion_energy,
)

DAILY_FIELDS = (
    "date", "daily_motion_exertion", "daily_hr_motion_exertion", "daily_peak_hr",
    "motion_exertion_per_hour", "hr_motion_exertion_per_hour", "hr_motion_corr",
    "hr_motion_gain", "hr_residual_mean", "motion_wear_time", "hr_motion_wear_time",
)


def _with_rotation_columns(motion: pd.DataFrame) -> pd.DataFrame:
    motion = motion.copy()
    sample_pattern = np.resize([0.0, 0.1], len(motion))
    for axis, scale in zip(("x", "y", "z"), (1.0, 2.0, 3.0)):
        motion[f"rotation_rate_{axis}"] = sample_pattern * scale
    return motion


def test_rotation_columns_use_rotation_rate_canonical_names():
    motion = pd.DataFrame({"Rotation X": [0.1], "Gyro Y": [0.2], "rotation_z": [0.3]})

    canonical = canonicalize_columns(motion)

    assert list(canonical.columns) == ["rotation_rate_x", "rotation_rate_y", "rotation_rate_z"]


def test_motion_energy_matches_reference_acceleration_and_rotation_formula():
    motion = pd.DataFrame(
        {
            "acceleration_x": [3.0],
            "acceleration_y": [4.0],
            "acceleration_z": [0.0],
            "rotation_rate_x": [0.0],
            "rotation_rate_y": [0.0],
            "rotation_rate_z": [2.0],
        }
    )

    np.testing.assert_allclose(compute_motion_energy(motion), [6.0])


def test_motion_window_requires_five_samples_by_default():
    timestamps = pd.date_range("2026-01-01", periods=5, freq="5s")
    motion = pd.DataFrame(
        {
            "timestamp": timestamps,
            "acceleration_x": [0.1, 0.2, 0.3, 0.4, 0.5],
            "acceleration_y": [0.0] * 5,
            "acceleration_z": [0.0] * 5,
            "rotation_rate_x": [0.2, 0.4, 0.6, 0.8, 1.0],
            "rotation_rate_y": [0.0] * 5,
            "rotation_rate_z": [0.0] * 5,
        }
    )

    result = compute_exertion_markers(ParticipantData(participant_id="P001", motion=motion))

    assert len(result.daily) == 1
    assert result.daily["daily_motion_exertion"].iloc[0] > 0


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
    motion = _with_rotation_columns(motion)
    heart_rate = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 00:00:00", periods=6, freq="30s"),
            "heart_rate": [60.0, 68.0, 76.0, 90.0, 82.0, 66.0],
        }
    )

    data = ParticipantData(participant_id="P001", motion=motion, heart_rate=heart_rate)
    result = compute_exertion_markers(data, min_motion_samples_per_window=1)

    assert len(result.daily) == 1
    assert result.markers["mean_daily_motion_exertion"].iloc[0] > 0
    assert result.daily["daily_motion_exertion"].notna().all()
    assert list(result.daily.columns) == list(DAILY_FIELDS)
    assert list(result.markers.columns) == [
        *(
            f"{stat}_{field}"
            for field in DAILY_FIELDS[1:]
            if field not in {
                "motion_exertion_per_hour",
                "hr_motion_exertion_per_hour",
                "motion_wear_time",
                "hr_motion_wear_time",
            }
            for stat in ("mean", "std", "ac1", "slope")
        ),
    ]


def test_exertion_uses_hr_readings_within_motion_windows():
    timestamps = pd.date_range("2026-01-01", periods=21, freq="30s")
    motion = pd.DataFrame({
        "timestamp": timestamps,
        "acceleration_x": [0.1 + index * 0.02 for index in range(21)],
        "acceleration_y": [0.0] * 21,
        "acceleration_z": [0.0] * 21,
    })
    motion = _with_rotation_columns(motion)
    heart_rate = pd.DataFrame({
        "timestamp": timestamps + pd.Timedelta(seconds=5),
        "heart_rate": list(range(60, 81)),
    })

    result = compute_exertion_markers(
        ParticipantData(participant_id="P001", motion=motion, heart_rate=heart_rate),
        use_short_smoothing=False,
        min_motion_samples_per_window=1,
    )

    assert result.daily["daily_peak_hr"].iloc[0] == 80.0
    assert result.daily["hr_motion_wear_time"].iloc[0] == 21 * 30
    assert result.daily["daily_hr_motion_exertion"].notna().all()


def test_exertion_excludes_detected_nonwear_time():
    count = 1200
    timestamps = pd.date_range("2026-01-01", periods=count, freq="100ms")
    active_acceleration = np.tile([0.0, 0.1], 300)
    active_rotation = np.tile([0.0, 0.2], 300)
    motion = pd.DataFrame(
        {
            "timestamp": timestamps,
            "acceleration_x": np.r_[active_acceleration, np.zeros(600)],
            "acceleration_y": np.zeros(count),
            "acceleration_z": np.zeros(count),
            "rotation_rate_x": np.r_[active_rotation, np.zeros(600)],
            "rotation_rate_y": np.zeros(count),
            "rotation_rate_z": np.zeros(count),
        }
    )

    result = compute_exertion_markers(
        ParticipantData(participant_id="P001", motion=motion),
        use_short_smoothing=False,
    )

    assert result.daily["motion_wear_time"].iloc[0] == 60
    assert result.details["windowed_exertion"]["motion_exertion"].notna().sum() == 2


def test_exertion_excludes_nonwear_heart_rate_from_markers():
    count = 1200
    timestamps = pd.date_range("2026-01-01", periods=count, freq="100ms")
    active_acceleration = np.tile([0.0, 0.1], 300)
    active_rotation = np.tile([0.0, 0.2], 300)
    motion = pd.DataFrame(
        {
            "timestamp": timestamps,
            "acceleration_x": np.r_[active_acceleration, np.zeros(600)],
            "acceleration_y": np.zeros(count),
            "acceleration_z": np.zeros(count),
            "rotation_rate_x": np.r_[active_rotation, np.zeros(600)],
            "rotation_rate_y": np.zeros(count),
            "rotation_rate_z": np.zeros(count),
        }
    )
    heart_rate = pd.DataFrame(
        {
            "timestamp": timestamps,
            "heart_rate": np.r_[np.tile(np.arange(60, 80), 30), np.full(600, 200.0)],
        }
    )

    result = compute_exertion_markers(
        ParticipantData(participant_id="P001", motion=motion, heart_rate=heart_rate),
        use_short_smoothing=False,
    )

    windowed = result.details["windowed_exertion"]
    assert result.daily["daily_peak_hr"].iloc[0] == 79.0
    assert result.daily["hr_motion_wear_time"].iloc[0] == 60
    np.testing.assert_allclose(windowed["hr_component"].iloc[:2], [0.5, 0.5])
    assert windowed["heart_rate_peak"].iloc[:2].max() == 79.0
    assert windowed["heart_rate_peak"].iloc[2:].isna().all()


def test_exertion_daily_and_summary_statistics_follow_reference_formulas():
    timestamps = pd.DatetimeIndex([
        timestamp
        for day in range(3)
        for timestamp in pd.date_range(f"2026-01-{day + 1:02d}", periods=21, freq="30s")
    ])
    motion = pd.DataFrame({
        "timestamp": timestamps,
        "acceleration_x": [0.1 + (index % 21) * 0.01 + (index // 21) * 0.05 for index in range(63)],
        "acceleration_y": [0.0] * 63,
        "acceleration_z": [0.0] * 63,
    })
    motion = _with_rotation_columns(motion)
    heart_rate = pd.DataFrame({
        "timestamp": timestamps + pd.Timedelta(seconds=5),
        "heart_rate": [60.0 + index % 21 + 2 * (index // 21) for index in range(63)],
    })

    result = compute_exertion_markers(
        ParticipantData(participant_id="P001", motion=motion, heart_rate=heart_rate),
        use_short_smoothing=False,
        min_motion_samples_per_window=1,
    )
    daily = result.daily
    summary = result.markers.iloc[0]
    first_windows = result.details["windowed_exertion"].loc["2026-01-01"]
    paired = first_windows[["motion_component", "hr_component"]].dropna()
    gain, intercept = np.polyfit(paired["motion_component"], paired["hr_component"], 1)
    residuals = paired["hr_component"] - (gain * paired["motion_component"] + intercept)

    assert len(daily) == 3
    assert daily["daily_peak_hr"].tolist() == [80.0, 82.0, 84.0]
    assert daily["motion_wear_time"].tolist() == [630, 630, 630]
    assert daily["hr_motion_wear_time"].tolist() == [630, 630, 630]
    np.testing.assert_allclose(daily["motion_exertion_per_hour"], daily["daily_motion_exertion"] / 30)
    np.testing.assert_allclose(
        daily["hr_motion_exertion_per_hour"].to_numpy(dtype=float),
        daily["daily_hr_motion_exertion"].to_numpy(dtype=float) / 30,
    )
    np.testing.assert_allclose(daily["hr_motion_corr"].iloc[0], paired.corr().iloc[0, 1])
    np.testing.assert_allclose(daily["hr_motion_gain"].iloc[0], gain)
    np.testing.assert_allclose(daily["hr_residual_mean"].iloc[0], residuals[residuals > 0].mean())
    np.testing.assert_allclose(summary["mean_daily_motion_exertion"], daily["daily_motion_exertion"].mean())
    np.testing.assert_allclose(summary["std_daily_motion_exertion"], daily["daily_motion_exertion"].std(ddof=0))
    np.testing.assert_allclose(summary["mean_daily_peak_hr"], 82.0)
    np.testing.assert_allclose(summary["std_daily_peak_hr"], np.sqrt(8 / 3))
    np.testing.assert_allclose(summary["ac1_daily_peak_hr"], 0.0)
    np.testing.assert_allclose(summary["slope_daily_peak_hr"], 2.0)


def test_exertion_keeps_motion_day_when_hr_is_missing_that_day():
    timestamps = pd.DatetimeIndex([
        *pd.date_range("2026-01-01", periods=21, freq="30s"),
        *pd.date_range("2026-01-02", periods=21, freq="30s"),
    ])
    motion = pd.DataFrame({
        "timestamp": timestamps,
        "acceleration_x": [0.1 + index * 0.01 for index in range(42)],
        "acceleration_y": [0.0] * 42,
        "acceleration_z": [0.0] * 42,
    })
    motion = _with_rotation_columns(motion)
    heart_rate = pd.DataFrame({
        "timestamp": timestamps[:21] + pd.Timedelta(seconds=5),
        "heart_rate": list(range(60, 81)),
    })

    result = compute_exertion_markers(
        ParticipantData(participant_id="P001", motion=motion, heart_rate=heart_rate),
        use_short_smoothing=False,
        min_motion_samples_per_window=1,
    )

    assert len(result.daily) == 2
    assert result.daily["daily_motion_exertion"].notna().all()
    assert result.daily["daily_hr_motion_exertion"].iloc[0] is not None
    assert result.daily["daily_hr_motion_exertion"].iloc[1] is None
    assert result.daily["daily_peak_hr"].iloc[1] is None
    assert result.daily["hr_motion_wear_time"].iloc[1] == 0


def test_exertion_has_stable_outputs_without_valid_motion_windows():
    motion = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=2, freq="30s"),
        "acceleration_x": [np.nan, np.nan],
        "acceleration_y": [0.0, 0.0],
        "acceleration_z": [0.0, 0.0],
    })
    motion = _with_rotation_columns(motion)

    result = compute_exertion_markers(
        ParticipantData(participant_id="P001", motion=motion),
        min_motion_samples_per_window=1,
    )

    assert list(result.daily.columns) == list(DAILY_FIELDS)
    assert result.daily.empty
    assert result.daily.empty
    assert np.isnan(result.markers["mean_daily_motion_exertion"].iloc[0])
    assert result.markers["mean_daily_hr_motion_exertion"].iloc[0] is None


def test_exertion_summary_autocorrelation_uses_consecutive_calendar_days():
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    dates = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04", "2026-01-05"]))

    summary = _summarize_daily_values(values, dates)

    np.testing.assert_allclose(summary["ac1"], 0.6)


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
    motion = _with_rotation_columns(motion)
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
    assert result.daily["motion_wear_time"].iloc[0] == len(windowed) * 45
    assert result.daily["hr_motion_wear_time"].iloc[0] == len(windowed) * 45


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
            "rotation_rate_x": [100.0, -100.0, 50.0],
            "rotation_rate_y": [-100.0, 100.0, -50.0],
            "rotation_rate_z": [50.0, -50.0, 100.0],
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
        assert "motion data is missing required columns: acceleration_y, acceleration_z, rotation_rate_x, rotation_rate_y, rotation_rate_z" in str(exc)
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
    motion = _with_rotation_columns(motion)

    result = compute_exertion_markers(
        ParticipantData(participant_id="P001", motion=motion),
        min_motion_samples_per_window=1,
    )

    assert result.markers["mean_daily_motion_exertion"].iloc[0] > 0
    assert result.markers["mean_daily_peak_hr"].iloc[0] is None
    assert result.markers["mean_hr_motion_corr"].iloc[0] is None
    assert result.markers["mean_hr_motion_gain"].iloc[0] is None
    assert result.markers["mean_hr_residual_mean"].iloc[0] is None
    assert result.daily["daily_peak_hr"].map(lambda value: value is None).all()
    assert result.daily["hr_motion_wear_time"].map(lambda value: value is None).all()
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
    motion = _with_rotation_columns(motion)
    heart_rate = pd.DataFrame({"timestamp": timestamps, "heart_rate": list(range(60, 100, 2))})

    result = compute_exertion_markers(
        ParticipantData(participant_id="P001", motion=motion, heart_rate=heart_rate),
        min_motion_samples_per_window=1,
    )

    assert result.markers["mean_daily_motion_exertion"].iloc[0] > 0
    assert result.markers["mean_daily_peak_hr"].iloc[0] == 98.0
    assert result.daily["daily_peak_hr"].iloc[0] == 98.0
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
    motion = _with_rotation_columns(motion)
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
    motion = _with_rotation_columns(motion)
    heart_rate = pd.DataFrame({"timestamp": pd.to_datetime(timestamps), "heart_rate": [60.0, 75.0]})

    try:
        compute_exertion_markers(
            ParticipantData(participant_id="P001", motion=motion, heart_rate=heart_rate)
        )
    except ValueError as exc:
        assert "motion.timestamp must have a datetime dtype" in str(exc)
    else:
        raise AssertionError("Expected string timestamps to be rejected")

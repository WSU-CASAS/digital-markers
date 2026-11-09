import numpy as np
import pandas as pd
import pytest

from digital_markers import ParticipantData
from digital_markers.activity import compute_activity_markers
from digital_markers.data.validation import canonicalize_columns


def _activity_frame(start: str = "2026-01-01 06:00:00") -> pd.DataFrame:
    labels = ["Relax"] * 20 + ["Walk"] * 20 + [None] * 10
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=len(labels), freq="100ms"),
            "activity": labels,
        }
    )


def test_activity_markers_compute_daily_state_and_transition_measures():
    result = compute_activity_markers(
        ParticipantData(participant_id="P001", activity=_activity_frame()),
        include_hourly=True,
    )
    daily = result.daily.iloc[0]

    assert daily["observed_seconds"] == 5
    assert daily["unknown_fraction"] == pytest.approx(0.2)
    assert daily["num_states"] == 2
    assert daily["transitions"] == 1
    assert daily["transitions_per_hour"] == pytest.approx(900)
    assert daily["activity_entropy"] == pytest.approx(1)
    assert daily["dominant_fraction"] == pytest.approx(0.5)
    assert daily["relax_fraction"] == pytest.approx(0.5)
    assert daily["walk_fraction"] == pytest.approx(0.5)
    assert daily["relax_bouts"] == 1
    assert daily["relax_bout_mean_seconds"] == 2
    assert daily["walk_bouts"] == 1
    assert daily["walk_bout_mean_seconds"] == 2
    assert daily["relax_morning_fraction"] == pytest.approx(0.4)
    assert daily["relax_morning_share"] == pytest.approx(1)
    assert daily["relax_hour_06_fraction"] == pytest.approx(0.4)
    assert np.isnan(daily["relax_night_fraction"])
    assert result.markers.loc[0, "num_recorded_days"] == 1
    assert result.markers.loc[0, "mean_unknown_fraction"] == pytest.approx(0.2)
    assert result.metadata["states"] == ["unknown", "relax", "walk"]


def test_activity_counts_unique_100ms_slots_and_does_not_bridge_gaps():
    timestamps = pd.to_datetime(
        [
            "2026-01-01 12:00:00.000",
            "2026-01-01 12:00:00.000",
            "2026-01-01 12:00:00.100",
            "2026-01-01 12:00:02.000",
            "2026-01-01 12:00:02.100",
        ]
    )
    motion = pd.DataFrame(
        {"timestamp": timestamps, "activity": ["walk", "rest", "walk", "walk", "walk"]}
    )

    result = compute_activity_markers(ParticipantData(participant_id="P001", activity=motion))
    daily = result.daily.iloc[0]

    assert daily["observed_seconds"] == pytest.approx(0.4)
    assert daily["walk_bouts"] == 2
    assert daily["transitions"] == 0


def test_activity_converts_timezone_aware_timestamps_to_participant_timezone():
    activity = _activity_frame("2026-01-02 07:00:00+00:00")
    result = compute_activity_markers(
        ParticipantData(
            participant_id="P001",
            timezone="America/Los_Angeles",
            activity=activity,
        )
    )

    assert result.daily["date"].tolist() == [pd.Timestamp("2026-01-01")]


def test_activity_requires_datetime_and_activity_columns():
    with pytest.raises(ValueError, match="activity data is missing required columns"):
        compute_activity_markers(
            ParticipantData(
                participant_id="P001",
                activity=pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=1)}),
            )
        )

    with pytest.raises(ValueError, match="activity.timestamp.*datetime dtype"):
        compute_activity_markers(
            ParticipantData(
                participant_id="P001",
                activity=pd.DataFrame({"timestamp": ["2026-01-01"], "activity": ["walk"]}),
            )
        )


def test_activity_requires_activity_column_without_label_alias():
    motion = pd.DataFrame(
        {
            "stamp": pd.date_range("2026-01-01", periods=2, freq="100ms"),
            "user_activity_label": ["walking", "walking"],
        }
    )

    canonical = canonicalize_columns(motion)

    assert "activity" not in canonical
    with pytest.raises(ValueError, match="activity data is missing required columns: activity"):
        compute_activity_markers(ParticipantData(participant_id="P001", activity=motion))

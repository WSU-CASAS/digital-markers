import numpy as np
import pandas as pd
import pytest

from digital_markers import ParticipantData
from digital_markers.location import compute_location_markers


def _nighttime_location_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    location_times = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-01 21:00"),
            pd.Timestamp("2026-01-01 21:19"),
            pd.Timestamp("2026-01-02 21:00"),
            pd.Timestamp("2026-01-02 21:19"),
        ]
    )
    location = pd.DataFrame(
        {
            "timestamp": location_times,
            "latitude": [46.72, 46.72, 46.72, 46.72],
            "longitude": [-117.18, -117.18, -117.18, -117.18],
        }
    )
    motion = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(
                [
                    *pd.date_range("2026-01-01 21:00", periods=20, freq="min"),
                    *pd.date_range("2026-01-02 21:00", periods=20, freq="min"),
                ]
            )
        }
    )
    return location, motion


def test_location_outputs_return_reference_tables_as_dataframes():
    location, motion = _nighttime_location_data()

    result = compute_location_markers(
        ParticipantData(
            participant_id="P001",
            location=location,
            motion=motion,
        )
    )

    assert len(result.markers) == 1
    assert result.markers.loc[0, "id"] == "P001"
    assert result.markers.loc[0, "num_valid_days"] == 2
    assert list(result.daily.columns[:13]) == [
        "date",
        "total_distance",
        "max_distance_from_home",
        "mean_distance_from_home",
        "observed_time",
        "away_time",
        "first_time",
        "last_time",
        "num_points",
        "span",
        "home_time",
        "time_at_home_ratio",
        "num_distinct_locations",
    ]
    assert result.daily["n_recorded"].tolist() == [20, 20]
    assert result.daily["num_points"].tolist() == [20, 20]
    assert result.details["home_locations"].loc[0, "Home_Nights"] == 2
    assert result.details["staypoints"]["num_visits"].tolist() == [2]
    assert result.details["staypoints"]["total_duration"].tolist() == [2400.0]
    assert np.isclose(result.markers.loc[0, "location_entropy"], 0.0)


def test_location_requires_gps_stream():
    with pytest.raises(ValueError, match="Missing required data for location"):
        compute_location_markers(ParticipantData(participant_id="P001"))


def test_location_home_inference_handles_no_fix_without_silent_fallback():
    location = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 10:00"]),
            "latitude": [np.nan],
            "longitude": [np.nan],
        }
    )

    result = compute_location_markers(
        ParticipantData(participant_id="P001", location=location)
    )

    assert result.markers.empty
    assert result.daily.empty
    assert result.details["home_locations"].loc[0, "Home_Quality"] == "insufficient"

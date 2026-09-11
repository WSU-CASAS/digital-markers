import pandas as pd

from digital_markers import ParticipantData
from digital_markers.exertion import compute_exertion_markers


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

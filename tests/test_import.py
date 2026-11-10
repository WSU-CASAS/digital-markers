from importlib.metadata import version

import pandas as pd

import digital_markers as dm


def test_version():
    assert dm.__version__ == version("digital-markers")


def test_participant_data():
    data = dm.ParticipantData(participant_id="P001")
    assert data.participant_id == "P001"


def test_supported_marker_imports():
    from digital_markers.audio import compute_audio_markers
    from digital_markers.exertion import compute_exertion_markers
    from digital_markers.gait import compute_gait_markers
    from digital_markers.location import compute_location_markers
    from digital_markers.routine import compute_routine_markers

    assert all(
        callable(marker)
        for marker in (
            compute_exertion_markers,
            compute_audio_markers,
            compute_gait_markers,
            compute_location_markers,
            compute_routine_markers,
        )
    )


def test_participant_data_canonicalizes_sensor_column_aliases():
    motion = pd.DataFrame(
        {
            "stamp": pd.to_datetime(["2026-01-01"]),
            "user_acceleration_x": [0.1],
            "user_acceleration_y": [0.2],
            "user_acceleration_z": [0.3],
        }
    )
    heart_rate = pd.DataFrame({"stamp": pd.to_datetime(["2026-01-01"]), "hr": [70.0]})
    location = pd.DataFrame(
        {
            "stamp": pd.to_datetime(["2026-01-01"]),
            "latitude": [46.7],
            "longitude": [-117.2],
        }
    )

    participant = dm.ParticipantData(
        participant_id="P001",
        motion=motion,
        heart_rate=heart_rate,
        location=location,
    )

    assert list(participant.motion.columns) == [
        "timestamp",
        "acceleration_x",
        "acceleration_y",
        "acceleration_z",
    ]
    assert list(participant.heart_rate.columns) == ["timestamp", "heart_rate"]
    assert list(participant.location.columns) == ["timestamp", "latitude", "longitude"]
    assert list(motion.columns) == [
        "stamp",
        "user_acceleration_x",
        "user_acceleration_y",
        "user_acceleration_z",
    ]

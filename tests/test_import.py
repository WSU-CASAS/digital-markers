from importlib.metadata import version

import digital_markers as dm


def test_version():
    assert dm.__version__ == version("digital-markers")


def test_participant_data():
    data = dm.ParticipantData(participant_id="P001")
    assert data.participant_id == "P001"


def test_supported_marker_imports():
    from digital_markers.exertion import compute_exertion_markers
    from digital_markers.gait import compute_gait_markers
    from digital_markers.location import compute_location_markers
    from digital_markers.routine import compute_routine_markers

    assert all(
        callable(marker)
        for marker in (
            compute_exertion_markers,
            compute_gait_markers,
            compute_location_markers,
            compute_routine_markers,
        )
    )

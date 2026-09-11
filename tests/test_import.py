from importlib.metadata import version

import digital_markers as dm

def test_version():
    assert dm.__version__ == version("digital-markers")

def test_participant_data():
    data = dm.ParticipantData(participant_id="P001", demographics=dm.Demographics(age=72))
    assert data.demographics.age == 72

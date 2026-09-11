import digital_markers as dm

def test_version():
    assert dm.__version__ == "0.0.1"

def test_participant_data():
    data = dm.ParticipantData(participant_id="P001", demographics=dm.Demographics(age=72))
    assert data.demographics.age == 72

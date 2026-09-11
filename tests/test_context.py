from digital_markers import ParticipantData
from digital_markers.context import MarkerContext

def test_derived_data_is_separate():
    data = ParticipantData(participant_id="P001")
    context = MarkerContext(data)
    context.set_derived("activity_labels", "model-output")
    assert context.get_derived("activity_labels") == "model-output"
    assert not hasattr(data, "derived")
